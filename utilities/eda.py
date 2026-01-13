"""
eda_helpers.py

Practical helpers for quick exploratory data analysis (EDA).

Included:
- check_df: quick structural overview (shape, dtypes, preview, missingness, duplicates, optional describe)
- cat_summary: counts + proportions for a categorical column (+ optional count plot)
- num_summary: descriptive stats for a numeric column (+ optional histogram/box plot)
- grab_col_names: split columns into categorical, numeric, and categorical-but-cardinal buckets
- high_correlated_cols: find highly correlated numeric columns (+ optional heatmap + optional save)

Design notes:
- Plotting libs are imported only when needed.
- Functions validate inputs and fail loudly (better than silent wrong results).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd


# -------------------------
# Internal helpers
# -------------------------
def _require_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing column(s): {missing}")


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


# -------------------------
# Public API
# -------------------------
def check_df(df: pd.DataFrame, head: int = 5, tail: int = 5, detail: bool = False) -> None:
    """Print a concise overview of a DataFrame for EDA sanity-checking."""
    na = df.isna().sum()
    total_na = int(na.sum())
    dup = int(df.duplicated().sum())

    print("##################### Index #####################")
    print(df.index)
    print("##################### Shape #####################")
    print(df.shape)
    print("##################### Types #####################")
    print(df.dtypes)
    print("##################### Head #####################")
    print(df.head(head))
    print("##################### Tail #####################")
    print(df.tail(tail))
    print("##################### Duplicates #####################")
    print(f"Any duplicates: {dup > 0} (rows duplicated: {dup})")
    print("##################### Missing values #####################")
    print(na[na.gt(0)].sort_values(ascending=False) if total_na else "No missing values")
    print("##################### Total missing #####################")
    print(total_na)

    if detail:
        print("##################### Describe (all) #####################")
        print(df.describe(include="all").T)


def cat_summary(
    df: pd.DataFrame,
    col: str,
    plot: bool = False,
    figsize: Tuple[int, int] = (5, 3),
    dropna: bool = False,
) -> None:
    """
    Print value counts and proportions for a categorical column.
    If plot=True, show a count plot ordered by frequency.
    """
    _require_columns(df, [col])

    counts = df[col].value_counts(dropna=dropna)
    ratio = (counts / len(df) * 100).round(2)
    summary = pd.DataFrame({col: counts, "Ratio (%)": ratio})

    print(summary)
    print("##########################################")

    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns

        series = df[col]
        if not dropna:
            series = series.fillna("NaN")

        plt.figure(figsize=figsize)
        sns.countplot(x=series, order=series.value_counts().index)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()


def num_summary(
    df: pd.DataFrame,
    col: str,
    plot: bool = False,
    plot_type: str = "hist",
    figsize: Tuple[int, int] = (5, 3),
    bins: int = 30,
) -> None:
    """Print descriptive stats for a numeric column, plus optional plot."""
    _require_columns(df, [col])
    s = df[col]

    if not _is_numeric(s):
        raise TypeError(f"Column '{col}' must be numeric for num_summary (got {s.dtype}).")

    print(f"##################### {col} #####################")
    print("##################### Describe #####################")
    print(s.describe(), "\n")
    print("##################### Missing #####################")
    print(f"{col} NA: {int(s.isna().sum())} | Total NA in df: {int(df.isna().sum().sum())}")

    if not plot:
        return

    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_type = plot_type.strip().lower()
    plt.figure(figsize=figsize)

    if plot_type in {"hist", "histogram"}:
        s.plot(kind="hist", bins=bins)
        plt.xlabel(col)
        plt.title(f"{col} histogram")
    elif plot_type in {"box", "boxplot", "box_plot"}:
        sns.boxplot(x=s)
        plt.xlabel(col)
        plt.title(f"{col} box plot")
    else:
        raise ValueError("plot_type must be 'hist' or 'boxplot'.")

    plt.tight_layout()
    plt.show()


def grab_col_names(df: pd.DataFrame, cat_th: int = 10, car_th: int = 20) -> Tuple[List[str], List[str], List[str]]:
    """
    Return (cat_cols, num_cols, cat_but_car).

    Rules:
    - Object columns are categorical.
    - Numeric columns with low unique counts (< cat_th) are treated as categorical (num_but_cat).
    - Object columns with high unique counts (> car_th) are treated as cardinal (cat_but_car).
    """
    dtypes = df.dtypes
    is_obj = dtypes.eq("O")

    # Compute once (faster + cleaner)
    nunique = df.nunique(dropna=False)

    cat_cols = df.columns[is_obj].tolist()
    num_but_cat = df.columns[(~is_obj) & (nunique.lt(cat_th))].tolist()
    cat_but_car = df.columns[is_obj & (nunique.gt(car_th))].tolist()

    cat_cols = [c for c in (cat_cols + num_but_cat) if c not in set(cat_but_car)]
    num_cols = [c for c in df.columns[~is_obj].tolist() if c not in set(num_but_cat)]

    print(f"Observations: {df.shape[0]}")
    print(f"Variables: {df.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


def high_correlated_cols(
    df: pd.DataFrame,
    plot: bool = False,
    corr_th: float = 0.90,
    method: str = "pearson",
    fig_id: Optional[str] = None,
    images_path: Optional[Path | str] = None,
) -> List[str]:
    """
    Return a list of numeric columns that have |corr| > corr_th with any other numeric column.

    If plot=True, show a heatmap of the numeric correlation matrix.
    If fig_id and images_path are provided, save the figure as PNG.
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return []

    corr = num_df.corr(method=method)
    abs_corr = corr.abs()

    upper = abs_corr.where(np.triu(np.ones(abs_corr.shape, dtype=bool), k=1))
    drop_list = upper.columns[upper.gt(corr_th).any()].tolist()

    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(15, 15))
        sns.heatmap(corr, cmap="RdBu", center=0, annot=False)
        plt.title("Correlation heatmap")

        if fig_id and images_path is not None:
            out_dir = Path(images_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{fig_id}.png"
            plt.tight_layout()
            plt.savefig(out_file, dpi=300, format="png")

        plt.tight_layout()
        plt.show()

    return drop_list

