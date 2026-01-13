"""
eda_utils.py

Lightweight helpers for quick exploratory data analysis (EDA) on pandas DataFrames.

Functions included:
- check_df: quick structural overview (shape, dtypes, missingness, duplicates, preview, describe)
- cat_summary: counts/ratios for a categorical column (+ optional count plot)
- num_summary: descriptive stats for a numeric column (+ optional histogram/box plot)
- grab_col_names: split columns into categorical, numeric, and categorical-but-cardinal buckets
- high_correlated_cols: find highly correlated numeric columns (+ optional heatmap)
- outlier_thresholds / check_outlier / grab_outliers / remove_outlier: IQR-based outlier utilities
- missing_values_table: missing-value counts and ratios per column
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple, List

import numpy as np
import pandas as pd


# ----------------------------
# Small internal helpers
# ----------------------------
def _require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Column(s) not found in DataFrame: {missing}")


def _as_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    _require_columns(df, [col])
    s = df[col]
    if not pd.api.types.is_numeric_dtype(s):
        raise TypeError(f"Column '{col}' must be numeric, got dtype={s.dtype}")
    return s


# ----------------------------
# Public API
# ----------------------------
def check_df(df: pd.DataFrame, head: int = 5, tail: int = 5, detail: bool = False) -> None:
    """Print a compact, practical overview of a DataFrame."""
    na_counts = df.isna().sum()
    total_na = int(na_counts.sum())
    dup_count = int(df.duplicated().sum())

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
    print(f"Any duplicates: {dup_count > 0} (rows duplicated: {dup_count})")
    print("##################### Missing values #####################")
    print(na_counts[na_counts.gt(0)].sort_values(ascending=False) if total_na else "No missing values")
    print("##################### Total missing #####################")
    print(total_na)

    if detail:
        print("##################### Describe (all) #####################")
        # include='all' can be slow for very wide frames; keep it behind the flag
        print(df.describe(include="all").T)


def cat_summary(
    df: pd.DataFrame,
    col_name: str,
    plot: bool = False,
    figsize: Tuple[int, int] = (5, 3),
    dropna: bool = False,
) -> None:
    """Summarise a categorical column with counts and ratios (optionally plot a count chart)."""
    _require_columns(df, [col_name])

    counts = df[col_name].value_counts(dropna=dropna)
    ratios = (counts / len(df)).mul(100).round(2)

    summary = pd.DataFrame({col_name: counts, "Ratio (%)": ratios})
    print(summary)
    print("##########################################")

    if plot:
        # Import lazily to avoid forcing plotting deps for non-plot users
        import matplotlib.pyplot as plt
        import seaborn as sns

        plot_series = df[col_name]
        if not dropna:
            plot_series = plot_series.fillna("NaN")

        plt.figure(figsize=figsize)
        sns.countplot(x=plot_series, order=plot_series.value_counts().index)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()


def num_summary(
    df: pd.DataFrame,
    numerical_col: str,
    plot: bool = False,
    plot_type: str = "hist",
    figsize: Tuple[int, int] = (5, 3),
    bins: int = 30,
) -> None:
    """Summarise a numeric column (optionally plot histogram or box plot)."""
    s = _as_numeric_series(df, numerical_col)

    print("##################### Describe #####################")
    print(s.describe(), "\n")
    col_na = int(s.isna().sum())
    total_na = int(df.isna().sum().sum())
    print("##################### Missing #####################")
    print(f"{numerical_col} NA: {col_na} | Total NA in df: {total_na}")

    if not plot:
        return

    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_type = plot_type.strip().lower()
    plt.figure(figsize=figsize)

    if plot_type in {"hist", "histogram"}:
        s.plot(kind="hist", bins=bins)
        plt.xlabel(numerical_col)
        plt.title(f"{numerical_col} histogram")
    elif plot_type in {"box", "boxplot", "box_plot"}:
        sns.boxplot(x=s)
        plt.xlabel(numerical_col)
        plt.title(f"{numerical_col} box plot")
    else:
        raise ValueError("plot_type must be 'hist' or 'boxplot'.")

    plt.tight_layout()
    plt.show()


def grab_col_names(df: pd.DataFrame, cat_th: int = 10, car_th: int = 20) -> Tuple[List[str], List[str], List[str]]:
    """
    Return (cat_cols, num_cols, cat_but_car).

    - cat_cols: object columns + numeric columns with low unique counts (< cat_th)
    - num_cols: numeric columns excluding numeric-but-categorical
    - cat_but_car: object columns with high unique counts (> car_th)
    """
    dtypes = df.dtypes
    is_obj = dtypes.eq("O")

    nunique = df.nunique(dropna=False)

    cat_cols = df.columns[is_obj].tolist()
    num_but_cat = df.columns[(~is_obj) & (nunique.lt(cat_th))].tolist()
    cat_but_car = df.columns[is_obj & (nunique.gt(car_th))].tolist()

    cat_set_excluding_cardinal = [c for c in (cat_cols + num_but_cat) if c not in set(cat_but_car)]
    num_cols = [c for c in df.columns[~is_obj].tolist() if c not in set(num_but_cat)]

    print(f"Observations: {df.shape[0]}")
    print(f"Variables: {df.shape[1]}")
    print(f"cat_cols: {len(cat_set_excluding_cardinal)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_set_excluding_cardinal, num_cols, cat_but_car


def high_correlated_cols(
    df: pd.DataFrame,
    plot: bool = False,
    corr_th: float = 0.90,
    method: str = "pearson",
    fig_id: Optional[str] = None,
    images_path: Optional[Path | str] = None,
) -> List[str]:
    """
    Find columns to drop based on high absolute correlation among numeric features.

    Returns: list of column names whose correlation with any other column exceeds corr_th.
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] == 0:
        return []

    corr = num_df.corr(method=method)
    abs_corr = corr.abs()

    upper = abs_corr.where(np.triu(np.ones(abs_corr.shape, dtype=bool), k=1))
    drop_list = upper.columns[(upper.gt(corr_th)).any()].tolist()

    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(15, 15))
        sns.heatmap(corr, annot=False, cmap="RdBu", center=0)
        plt.title("Correlation heatmap")

        if fig_id and images_path is not None:
            out_dir = Path(images_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{fig_id}.png"
            plt.tight_layout()
            plt.savefig(out_file, format="png", dpi=300)

        plt.tight_layout()
        plt.show()

    return drop_list


def outlier_thresholds(df: pd.DataFrame, col_name: str, q1: float = 0.25, q3: float = 0.75) -> Tuple[float, float]:
    """Compute IQR-based lower/upper limits for outlier detection."""
    if not (0 <= q1 < q3 <= 1):
        raise ValueError("q1 and q3 must satisfy 0 <= q1 < q3 <= 1.")

    s = _as_numeric_series(df, col_name)
    q_low, q_high = s.quantile([q1, q3]).to_numpy()
    iqr = q_high - q_low
    low_limit = q_low - 1.5 * iqr
    up_limit = q_high + 1.5 * iqr
    return float(low_limit), float(up_limit)


def check_outlier(df: pd.DataFrame, col_name: str, q1: float = 0.25, q3: float = 0.75) -> bool:
    """Return True if the column contains any IQR outliers."""
    low, up = outlier_thresholds(df, col_name, q1=q1, q3=q3)
    s = _as_numeric_series(df, col_name)
    mask = s.lt(low) | s.gt(up)
    return bool(mask.any())


def grab_outliers(
    df: pd.DataFrame,
    col_name: str,
    index: bool = False,
    head: int = 5,
    q1: float = 0.25,
    q3: float = 0.75,
    print_table: bool = True,
) -> Optional[pd.Index]:
    """
    Print rows containing outliers for a numeric column, optionally return their indices.
    """
    low, up = outlier_thresholds(df, col_name, q1=q1, q3=q3)
    s = _as_numeric_series(df, col_name)

    mask = s.lt(low) | s.gt(up)
    outliers = df.loc[mask]

    if print_table:
        to_print = outliers.head(head) if len(outliers) > 10 else outliers
        print(to_print)

    if index:
        return outliers.index
    return None


def remove_outlier(df: pd.DataFrame, col_name: str, q1: float = 0.10, q3: float = 0.99) -> pd.DataFrame:
    """Return a copy of df with IQR outliers removed for the given numeric column."""
    low, up = outlier_thresholds(df, col_name, q1=q1, q3=q3)
    s = _as_numeric_series(df, col_name)
    mask = s.between(low, up, inclusive="both")
    return df.loc[mask].copy()


def missing_values_table(df: pd.DataFrame, na_name: bool = False) -> Optional[List[str]]:
    """
    Print a missing-values table (count and ratio) for columns with at least one missing value.
    If na_name=True, return the column names list.
    """
    miss = df.isna().sum()
    miss = miss[miss.gt(0)].sort_values(ascending=False)

    if miss.empty:
        print("No missing values")
        return [] if na_name else None

    ratio = (miss / len(df)).round(3)
    missing_df = pd.DataFrame({"n_miss": miss.astype(int), "ratio": ratio})
    print(missing_df)

    if na_name:
        return miss.index.tolist()
    return None
