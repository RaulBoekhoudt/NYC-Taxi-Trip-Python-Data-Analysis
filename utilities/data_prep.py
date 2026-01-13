from __future__ import annotations

from typing import Optional, Tuple, List
import numpy as np
import pandas as pd


def _require_column(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame.")


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    _require_column(df, col)
    s = df[col]
    if pd.api.types.is_datetime64_any_dtype(s):
        raise TypeError(f"Column '{col}' is datetime-like; quartile outlier rules are not appropriate.")
    if not pd.api.types.is_numeric_dtype(s):
        raise TypeError(f"Column '{col}' must be numeric, got dtype={s.dtype}.")
    return s


def outlier_thresholds(
    df: pd.DataFrame,
    col: str,
    q1: float = 0.25,
    q3: float = 0.75,
    *,
    whisker_scale: float = 1.5,
) -> Tuple[float, float]:
    """
    Compute IQR-based lower/upper limits for outlier detection.

    Notes
    - This is the classic Tukey rule: [Q1 - k*IQR, Q3 + k*IQR].
    - If your distribution is heavily skewed, consider transforming the data
      (e.g. log1p) or using a robust method (MAD) rather than forcing the median.
    """
    if not (0 <= q1 < q3 <= 1):
        raise ValueError("Quantiles must satisfy 0 <= q1 < q3 <= 1.")

    s = _numeric_series(df, col).dropna()
    if s.empty:
        return (np.nan, np.nan)

    q_low, q_high = s.quantile([q1, q3]).to_numpy()
    iqr = q_high - q_low
    low = q_low - whisker_scale * iqr
    high = q_high + whisker_scale * iqr
    return float(low), float(high)


def check_outlier(df: pd.DataFrame, col: str, q1: float = 0.25, q3: float = 0.75) -> bool:
    """Return True if the column contains any outliers under the IQR rule."""
    s = _numeric_series(df, col)
    low, high = outlier_thresholds(df, col, q1=q1, q3=q3)
    if np.isnan(low) or np.isnan(high):
        return False
    return bool((s.lt(low) | s.gt(high)).any())


def grab_outliers(
    df: pd.DataFrame,
    col: str,
    *,
    return_index: bool = False,
    head: int = 5,
    q1: float = 0.25,
    q3: float = 0.75,
    print_table: bool = True,
) -> Optional[pd.Index]:
    """
    Print rows containing outliers for `col`. Optionally return their index.
    """
    s = _numeric_series(df, col)
    low, high = outlier_thresholds(df, col, q1=q1, q3=q3)
    if np.isnan(low) or np.isnan(high):
        if print_table:
            print("No outliers (empty / all-NA column).")
        return df.index[:0] if return_index else None

    mask = s.lt(low) | s.gt(high)
    outliers = df.loc[mask]

    if print_table:
        print(outliers.head(head) if len(outliers) > 10 else outliers)

    return outliers.index if return_index else None


def remove_outlier(df: pd.DataFrame, col: str, q1: float = 0.10, q3: float = 0.99) -> pd.DataFrame:
    """
    Return a copy of df with outliers (per IQR rule on `col`) removed.
    """
    s = _numeric_series(df, col)
    low, high = outlier_thresholds(df, col, q1=q1, q3=q3)
    if np.isnan(low) or np.isnan(high):
        return df.copy()

    keep = s.between(low, high, inclusive="both")
    return df.loc[keep].copy()


def missing_values_table(df: pd.DataFrame, *, return_cols: bool = False) -> Optional[List[str]]:
    """
    Print a missing-values table (count + ratio) for columns with any NA.
    Optionally return the list of columns that contain missing values.
    """
    miss = df.isna().sum()
    miss = miss[miss.gt(0)].sort_values(ascending=False)

    if miss.empty:
        print("No missing values.")
        return [] if return_cols else None

    ratio = (miss / len(df)).round(3)
    table = pd.DataFrame({"n_miss": miss.astype(int), "ratio": ratio})
    print(table)

    return table.index.tolist() if return_cols else None
