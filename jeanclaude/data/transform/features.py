"""
Macro feature engineering.

Transforms raw macro series (rates, spreads, VIX...) into
the normalized state variables used by the HMM regime detector
and the LSTM covariance estimator.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


# Default macro state variables consumed by the signals module.
# Keys are friendly names; values are FRED series IDs used in FREDSource.
DEFAULT_MACRO_SERIES: dict[str, str] = {
    "VIX":           "VIXCLS",
    "TED_spread":    "TEDRATE",
    "yield_slope":   "T10Y2Y",
    "credit_spread": "BAMLH0A0HYM2",
    "fed_funds":     "FEDFUNDS",
    "cpi_yoy":       "CPIAUCSL",
    "unemployment":  "UNRATE",
    "indpro":        "INDPRO",
}

# Minimum observations required before emitting a normalised value in
# expanding mode.  Rows before this threshold are NaN (callers must dropna).
_EXPANDING_MIN_OBS: int = 60


def build_state_variables(
    macro_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    normalize: bool = True,
    normalization: str = "expanding",
    fillna_method: str = "ffill",
) -> pd.DataFrame:
    """Build the state variable matrix for regime detection / LSTM.

    Parameters
    ----------
    macro_df : pd.DataFrame
        Raw macro series, date-indexed (one column per indicator).
    feature_cols : list[str], optional
        Columns to include. Defaults to all columns in ``macro_df``.
    normalize : bool
        If True, normalise each column to zero median / unit IQR.
    normalization : str
        ``'expanding'`` (default) — expanding median/IQR, causal, no
        look-ahead.  The first ``_EXPANDING_MIN_OBS`` rows will be NaN;
        callers should call ``.dropna()`` after ``build_state_variables``.

        ``'full'`` — RobustScaler fit on the entire sample.  Introduces
        look-ahead bias in backtests (future statistics normalise past
        values).  Emits a :class:`UserWarning`; kept for unit tests that
        exercise HMM mechanics in isolation.
    fillna_method : str
        How to fill gaps: ``'ffill'`` (forward-fill) or ``'interpolate'``.

    Returns
    -------
    pd.DataFrame
        State variable matrix, same index as input (after alignment).
        Columns are the selected features.  With ``normalization='expanding'``
        and ``normalize=True``, the first ``_EXPANDING_MIN_OBS - 1`` rows
        will contain NaN — drop them at the call site.
    """
    df = macro_df.copy()

    if feature_cols is not None:
        missing = set(feature_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Columns not found in macro_df: {missing}")
        df = df[feature_cols]

    # Fill gaps
    if fillna_method == "ffill":
        df = df.ffill()
    elif fillna_method == "interpolate":
        df = df.interpolate(method="time")

    # Drop rows still all-NaN (at the very start of history)
    df = df.dropna(how="all")

    if normalize:
        if normalization == "expanding":
            # Causal normalisation: at each t only data up to t is used.
            # Rows before _EXPANDING_MIN_OBS are NaN — callers must dropna.
            med = df.expanding(_EXPANDING_MIN_OBS).median()
            q75 = df.expanding(_EXPANDING_MIN_OBS).quantile(0.75)
            q25 = df.expanding(_EXPANDING_MIN_OBS).quantile(0.25)
            iqr = (q75 - q25).replace(0.0, np.nan)
            df = (df - med) / iqr
        elif normalization == "full":
            warnings.warn(
                "normalization='full' usa mediana/IQR dell'intero campione: "
                "look-ahead nei backtest. Usare 'expanding'.",
                UserWarning,
                stacklevel=2,
            )
            scaler = RobustScaler()
            df = pd.DataFrame(
                scaler.fit_transform(df),
                index=df.index,
                columns=df.columns,
            )
        else:
            raise ValueError(f"normalization sconosciuta: {normalization!r}")

    return df


def add_momentum_features(
    macro_df: pd.DataFrame,
    windows: list[int] = [5, 21, 63],
) -> pd.DataFrame:
    """Add rolling momentum (rate-of-change) features for each macro series.

    Useful as additional signals for regime detection — captures whether
    conditions are improving or deteriorating, not just their level.

    Parameters
    ----------
    macro_df : pd.DataFrame
        Raw macro series.
    windows : list[int]
        Rolling windows in business days (default: 1W, 1M, 3M).

    Returns
    -------
    pd.DataFrame
        Original columns + ``{col}_roc_{window}d`` for each window.
    """
    result = macro_df.copy()

    for col in macro_df.columns:
        for w in windows:
            roc = macro_df[col].pct_change(periods=w)
            result[f"{col}_roc_{w}d"] = roc

    return result


def align_to_prices(
    macro_df: pd.DataFrame,
    price_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Align macro data to the price/returns index.

    Macro series often have different frequencies (monthly GDP, daily VIX).
    This reindexes everything to the given business-day calendar.

    Parameters
    ----------
    macro_df : pd.DataFrame
        Macro series, any frequency.
    price_index : pd.DatetimeIndex
        Target index (e.g. from daily price DataFrame).

    Returns
    -------
    pd.DataFrame
        Macro data reindexed and forward-filled to match price_index.
    """
    return macro_df.reindex(price_index).ffill()
