"""Il parametro adjustments arriva a ld.get_history e finisce nella cache key."""
from unittest.mock import MagicMock, patch

import pandas as pd

from jeanclaude.data.loader import DataConfig


def test_dataconfig_has_adjustments_default():
    cfg = DataConfig()
    assert cfg.price_adjustments == ("CCH", "CRE", "RTS", "RPO")


def test_cache_key_includes_adjustments():
    from jeanclaude.data.loader import DataLoader
    cfg_adj = DataConfig(price_adjustments=("CCH", "CRE", "RTS", "RPO"))
    cfg_raw = DataConfig(price_adjustments=())
    k1 = DataLoader(cfg_adj)._price_cache_key(["SPY.P"], "close", "Refinitiv")
    k2 = DataLoader(cfg_raw)._price_cache_key(["SPY.P"], "close", "Refinitiv")
    assert k1 != k2


def test_cache_key_includes_source():
    """Cache keys must differ when source differs, even for same tickers and field."""
    from jeanclaude.data.loader import DataLoader
    cfg = DataConfig(price_adjustments=("CCH", "CRE", "RTS", "RPO"))
    loader = DataLoader(cfg)
    k_refinitiv = loader._price_cache_key(["SPY.P"], "close", "Refinitiv")
    k_yahoo = loader._price_cache_key(["SPY.P"], "close", "Yahoo Finance")
    assert k_refinitiv != k_yahoo
    # Verify source tag is in the key
    assert "src-refinitiv" in k_refinitiv
    assert "src-yahoo_finance" in k_yahoo


def test_refinitiv_passes_adjustments_to_get_history():
    from jeanclaude.data.ingestion import refinitiv as rmod
    src = rmod.RefinitivSource.__new__(rmod.RefinitivSource)  # niente sessione
    src._session_type = "platform"
    src._session = object()  # non-None → _ensure_session() skip
    fake_df = pd.DataFrame(
        {"AAPL.O": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2026-01-02", "2026-01-05"]),
    )
    with patch.object(rmod, "ld") as mock_ld:
        mock_ld.get_history.return_value = fake_df
        src.get_prices(["AAPL.O"], start="2026-01-01", end="2026-02-01",
                       adjustments=["CCH", "CRE"])
        kwargs = mock_ld.get_history.call_args.kwargs
        # lseg-data summaries.Definition accepts a list of individual values;
        # the top-level type hint says str but the actual layer uses try_copy_to_list.
        assert kwargs.get("adjustments") == ["CCH", "CRE"]
