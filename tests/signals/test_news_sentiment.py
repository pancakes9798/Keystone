"""Tests for signals/news/ — NewsSentiment, FinBERTScorer, NewsAggregator, RefinitivNewsSource."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Task 1 — NewsSentiment
# ---------------------------------------------------------------------------

from jeanclaude.signals.news.types import NewsSentiment


def test_news_sentiment_construction():
    ns = NewsSentiment(
        scores=pd.Series({"equity": 0.5, "bonds": -0.3}),
        confidence=0.8,
        timestamp=pd.Timestamp("2025-01-10"),
        n_articles=40,
    )
    assert ns.confidence == pytest.approx(0.8)
    assert ns.n_articles == 40
    assert ns.scores["equity"] == pytest.approx(0.5)


def test_news_sentiment_is_frozen():
    ns = NewsSentiment(
        scores=pd.Series({"equity": 0.0}),
        confidence=0.5,
        timestamp=pd.Timestamp("2025-01-10"),
        n_articles=10,
    )
    with pytest.raises((AttributeError, TypeError)):
        ns.confidence = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Task 2 — FinBERTScorer
# ---------------------------------------------------------------------------

from jeanclaude.signals.news.scorer import FinBERTScorer


def _make_scorer_with_mock(label: str, score_val: float = 0.95) -> FinBERTScorer:
    """Return a FinBERTScorer whose pipeline is pre-mocked."""
    scorer = FinBERTScorer()
    mock_pipe = MagicMock(return_value=[[{"label": label, "score": score_val}]])
    scorer._pipeline = mock_pipe
    return scorer


def test_scorer_maps_positive_to_plus1():
    scorer = _make_scorer_with_mock("positive")
    df = pd.DataFrame({"headline": ["Markets rally strongly"]})
    result = scorer.score(df)
    assert result["sentiment_score"].iloc[0] == pytest.approx(1.0)


def test_scorer_maps_negative_to_minus1():
    scorer = _make_scorer_with_mock("negative")
    df = pd.DataFrame({"headline": ["Stocks plunge on recession fears"]})
    result = scorer.score(df)
    assert result["sentiment_score"].iloc[0] == pytest.approx(-1.0)


def test_scorer_maps_neutral_to_zero():
    scorer = _make_scorer_with_mock("neutral")
    df = pd.DataFrame({"headline": ["Fed holds rates steady"]})
    result = scorer.score(df)
    assert result["sentiment_score"].iloc[0] == pytest.approx(0.0)


def test_scorer_preserves_original_columns():
    scorer = _make_scorer_with_mock("positive")
    df = pd.DataFrame({
        "headline": ["Markets up"],
        "asset_class": ["equity"],
        "story_id": ["abc123"],
    })
    result = scorer.score(df)
    assert "asset_class" in result.columns
    assert "story_id" in result.columns
    assert "sentiment_score" in result.columns


def test_scorer_handles_batch_larger_than_batch_size():
    """50 headlines with batch_size=32 must process all rows."""
    scorer = FinBERTScorer(batch_size=32)
    def mock_pipe(texts, **kwargs):
        return [{"label": "neutral", "score": 0.9}] * len(texts)
    scorer._pipeline = mock_pipe
    df = pd.DataFrame({"headline": [f"headline {i}" for i in range(50)]})
    result = scorer.score(df)
    assert len(result) == 50
    assert (result["sentiment_score"] == 0.0).all()


def test_scorer_raises_import_error_without_transformers():
    scorer = FinBERTScorer()
    scorer._pipeline = None
    with patch.dict("sys.modules", {"transformers": None}):
        with pytest.raises(ImportError, match="transformers"):
            scorer._get_pipeline()


# ---------------------------------------------------------------------------
# Task 3 — NewsAggregator
# ---------------------------------------------------------------------------

from jeanclaude.signals.news.aggregator import NewsAggregator


def _make_scored_df(dates_and_classes: list[tuple[str, str, float]]) -> pd.DataFrame:
    """Helper: crea un DataFrame scored da lista di (date, asset_class, score)."""
    rows = [
        {"date": pd.Timestamp(d), "asset_class": ac, "headline": "test", "story_id": "x", "sentiment_score": s}
        for d, ac, s in dates_and_classes
    ]
    return pd.DataFrame(rows).set_index("date")


def test_aggregator_5day_window_excludes_old_headlines():
    df = _make_scored_df([
        ("2025-01-10", "equity", 1.0),
        ("2025-01-04", "equity", -1.0),  # 6 days before as_of=Jan10 → outside window
    ])
    agg = NewsAggregator(window_days=5)
    result = agg.aggregate(df, as_of=pd.Timestamp("2025-01-10"))
    assert result.scores["equity"] == pytest.approx(1.0)


def test_aggregator_averages_by_asset_class():
    df = _make_scored_df([
        ("2025-01-10", "equity", 1.0),
        ("2025-01-09", "equity", -0.5),
        ("2025-01-10", "bonds", 0.5),
    ])
    agg = NewsAggregator(window_days=5)
    result = agg.aggregate(df, as_of=pd.Timestamp("2025-01-10"))
    assert result.scores["equity"] == pytest.approx((1.0 + (-0.5)) / 2)
    assert result.scores["bonds"] == pytest.approx(0.5)


def test_aggregator_missing_asset_class_gets_zero():
    df = _make_scored_df([("2025-01-10", "equity", 0.8)])
    agg = NewsAggregator(window_days=5)
    result = agg.aggregate(df, as_of=pd.Timestamp("2025-01-10"))
    assert "equity" in result.scores.index
    assert "bonds" not in result.scores.index


def test_aggregator_confidence_50_articles_gives_1():
    rows = [
        {"date": pd.Timestamp("2025-01-10"), "asset_class": "equity",
         "headline": "x", "story_id": str(i), "sentiment_score": 0.0}
        for i in range(50)
    ]
    df = pd.DataFrame(rows).set_index("date")
    agg = NewsAggregator(window_days=5)
    result = agg.aggregate(df, as_of=pd.Timestamp("2025-01-10"))
    assert result.confidence == pytest.approx(1.0)
    assert result.n_articles == 50


def test_aggregator_confidence_25_articles_gives_half():
    rows = [
        {"date": pd.Timestamp("2025-01-10"), "asset_class": "equity",
         "headline": "x", "story_id": str(i), "sentiment_score": 0.0}
        for i in range(25)
    ]
    df = pd.DataFrame(rows).set_index("date")
    agg = NewsAggregator(window_days=5)
    result = agg.aggregate(df, as_of=pd.Timestamp("2025-01-10"))
    assert result.confidence == pytest.approx(0.5)


def test_aggregator_empty_dataframe_returns_zero_confidence():
    df = pd.DataFrame(
        columns=["asset_class", "headline", "story_id", "sentiment_score"]
    ).set_index(pd.DatetimeIndex([], name="date"))
    agg = NewsAggregator(window_days=5)
    result = agg.aggregate(df, as_of=pd.Timestamp("2025-01-10"))
    assert result.confidence == pytest.approx(0.0)
    assert result.n_articles == 0
    assert result.scores.empty


def test_aggregator_all_outside_window_returns_zero_confidence():
    df = _make_scored_df([("2025-01-01", "equity", 1.0)])  # 9 days before as_of
    agg = NewsAggregator(window_days=5)
    result = agg.aggregate(df, as_of=pd.Timestamp("2025-01-10"))
    assert result.confidence == pytest.approx(0.0)
    assert result.n_articles == 0


# ---------------------------------------------------------------------------
# Task 4 — RefinitivNewsSource
# ---------------------------------------------------------------------------

from jeanclaude.signals.news.fetcher import RefinitivNewsSource

_QUERIES = {
    "equity": ["SPY", "S&P 500"],
    "bonds": ["Treasury", "TLT"],
}


def test_fetcher_returns_dataframe_with_correct_schema():
    src = RefinitivNewsSource()
    mock_ld = MagicMock()
    mock_df = pd.DataFrame({
        "headline": ["S&P 500 rises", "SPY ETF gains"],
        "storyId": ["id1", "id2"],
    }, index=pd.DatetimeIndex(["2025-01-10", "2025-01-10"], name="date"))
    mock_ld.news.Headlines.Definition.return_value.get_data.return_value = mock_df
    src._lseg_data = mock_ld

    result = src.get_headlines(_QUERIES, start="2025-01-06", end="2025-01-10")

    assert isinstance(result, pd.DataFrame)
    assert "asset_class" in result.columns
    assert "headline" in result.columns
    assert "story_id" in result.columns
    assert result.index.name == "date"


def test_fetcher_labels_asset_class_correctly():
    src = RefinitivNewsSource()
    mock_ld = MagicMock()
    mock_df = pd.DataFrame({
        "headline": ["Treasury yields rise"],
        "storyId": ["id1"],
    }, index=pd.DatetimeIndex(["2025-01-10"], name="date"))
    mock_ld.news.Headlines.Definition.return_value.get_data.return_value = mock_df
    src._lseg_data = mock_ld

    result = src.get_headlines({"bonds": ["Treasury"]}, start="2025-01-06", end="2025-01-10")
    assert result["asset_class"].iloc[0] == "bonds"


def test_fetcher_returns_empty_df_when_no_results():
    src = RefinitivNewsSource()
    mock_ld = MagicMock()
    mock_ld.news.Headlines.Definition.return_value.get_data.return_value = pd.DataFrame()
    src._lseg_data = mock_ld

    result = src.get_headlines(_QUERIES, start="2025-01-06", end="2025-01-10")
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_fetcher_raises_import_error_without_lseg():
    src = RefinitivNewsSource()
    src._lseg_data = None
    with patch.dict("sys.modules", {"lseg": None, "lseg.data": None}):
        with pytest.raises(ImportError, match="lseg-data"):
            src._ensure_lseg()


# ---------------------------------------------------------------------------
# Task 6 — CompositeSignalBuilder con news_sentiment
# ---------------------------------------------------------------------------

from jeanclaude.signals.composite.builder import CompositeSignalBuilder
from jeanclaude.signals.composite.types import DEFAULT_REGIME_TABLE
from jeanclaude.signals.macro.labels import RegimeLabel, RegimeResult


def _make_regime_result_for_builder():
    """RegimeResult minimo: tutto EXPANSION con alta confidenza."""
    dates = pd.date_range("2025-01-01", periods=10)
    labels = pd.Series([RegimeLabel.EXPANSION.value] * 10, index=dates, dtype="int64")
    proba = pd.DataFrame(
        {0: 0.9, 1: 0.05, 2: 0.05}, index=dates
    )
    return RegimeResult(labels=labels, probabilities=proba, model=MagicMock())


def test_builder_without_news_behaves_as_before():
    """build(result) senza news_sentiment deve produrre lo stesso output di prima."""
    builder = CompositeSignalBuilder(DEFAULT_REGIME_TABLE)
    result = _make_regime_result_for_builder()
    signal = builder.build(result)
    assert signal.scores["equity"] > 0


def test_builder_blends_macro_and_news():
    """Con news_sentiment, lo score finale deve essere il blend atteso."""
    builder = CompositeSignalBuilder(DEFAULT_REGIME_TABLE)
    result = _make_regime_result_for_builder()

    macro_signal = builder.build(result)
    macro_equity = macro_signal.scores["equity"]

    news = NewsSentiment(
        scores=pd.Series({"equity": -0.5, "bonds": 0.0, "gold": 0.0, "cash": 0.0}),
        confidence=1.0,
        timestamp=pd.Timestamp("2025-01-10"),
        n_articles=50,
    )
    blended_signal = builder.build(result, news_sentiment=news, macro_weight=0.7)

    expected = 0.7 * macro_equity + 0.3 * (-0.5) * 1.0
    assert blended_signal.scores["equity"] == pytest.approx(expected, abs=1e-6)


def test_builder_ignores_news_with_low_confidence():
    """confidence < 0.05 → il builder usa solo macro, ignora news."""
    builder = CompositeSignalBuilder(DEFAULT_REGIME_TABLE)
    result = _make_regime_result_for_builder()

    macro_signal = builder.build(result)

    news = NewsSentiment(
        scores=pd.Series({"equity": -1.0, "bonds": 1.0, "gold": 0.0, "cash": 0.0}),
        confidence=0.02,
        timestamp=pd.Timestamp("2025-01-10"),
        n_articles=1,
    )
    signal_with_low_conf = builder.build(result, news_sentiment=news)

    pd.testing.assert_series_equal(signal_with_low_conf.scores, macro_signal.scores)


def test_builder_fills_missing_asset_class_with_zero():
    """Se news non copre una asset class, il suo score news è 0.0."""
    builder = CompositeSignalBuilder(DEFAULT_REGIME_TABLE)
    result = _make_regime_result_for_builder()

    macro_signal = builder.build(result)
    macro_bonds = macro_signal.scores["bonds"]

    news = NewsSentiment(
        scores=pd.Series({"equity": 1.0}),  # bonds/gold/cash assenti
        confidence=1.0,
        timestamp=pd.Timestamp("2025-01-10"),
        n_articles=50,
    )
    signal = builder.build(result, news_sentiment=news, macro_weight=0.7)

    expected_bonds = 0.7 * macro_bonds + 0.3 * 0.0 * 1.0
    assert signal.scores["bonds"] == pytest.approx(expected_bonds, abs=1e-6)
