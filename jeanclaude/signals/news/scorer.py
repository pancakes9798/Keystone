"""FinBERTScorer — batch FinBERT inference on financial headlines."""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_LABEL_TO_SCORE: dict[str, float] = {
    "positive": 1.0,
    "neutral": 0.0,
    "negative": -1.0,
}


class FinBERTScorer:
    """Scores financial headlines using ProsusAI/finbert.

    The pipeline is loaded lazily on first call to ``score()`` — no model
    download happens at construction time. This keeps imports fast and
    makes unit testing (via mock injection) straightforward.

    Parameters
    ----------
    batch_size : int
        Number of headlines per FinBERT forward pass. Default 32.
    device : str
        ``"cpu"`` or ``"cuda"``. Default ``"cpu"``.
    """

    def __init__(self, batch_size: int = 32, device: str = "cpu") -> None:
        self._batch_size = batch_size
        self._device = device
        self._pipeline: Any = None  # injected in tests or lazy-loaded

    def score(self, headlines: pd.DataFrame) -> pd.DataFrame:
        """Add ``sentiment_score`` column to ``headlines``.

        Parameters
        ----------
        headlines : pd.DataFrame
            Must contain a ``"headline"`` column (str).

        Returns
        -------
        pd.DataFrame
            Original DataFrame plus a ``"sentiment_score"`` column
            with values in ``{-1.0, 0.0, 1.0}``.
        """
        if headlines.empty:
            return headlines.assign(sentiment_score=pd.Series(dtype=float))

        pipe = self._get_pipeline()
        texts = headlines["headline"].tolist()
        scores: list[float] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            try:
                results = pipe(batch, truncation=True, max_length=512)
                # pipeline may return list-of-list (batched) or list-of-dict
                if results and isinstance(results[0], list):
                    results = [r[0] for r in results]
                for r in results:
                    label = r.get("label", "neutral").lower()
                    scores.append(_LABEL_TO_SCORE.get(label, 0.0))
            except Exception as exc:
                logger.warning(
                    "FinBERT batch %d-%d failed: %s — filling with 0.0",
                    i, i + self._batch_size, exc,
                )
                scores.extend([0.0] * len(batch))

        result = headlines.copy()
        result["sentiment_score"] = scores
        return result

    def _get_pipeline(self) -> Any:
        """Return the FinBERT pipeline, loading it lazily on first call."""
        if self._pipeline is not None:
            return self._pipeline

        try:
            from transformers import pipeline  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "transformers is not installed. "
                "Run: pip install transformers torch"
            ) from exc

        import torch  # noqa: PLC0415
        device = 0 if (self._device == "cuda" and torch.cuda.is_available()) else -1

        self._pipeline = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            device=device,
            batch_size=self._batch_size,
        )
        logger.info("FinBERT pipeline loaded (device=%s)", self._device)
        return self._pipeline
