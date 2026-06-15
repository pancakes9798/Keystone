"""
HMM-based macro regime detector — pomegranate Student-t emissions + sticky Dirichlet.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import torch
from pomegranate.distributions import IndependentComponents, StudentT
from pomegranate.hmm import DenseHMM

logger = logging.getLogger(__name__)

from .labels import RegimeLabel, RegimeResult, _build_label_map


class RegimeDetector:
    """Detects macro regimes using a Hidden Markov Model with Student-t emissions.

    Classifies each time step into one of three regimes:
    EXPANSION, CONTRACTION, or TRANSITION.

    Improvements over Gaussian HMM:
    - Student-t emissions: robust to fat tails in VIX, spreads, rates.
    - Sticky Dirichlet prior on transition matrix: encourages regime persistence.

    Parameters
    ----------
    n_components : int
        Number of hidden states (regimes). Must be 3.
    n_iter : int
        Maximum EM iterations for fitting.
    random_state : int
        Seed for reproducibility.  The seed is forwarded to both
        ``torch.manual_seed`` / ``np.random.seed`` *and* to the
        ``DenseHMM(random_state=…)`` constructor, which seeds the
        internal KMeans initialiser.  Without passing it to DenseHMM
        the KMeans centroid selection uses an unseeded
        ``numpy.random.RandomState(None)`` and produces different
        initial parameters each run, causing EM to converge to
        different local optima even with identical torch seeds.
    label_feature : str
        Column in state_vars used to order states into regime labels.
        Default: ``"VIX"`` (ascending → low VIX = expansion).
    label_order : str
        ``"ascending"`` or ``"descending"``. Controls which end of the
        feature mean range maps to EXPANSION.
    student_t_df : int
        Degrees of freedom for Student-t emissions. Must be a positive integer.
        Default 5: heavy tails, finite variance and kurtosis (~6).
        Higher values → lighter tails (ν → ∞ = Gaussian).
    sticky_concentration : float
        Dirichlet prior weight on self-transitions. Default 8.0: the
        self-transition entry gets ~8x more prior mass than off-diagonal
        entries, favouring persistent regimes.
    """

    _N_RESTARTS: int = 5

    def __init__(
        self,
        n_components: int = 3,
        n_iter: int = 100,
        random_state: int = 42,
        label_feature: str = "VIX",
        label_order: str = "ascending",
        student_t_df: int = 5,
        sticky_concentration: float = 8.0,
    ) -> None:
        if n_components != 3:
            raise ValueError(
                f"n_components must be 3 (EXPANSION/CONTRACTION/TRANSITION). Got {n_components}."
            )
        if label_order not in ("ascending", "descending"):
            raise ValueError(
                f"label_order must be 'ascending' or 'descending', got {label_order!r}"
            )
        if student_t_df < 1:
            raise ValueError(f"student_t_df must be >= 1. Got {student_t_df}.")
        if sticky_concentration <= 0:
            raise ValueError(
                f"sticky_concentration must be > 0. Got {sticky_concentration}."
            )

        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state
        self.label_feature = label_feature
        self.label_order = label_order
        self.student_t_df = student_t_df
        self.sticky_concentration = sticky_concentration

        self._model: DenseHMM | None = None
        self._label_map: dict[int, RegimeLabel] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, state_vars: pd.DataFrame) -> RegimeResult:
        """Fit the HMM on the full available history (expanding window).

        Parameters
        ----------
        state_vars : pd.DataFrame
            Normalised macro feature matrix from ``build_state_variables``.
            Must have a ``DatetimeIndex`` and no NaN values.

        Returns
        -------
        RegimeResult
        """
        self._validate(state_vars)
        X = state_vars.values.astype(np.float32)

        model = self._fit_best(X)

        X_tensor = torch.tensor(X).unsqueeze(0)                # (1, T, n_features)
        raw_states = model.predict(X_tensor)[0].numpy()        # (T,)
        raw_probas = model.predict_proba(X_tensor)[0].numpy()  # (T, n_components)

        label_map = _build_label_map(
            X, raw_states, state_vars,
            self.label_feature, self.label_order, self.n_components,
        )

        self._model = model
        self._label_map = label_map

        return self._build_result(state_vars, raw_states, raw_probas, model, label_map)

    def predict(self, state_vars: pd.DataFrame) -> RegimeResult:
        """Infer regimes on new data using the already-fitted model.

        Parameters
        ----------
        state_vars : pd.DataFrame
            Same feature layout as the data used in ``fit``.

        Returns
        -------
        RegimeResult
        """
        if self._model is None or self._label_map is None:
            raise RuntimeError("Call fit() before predict().")
        self._validate(state_vars)
        X = state_vars.values.astype(np.float32)
        X_tensor = torch.tensor(X).unsqueeze(0)
        raw_states = self._model.predict(X_tensor)[0].numpy()
        raw_probas = self._model.predict_proba(X_tensor)[0].numpy()
        return self._build_result(state_vars, raw_states, raw_probas, self._model, self._label_map)

    def current_regime(
        self, state_vars: pd.DataFrame
    ) -> tuple[RegimeLabel, np.ndarray]:
        """Return the regime label and posterior probability vector for the last row.

        Returns
        -------
        tuple[RegimeLabel, np.ndarray]
            ``(label, proba)`` where ``proba`` has shape ``(n_components,)``.
        """
        result = self.predict(state_vars)
        last_label = RegimeLabel(int(result.labels.iloc[-1]))
        last_proba = result.probabilities.iloc[-1].values
        return last_label, last_proba

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_sticky_transmat(self) -> torch.Tensor:
        """Build row-normalised sticky transition matrix.

        Diagonal entries get ``sticky_concentration`` weight;
        off-diagonal entries get 1.0. After normalisation the
        self-transition probability is approximately
        ``sticky_concentration / (sticky_concentration + n_components - 1)``.
        """
        transmat = torch.ones(self.n_components, self.n_components)
        transmat.fill_diagonal_(self.sticky_concentration)
        return transmat / transmat.sum(dim=1, keepdim=True)

    def _build_emissions(self, n_features: int) -> list:
        """One IndependentComponents(StudentT×n_features) per state.

        ``dofs`` must be an integer (pomegranate 1.x constraint).
        """
        return [
            IndependentComponents(
                [StudentT(dofs=self.student_t_df) for _ in range(n_features)]
            )
            for _ in range(self.n_components)
        ]

    def _fit_best(self, X: np.ndarray) -> DenseHMM:
        """Fit _N_RESTARTS models, return the one with highest log-likelihood."""
        n_features = X.shape[1]
        X_tensor = torch.tensor(X).unsqueeze(0)
        best_model: DenseHMM | None = None
        best_score = -np.inf

        for i in range(self._N_RESTARTS):
            seed_i = self.random_state + i
            torch.manual_seed(seed_i)
            np.random.seed(seed_i)
            emissions = self._build_emissions(n_features)
            transmat = self._build_sticky_transmat()
            model = DenseHMM(
                distributions=emissions,
                edges=transmat,
                max_iter=self.n_iter,
                verbose=False,
                # random_state seeds the internal KMeans centroid selector;
                # without it DenseHMM uses numpy.random.RandomState(None)
                # and KMeans picks different initial centroids each run,
                # making EM converge to different local optima.
                random_state=seed_i,
            )
            try:
                model.fit(X_tensor)
                score = float(model.log_probability(X_tensor))
            except Exception as e:
                logger.warning("RegimeDetector: restart %d failed: %s", i, e)
                continue
            if not math.isfinite(score):
                logger.warning("RegimeDetector: restart %d produced non-finite log-prob: %s", i, score)
                continue
            if score > best_score:
                best_score = score
                best_model = model

        if best_model is None:
            raise RuntimeError("All HMM restarts failed to converge.")
        return best_model

    @staticmethod
    def _validate(state_vars: pd.DataFrame) -> None:
        if state_vars.isnull().any().any():
            raise ValueError(
                "state_vars contains NaN values. "
                "Run build_state_variables(...) with fillna_method='ffill' first."
            )
        if not isinstance(state_vars.index, pd.DatetimeIndex):
            raise ValueError(
                "state_vars.index must be a DatetimeIndex. "
                "Pass a DataFrame with date-indexed rows."
            )

    @staticmethod
    def _build_result(
        state_vars: pd.DataFrame,
        raw_states: np.ndarray,
        raw_probas: np.ndarray,
        model: DenseHMM,
        label_map: dict[int, RegimeLabel],
    ) -> RegimeResult:
        n = model.n_distributions

        mapped_labels = pd.Series(
            [label_map[s] for s in raw_states],
            index=state_vars.index,
            name="regime",
        )

        # Re-order probability columns: col 0=EXPANSION, 1=CONTRACTION, 2=TRANSITION
        inv_map = {v: k for k, v in label_map.items()}
        col_order = [inv_map[RegimeLabel(i)] for i in range(n)]
        proba_df = pd.DataFrame(
            raw_probas[:, col_order],
            index=state_vars.index,
            columns=list(range(n)),
        )

        return RegimeResult(labels=mapped_labels, probabilities=proba_df, model=model)
