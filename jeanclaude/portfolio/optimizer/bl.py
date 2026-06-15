# jeanclaude/portfolio/optimizer/bl.py
"""Black-Litterman Optimizer.

Traduce un CompositeSignal (output Fase 4) in pesi di portafoglio finali
usando il framework Black-Litterman.

Riferimento: He & Litterman (1999), Idzorek (2005).

Flusso:
    1. fit(returns, regime_labels)  → calibra μ per regime
    2. optimize(signal, w_mkt, returns) → BL posterior + max-Sharpe
"""
from __future__ import annotations

import logging

import cvxpy as cp
import numpy as np
import pandas as pd

from jeanclaude.signals.composite.types import CompositeSignal
from jeanclaude.signals.macro.labels import RegimeLabel

logger = logging.getLogger(__name__)

_CONFIDENCE_EPS = 1e-6


class BLOptimizer:
    """Black-Litterman portfolio optimizer guidato da CompositeSignal.

    Parameters
    ----------
    asset_class_map : dict[str, str]
        Mappa ticker → asset class (es. ``"equity"``, ``"bonds"``, ``"gold"``, ``"cash"``).
    tau : float
        Scaling del prior sulla covarianza (tipico 0.01–0.05).
    delta : float
        Coefficiente di avversione al rischio implicito del mercato.
    annualization_factor : int
        252 per rendimenti giornalieri, 52 per settimanali.
    min_obs : int
        Numero minimo di osservazioni per regime richiesto in ``fit()``.
    """

    def __init__(
        self,
        asset_class_map: dict[str, str],
        tau: float = 0.025,
        delta: float = 2.5,
        annualization_factor: int = 252,
        min_obs: int = 5,
        max_weight: float = 0.35,
        min_weight: float = 0.00,
        view_magnitude: float = 0.03,
        turnover_penalty: float = 0.10,
        max_confidence: float = 0.80,
    ) -> None:
        self.asset_class_map = asset_class_map
        self.tau = tau
        self.delta = delta
        self.annualization_factor = annualization_factor
        self.min_obs = min_obs
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.view_magnitude = view_magnitude
        self.turnover_penalty = turnover_penalty
        self.max_confidence = max_confidence
        self._mu_regime: dict[RegimeLabel, pd.Series] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        returns: pd.DataFrame,
        regime_labels: pd.Series,
    ) -> "BLOptimizer":
        """Calibra le medie di rendimento per regime.

        Parameters
        ----------
        returns : pd.DataFrame
            T × N rendimenti giornalieri/settimanali.
        regime_labels : pd.Series
            Index allineato a ``returns``, valori ∈ ``RegimeLabel``.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            Se un ticker di ``asset_class_map`` non è in ``returns.columns``.
        ValueError
            Se un regime ha meno di ``min_obs`` osservazioni.
        """
        missing = set(self.asset_class_map) - set(returns.columns)
        if missing:
            raise ValueError(
                f"Ticker non presenti in returns.columns: {sorted(missing)}"
            )

        aligned = regime_labels.reindex(returns.index).dropna()
        aligned_returns = returns.loc[aligned.index]

        mu_regime: dict[RegimeLabel, pd.Series] = {}
        for r in RegimeLabel:
            mask = aligned == r
            subset = aligned_returns[mask]
            if len(subset) < self.min_obs:
                raise ValueError(
                    f"Regime {r.name} ha {len(subset)} osservazioni "
                    f"(min_obs={self.min_obs})"
                )
            mu_regime[r] = subset.mean()

        self._mu_regime = mu_regime
        return self

    def optimize(
        self,
        signal: CompositeSignal,
        prior_weights: pd.Series,
        returns: pd.DataFrame,
    ) -> pd.Series:
        """Calcola i pesi di portafoglio ottimali tramite Black-Litterman + max-Sharpe.

        Parameters
        ----------
        signal : CompositeSignal
            Output di ``CompositeSignalBuilder.build()``.
        prior_weights : pd.Series
            Pesi di mercato (HRP/NCO), index=ticker.
        returns : pd.DataFrame
            Rendimenti recenti per stimare Σ (stessa struttura di ``fit``).

        Returns
        -------
        pd.Series
            Pesi finali ∈ [0, 1], sum = 1, index = ticker.

        Raises
        ------
        RuntimeError
            Se chiamato prima di ``fit()``.
        """
        if self._mu_regime is None:
            raise RuntimeError("Chiama fit() prima di optimize()")

        tickers = list(self.asset_class_map)
        N = len(tickers)

        # 1. Covarianza annualizzata e rendimenti di equilibrio
        Sigma = returns[tickers].cov().values * self.annualization_factor
        w_mkt = prior_weights.reindex(tickers).fillna(0.0).values
        pi = self.delta * Sigma @ w_mkt  # shape (N,)

        # 2. P matrix e Q vector (costruito da signal scores + π di equilibrio)
        P, classes = self._build_p_matrix()
        Q = self._build_q_from_signal(signal, classes, pi, tickers)

        # 3. Omega matrix (diagonale) calibrata su confidence
        # Capped a max_confidence per evitare che le views dominino completamente π
        confidence = float(np.clip(signal.confidence, _CONFIDENCE_EPS, self.max_confidence))
        tau_Sigma = self.tau * Sigma
        base_omega_diag = np.diag(P @ tau_Sigma @ P.T)
        omega_factor = (1.0 - confidence) / confidence
        Omega = np.diag(base_omega_diag * omega_factor)

        # 4. Posterior BL
        mu_hat, Sigma_hat = _bl_posterior(pi, tau_Sigma, P, Q, Omega)

        # 5. Max-utility via cvxpy con weight bounds e turnover penalty
        w = cp.Variable(N)
        # Ancora per il penalty: equal-weight, indipendente dal prior BL
        # Evita che prior sbilanciati (inv-vol, HRP) blocchino la riallocazione
        w_anchor = np.full(N, 1.0 / N)

        # Project risk_matrix onto PSD cone to avoid cvxpy DCP errors from
        # numerical inversion noise in Sigma_hat (M_inv from BL posterior).
        risk_raw = Sigma + Sigma_hat
        eigvals, eigvecs = np.linalg.eigh(risk_raw)
        risk_matrix = eigvecs @ np.diag(np.maximum(eigvals, 1e-8)) @ eigvecs.T
        risk_matrix = (risk_matrix + risk_matrix.T) / 2  # enforce symmetry

        # Turnover penalty: penalizza ||w - w_anchor||_1 (vs equal-weight)
        objective = cp.Maximize(
            mu_hat @ w
            - (self.delta / 2.0) * cp.quad_form(w, risk_matrix)
            - self.turnover_penalty * cp.norm1(w - w_anchor)
        )
        constraints = [
            cp.sum(w) == 1,
            w >= self.min_weight,
            w <= self.max_weight,
        ]
        prob = cp.Problem(objective, constraints)
        prob.solve()

        if w.value is None:
            logger.warning("cvxpy infeasible/unbounded — fallback a prior")
            return prior_weights.reindex(tickers).fillna(0.0)

        return pd.Series(np.asarray(w.value), index=tickers)

    # ------------------------------------------------------------------
    # Internal helpers (exposed for testability)
    # ------------------------------------------------------------------

    def _build_p_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Costruisce la P matrix (k × N) e la lista delle asset class.

        Ogni riga i rappresenta il portafoglio equal-weighted degli asset
        della classe i: P[i, j] = 1/|class_i| se ticker j ∈ class_i, else 0.
        Ogni riga somma a 1.0.

        Returns
        -------
        P : np.ndarray, shape (k, N)
        classes : list[str]
            Nomi delle asset class ordinati (righe di P).
        """
        tickers = list(self.asset_class_map)
        classes = sorted(set(self.asset_class_map.values()))
        P = np.zeros((len(classes), len(tickers)))

        for i, c in enumerate(classes):
            members_idx = [j for j, t in enumerate(tickers) if self.asset_class_map[t] == c]
            if members_idx:
                P[i, members_idx] = 1.0 / len(members_idx)

        return P, classes

    def _build_q_vector(
        self,
        signal: CompositeSignal,
        classes: list[str],
    ) -> np.ndarray:
        """Calcola il vettore Q (k,) con soft-weighting su regime_proba.

        Q[c] = Σ_r  signal.regime_proba[r] · μ_regime[r][assets_c].mean()

        Usa l'intero vettore di probabilità — non solo il regime dominante.
        """
        assert self._mu_regime is not None
        tickers = list(self.asset_class_map)
        regime_list = list(RegimeLabel)  # [EXPANSION, CONTRACTION, TRANSITION]

        Q = np.zeros(len(classes))
        for i, c in enumerate(classes):
            assets_in_class = [t for t in tickers if self.asset_class_map[t] == c]
            q_val = 0.0
            for r_idx, r in enumerate(regime_list):
                mu_r_class = float(self._mu_regime[r].loc[assets_in_class].mean())
                q_val += float(signal.regime_proba[r_idx]) * mu_r_class
            Q[i] = q_val

        return Q

    def _build_q_from_signal(
        self,
        signal: CompositeSignal,
        classes: list[str],
        pi: np.ndarray,
        tickers: list[str],
    ) -> np.ndarray:
        """Costruisce Q da signal.scores + π di equilibrio (forward-looking).

        Q[c] = π_c + signal.scores[c] × view_magnitude

        Dove π_c è la media di π sugli asset della classe c (annualizzato).
        signal.scores[c] ∈ [-1, 1] determina la direzione e intensità del tilt.
        view_magnitude (default 0.03) è il massimo tilt annuale in rendimento.

        Questo garantisce che:
        - In CONTRACTION con score[equity]=-0.8 → Q[equity] < π → sell equity
        - In EXPANSION con score[equity]=+0.8 → Q[equity] > π → buy equity
        """
        Q = np.zeros(len(classes))
        for i, c in enumerate(classes):
            assets_idx = [j for j, t in enumerate(tickers) if self.asset_class_map[t] == c]
            pi_c = float(np.mean(pi[assets_idx])) if assets_idx else 0.0
            score = float(signal.scores.get(c, 0.0))
            Q[i] = pi_c + score * self.view_magnitude
        return Q


# ---------------------------------------------------------------------------
# Module-level BL math
# ---------------------------------------------------------------------------


def _bl_posterior(
    pi: np.ndarray,
    tau_Sigma: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcola il posterior Black-Litterman (μ̂, Σ̂).

    Formule standard (He & Litterman 1999):
        M   = (τΣ)⁻¹ + P' Ω⁻¹ P
        μ̂   = M⁻¹ [(τΣ)⁻¹ π + P' Ω⁻¹ Q]
        Σ̂   = M⁻¹

    Parameters
    ----------
    pi : np.ndarray, shape (N,)
        Rendimenti di equilibrio impliciti: δ · Σ · w_mkt.
    tau_Sigma : np.ndarray, shape (N, N)
        τ · Σ, già calcolata.
    P : np.ndarray, shape (k, N)
        Matrice delle views (righe normalizzate).
    Q : np.ndarray, shape (k,)
        Vettore dei rendimenti attesi delle views.
    Omega : np.ndarray, shape (k, k)
        Matrice di incertezza delle views (diagonale).

    Returns
    -------
    mu_hat : np.ndarray, shape (N,)
    Sigma_hat : np.ndarray, shape (N, N)
    """
    N = tau_Sigma.shape[0]
    k = Omega.shape[0]

    eps_N = np.eye(N) * 1e-8
    eps_k = np.eye(k) * 1e-10

    tau_Sigma_inv = np.linalg.inv(tau_Sigma + eps_N)
    Omega_inv = np.linalg.inv(Omega + eps_k)

    M = tau_Sigma_inv + P.T @ Omega_inv @ P
    M_inv = np.linalg.inv(M + np.eye(N) * 1e-10)

    mu_hat = M_inv @ (tau_Sigma_inv @ pi + P.T @ Omega_inv @ Q)
    Sigma_hat = M_inv

    return mu_hat, Sigma_hat


# ---------------------------------------------------------------------------
# Ticker-level BL optimizer (FinBERT sentiment views)
# ---------------------------------------------------------------------------


class TickerBLOptimizer:
    """Black-Litterman con views a livello di singolo ticker (RIC).

    Usa il sentiment FinBERT per costruire views individuali per ogni ticker:
        P  = matrice identità N×N (una view per ticker)
        Q[i] = π[i] + sentiment_score[i] × view_magnitude
        Ω[i,i] = τ·Σ[i,i] · (1 - confidence[i]) / confidence[i]

    Ticker senza copertura news ottengono confidence→0 → Ω→∞ → view ignorata.

    Parameters
    ----------
    tau : float
        Scaling del prior sulla covarianza (default 0.025).
    delta : float
        Avversione al rischio implicita (default 2.5).
    annualization_factor : int
        252 per rendimenti giornalieri.
    view_magnitude : float
        Max tilt annuo in rendimento per ogni ticker (default 0.05 = 5%).
    max_weight : float
        Peso massimo per singolo ticker (default 0.20).
    min_weight : float
        Peso minimo (default 0.00, no short).
    turnover_penalty : float
        Penalità L1 sul turnover verso equal-weight anchor (default 0.02).
    min_confidence : float
        Soglia minima di confidence per includere una view (default 0.05).
    """

    def __init__(
        self,
        tau: float = 0.025,
        delta: float = 2.5,
        annualization_factor: int = 252,
        view_magnitude: float = 0.05,
        max_weight: float = 0.20,
        min_weight: float = 0.00,
        turnover_penalty: float = 0.02,
        min_confidence: float = 0.05,
    ) -> None:
        self.tau = tau
        self.delta = delta
        self.annualization_factor = annualization_factor
        self.view_magnitude = view_magnitude
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.turnover_penalty = turnover_penalty
        self.min_confidence = min_confidence

    def optimize(
        self,
        ticker_sentiment: "TickerSentiment",  # type: ignore[name-defined]
        prior_weights: pd.Series,
        returns: pd.DataFrame,
    ) -> pd.Series:
        """Calcola i pesi ottimali con views FinBERT per singolo ticker.

        Parameters
        ----------
        ticker_sentiment : TickerSentiment
            Sentiment per ticker. Solo i ticker in ``returns.columns`` vengono usati.
        prior_weights : pd.Series
            Pesi prior (HRP/TrendMomentum), index = RIC.
        returns : pd.DataFrame
            Rendimenti storici, T × N.

        Returns
        -------
        pd.Series
            Pesi finali ∈ [0,1], sum=1, index = RIC.
        """
        from jeanclaude.signals.news.ticker_types import TickerSentiment as _TS

        tickers = list(returns.columns)
        N = len(tickers)

        Sigma = returns.cov().values * self.annualization_factor
        w_mkt = prior_weights.reindex(tickers).fillna(1.0 / N).values
        pi = self.delta * Sigma @ w_mkt  # equilibrium returns, shape (N,)

        # Build views only for tickers with sufficient confidence
        view_idx: list[int] = []
        Q_vals:   list[float] = []
        omega_diag: list[float] = []

        for j, ric in enumerate(tickers):
            conf  = float(ticker_sentiment.confidence.get(ric, 0.0))
            score = float(ticker_sentiment.scores.get(ric, 0.0))
            if conf < self.min_confidence:
                continue
            view_idx.append(j)
            Q_vals.append(float(pi[j]) + score * self.view_magnitude)
            # Idzorek uncertainty: Omega[i] = tau*Sigma[j,j] * (1-conf)/conf
            tau_sigma_jj = self.tau * float(Sigma[j, j])
            omega_diag.append(tau_sigma_jj * (1.0 - conf) / max(conf, 1e-6))

        if not view_idx:
            # No valid views → return prior
            logger.debug("TickerBLOptimizer: no views with confidence≥%.2f — using prior", self.min_confidence)
            return prior_weights.reindex(tickers).fillna(1.0 / N)

        k = len(view_idx)
        P = np.zeros((k, N))
        for row, col in enumerate(view_idx):
            P[row, col] = 1.0
        Q = np.array(Q_vals)
        Omega = np.diag(omega_diag)

        tau_Sigma = self.tau * Sigma
        mu_hat, Sigma_hat = _bl_posterior(pi, tau_Sigma, P, Q, Omega)

        # Project risk_matrix onto PSD cone
        risk_raw = Sigma + Sigma_hat
        eigvals, eigvecs = np.linalg.eigh(risk_raw)
        risk_matrix = eigvecs @ np.diag(np.maximum(eigvals, 1e-8)) @ eigvecs.T
        risk_matrix = (risk_matrix + risk_matrix.T) / 2

        w = cp.Variable(N)
        w_anchor = np.full(N, 1.0 / N)
        objective = cp.Maximize(
            mu_hat @ w
            - (self.delta / 2.0) * cp.quad_form(w, risk_matrix)
            - self.turnover_penalty * cp.norm1(w - w_anchor)
        )
        constraints = [cp.sum(w) == 1, w >= self.min_weight, w <= self.max_weight]
        cp.Problem(objective, constraints).solve()

        if w.value is None:
            logger.warning("TickerBLOptimizer: cvxpy infeasible — fallback a prior")
            return prior_weights.reindex(tickers).fillna(1.0 / N)

        result = pd.Series(np.asarray(w.value), index=tickers)
        logger.debug(
            "TickerBLOptimizer: %d views used | top=%s %.1f%%",
            k, result.idxmax(), result.max() * 100,
        )
        return result
