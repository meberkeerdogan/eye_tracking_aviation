"""Polynomial ridge regression gaze mapper with serialisation."""

from __future__ import annotations

import base64
import logging
import pickle
from typing import Optional

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

logger = logging.getLogger(__name__)


class EMAFilter:
    """Exponential moving average for 2-D gaze point smoothing."""

    def __init__(self, alpha: float = 0.3) -> None:
        self.alpha = alpha
        self._x: Optional[float] = None
        self._y: Optional[float] = None

    def update(self, x: float, y: float) -> tuple[float, float]:
        if self._x is None:
            self._x, self._y = x, y
        else:
            self._x = self.alpha * x + (1.0 - self.alpha) * self._x
            self._y = self.alpha * y + (1.0 - self.alpha) * self._y
        return self._x, self._y  # type: ignore[return-value]

    def reset(self) -> None:
        self._x = None
        self._y = None


class GazeMapper:
    """Maps eye feature vectors to normalised screen coordinates (0-1).

    Uses separate polynomial ridge regression models for X and Y to allow
    better fit along each axis independently.

    Degree-2 polynomial with Ridge regularisation is a robust MVP choice:
    it captures simple non-linearities without over-fitting on the small
    (~9 × 30 = 270 sample) calibration dataset.
    """

    def __init__(self, degree: int = 2, alpha: float = 1.0) -> None:
        self.degree = degree
        self.alpha = alpha
        self._pipe_x: Optional[Pipeline] = None
        self._pipe_y: Optional[Pipeline] = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, features: np.ndarray, targets: np.ndarray) -> float:
        """Fit and return RMS prediction error on the training data.

        Args:
            features: shape (N, F)
            targets:  shape (N, 2) – (screen_x_norm, screen_y_norm)

        Returns:
            RMS error in normalised screen units.
        """

        def make_pipe() -> Pipeline:
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("poly", PolynomialFeatures(degree=self.degree, include_bias=False)),
                    ("reg", Ridge(alpha=self.alpha)),
                ]
            )

        self._pipe_x = make_pipe()
        self._pipe_y = make_pipe()
        self._pipe_x.fit(features, targets[:, 0])
        self._pipe_y.fit(features, targets[:, 1])
        self._is_fitted = True

        pred_x = self._pipe_x.predict(features)
        pred_y = self._pipe_y.predict(features)
        residuals = np.sqrt((pred_x - targets[:, 0]) ** 2 + (pred_y - targets[:, 1]) ** 2)
        rms = float(np.sqrt(np.mean(residuals ** 2)))
        logger.info("GazeMapper fitted.  RMS=%.4f  N=%d", rms, len(features))
        return rms

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, features: np.ndarray) -> tuple[float, float]:
        """Return (gaze_x_norm, gaze_y_norm) clamped to [0, 1]."""
        if not self._is_fitted:
            raise RuntimeError("GazeMapper has not been fitted yet.")
        x = float(self._pipe_x.predict(features.reshape(1, -1))[0])  # type: ignore[union-attr]
        y = float(self._pipe_y.predict(features.reshape(1, -1))[0])  # type: ignore[union-attr]
        return float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, 1.0))

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        if not self._is_fitted:
            return {}
        return {
            "type": "polynomial_ridge",
            "degree": self.degree,
            "alpha": self.alpha,
            "pipe_x": base64.b64encode(pickle.dumps(self._pipe_x)).decode(),
            "pipe_y": base64.b64encode(pickle.dumps(self._pipe_y)).decode(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GazeMapper":
        mapper = cls(degree=data["degree"], alpha=data["alpha"])
        mapper._pipe_x = pickle.loads(base64.b64decode(data["pipe_x"]))
        mapper._pipe_y = pickle.loads(base64.b64decode(data["pipe_y"]))
        mapper._is_fitted = True
        return mapper
