from unittest.mock import MagicMock

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit

# Use non-interactive backend so plots render cleanly in CI/headless terminals
matplotlib.use("Agg")

from bayesian_holidays.plot_utils import (
    get_holiday_lift,
    plot_decomposition,
    plot_individual_holidays,
)


def _reference_dense_holiday_lift(
    h_skew: np.ndarray,
    h_shape: np.ndarray,
    h_scale: np.ndarray,
    h_loc: np.ndarray,
    intensity: np.ndarray,
    d_peak: np.ndarray,
    hol_mask: np.ndarray,
    return_sum: bool = True,
) -> np.ndarray:
    """Dense 3D reference implementation to validate sparse active-window slicing."""
    z = (d_peak[None, :, :] - h_loc[:, :, None]) / h_scale[:, :, None]
    tdd = (
        2.0
        * intensity[:, :, None]
        * np.exp(-((np.square(z) + 1e-12) ** h_shape[:, :, None]))
        * expit(h_skew[:, :, None] * z)
        * hol_mask[None, :, :]
    )
    result = tdd.sum(axis=1) if return_sum else tdd
    return np.asarray(result, dtype=np.float64)


class TestPlotUtils:
    def test_get_holiday_lift_matches_dense_reference(self) -> None:
        rng = np.random.default_rng(123)
        num_draws, num_holidays, num_dates = 15, 4, 50

        h_skew = rng.normal(0.0, 0.5, size=(num_draws, num_holidays))
        h_shape = np.exp(rng.normal(0.0, 0.2, size=(num_draws, num_holidays)))
        h_scale = rng.gamma(2.0, 1.0, size=(num_draws, num_holidays))
        h_loc = rng.normal(0.0, 0.2, size=(num_draws, num_holidays))
        intensity = rng.gamma(1.5, 0.5, size=(num_draws, num_holidays))
        d_peak = rng.uniform(-4.0, 4.0, size=(num_holidays, num_dates))

        # Sparse mask with zeros and fractional logistic weights
        hol_mask = np.zeros((num_holidays, num_dates))
        hol_mask[0, 5:15] = np.linspace(0.01, 0.99, 10)
        hol_mask[1, 20:35] = 1.0
        # Holiday 2 intentionally all-zero (inactive in window)
        hol_mask[3, 40:48] = 0.75

        sparse_3d = get_holiday_lift(
            h_skew,
            h_shape,
            h_scale,
            h_loc,
            intensity,
            d_peak,
            hol_mask,
            return_sum=False,
        )
        dense_3d = _reference_dense_holiday_lift(
            h_skew,
            h_shape,
            h_scale,
            h_loc,
            intensity,
            d_peak,
            hol_mask,
            return_sum=False,
        )
        np.testing.assert_allclose(sparse_3d, dense_3d, rtol=1e-12, atol=1e-12)

        sparse_sum = get_holiday_lift(
            h_skew,
            h_shape,
            h_scale,
            h_loc,
            intensity,
            d_peak,
            hol_mask,
            return_sum=True,
        )
        np.testing.assert_allclose(
            sparse_sum, dense_3d.sum(axis=1), rtol=1e-12, atol=1e-12
        )

    def test_plot_individual_holidays_renders_grid(self, monkeypatch: object) -> None:
        monkeypatch.setattr(plt, "show", lambda: None)  # type: ignore[attr-defined]
        rng = np.random.default_rng(7)
        times = pd.date_range("2023-01-01", periods=30, freq="W")
        tdd = rng.normal(0.0, 0.1, size=(20, 4, 30))
        hol_names = ["New Year", "Easter", "Thanksgiving", "Christmas"]

        fig, axes = plot_individual_holidays(
            times, tdd, hol_names, ncols=2, exp_scale=True
        )
        assert axes.shape == (2, 2)
        plt.close(fig)

    def test_plot_decomposition_renders_four_panels(self, monkeypatch: object) -> None:
        monkeypatch.setattr(plt, "show", lambda: None)  # type: ignore[attr-defined]
        rng = np.random.default_rng(99)
        num_draws, num_total = 25, 50
        split_idx = 40  # 80% of 50

        df = pd.DataFrame(
            {
                "date": pd.date_range("2022-01-02", periods=num_total, freq="W"),
                "observed": rng.poisson(50, size=num_total),
            }
        )

        # Mock CmdStanMCMC.stan_variable outputs
        mock_vars = {
            "log_baseline_real": rng.normal(3.8, 0.1, size=num_draws),
            "log_seasonality": rng.normal(0.0, 0.1, size=(num_draws, split_idx)),
            "holiday_effect": rng.uniform(0.0, 0.3, size=(num_draws, split_idx)),
            "log_obs_mean": rng.normal(3.9, 0.15, size=(num_draws, split_idx)),
            "test_log_seasonality": rng.normal(
                0.0, 0.1, size=(num_draws, num_total - split_idx)
            ),
            "test_holiday_effect": rng.uniform(
                0.0, 0.3, size=(num_draws, num_total - split_idx)
            ),
            "test_log_obsmean": rng.normal(
                3.9, 0.15, size=(num_draws, num_total - split_idx)
            ),
        }
        mock_fit = MagicMock()
        mock_fit.stan_variable.side_effect = lambda name: mock_vars[name]

        fig, axes = plot_decomposition(
            df, mock_fit, name="Unit Test Decomposition", train_split_percentage=80
        )
        assert len(axes) == 4
        plt.close(fig)
