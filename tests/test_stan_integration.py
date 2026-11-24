from importlib.resources import as_file, files
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from cmdstanpy import CmdStanModel

from bayesian_holidays.utils import (
    create_d_peak,
    create_mask_logistic,
    create_stan_data,
    fourier_design_matrix,
    get_holiday_dataframe,
)


def _cmdstan_available() -> bool:
    """Check if CmdStan is installed so CI environments without C++ toolchains skip gracefully."""
    try:
        from cmdstanpy.utils import cmdstan_path

        cmdstan_path()
        return True
    except ValueError:
        return False


@pytest.fixture(scope="module")
def compiled_holiday_model() -> CmdStanModel:
    """Compile holiday_model.stan once for all integration tests in this module."""
    stan_resource = files("bayesian_holidays") / "holiday_model.stan"
    with as_file(stan_resource) as stan_path:
        return CmdStanModel(stan_file=str(stan_path))


@pytest.fixture(scope="module")
def synthetic_stan_data() -> dict[str, object]:
    """
    Build a 2-year weekly synthetic dataset (80 train weeks, 24 test weeks)
    with a pronounced spike around Halloween (HolidayId for Halloween).
    """
    dates = pd.date_range("2022-01-02", periods=104, freq="W-SUN")
    split_idx = 80
    rng = np.random.default_rng(42)

    holiday_list = get_holiday_dataframe(
        years=list((2021, 2022, 2023, 2024)), country="UnitedStates"
    )
    d_peak_all = create_d_peak(dates, holiday_list)
    hol_mask_all = create_mask_logistic(dates, holiday_list)

    week_of_year = dates.isocalendar().week.to_numpy(dtype=np.int64)
    num_modes = 2
    X_year_all = fourier_design_matrix(
        week_of_year, period=365.25 / 7.0, num_modes=num_modes
    )

    # Generate synthetic Poisson counts with baseline ~40 and a 2.5x Halloween boost
    halloween_id = int(
        holiday_list.loc[holiday_list["HolidayName"] == "Halloween", "HolidayId"]
        .to_numpy()
        .item(0)
    )

    halloween_idx = halloween_id - 1
    halloween_bump = (
        0.9
        * np.exp(-np.square(d_peak_all[halloween_idx] / 1.5))
        * hol_mask_all[halloween_idx]
    )
    log_rate = np.log(40.0) + halloween_bump
    obs_all = rng.poisson(np.exp(log_rate))

    return create_stan_data(
        observed=obs_all[:split_idx],
        num_modes_year=num_modes,
        X_year=X_year_all[:, :split_idx],
        X_year_test=X_year_all[:, split_idx:],
        d_peak=d_peak_all[:, :split_idx],
        d_peak_test=d_peak_all[:, split_idx:],
        hol_mask=hol_mask_all[:, :split_idx],
        hol_mask_test=hol_mask_all[:, split_idx:],
        use_seasonality=1,
        use_holidays=1,
    )


@pytest.mark.integration
@pytest.mark.skipif(not _cmdstan_available(), reason="CmdStan toolchain not installed")
class TestStanModelIntegration:
    def test_full_model_sampling_and_sparse_mask_zeros(
        self,
        compiled_holiday_model: CmdStanModel,
        synthetic_stan_data: dict[str, object],
        tmp_path: Path,
    ) -> None:
        fit = compiled_holiday_model.sample(
            data=synthetic_stan_data,
            chains=2,
            parallel_chains=2,
            iter_warmup=150,
            iter_sampling=150,
            seed=42,
            show_progress=False,
            output_dir=str(tmp_path),
        )

        # 1. Verify posterior shapes
        holiday_effect = fit.stan_variable("holiday_effect")
        test_obs = fit.stan_variable("test_obs")
        assert holiday_effect.shape == (300, 80)
        assert test_obs.shape == (300, 24)
        assert np.all(np.isfinite(holiday_effect))
        assert np.all(test_obs >= 0)

        # 2. Verify sparse run-slicing produces exact 0.0 where all holiday masks are 0
        hol_mask = np.asarray(synthetic_stan_data["hol_mask"])
        all_masked_out = np.all(hol_mask == 0.0, axis=0)
        if np.any(all_masked_out):
            np.testing.assert_array_equal(holiday_effect[:, all_masked_out], 0.0)

        # 3. Verify model recovers a positive lift during Halloween weeks
        mean_lift = holiday_effect.mean(axis=0)
        assert mean_lift.max() > 0.3

    def test_holiday_toggle_off_zeroes_holiday_effect(
        self,
        compiled_holiday_model: CmdStanModel,
        synthetic_stan_data: dict[str, object],
        tmp_path: Path,
    ) -> None:
        ablation_data = dict(synthetic_stan_data)
        ablation_data["use_holidays"] = 0

        fit = compiled_holiday_model.sample(
            data=ablation_data,
            chains=1,
            iter_warmup=50,
            iter_sampling=50,
            seed=42,
            show_progress=False,
            output_dir=str(tmp_path),
        )

        np.testing.assert_array_equal(fit.stan_variable("holiday_effect"), 0.0)
        np.testing.assert_array_equal(fit.stan_variable("test_holiday_effect"), 0.0)
