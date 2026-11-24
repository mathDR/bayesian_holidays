from datetime import date

import numpy as np
import pandas as pd
import pytest

from bayesian_holidays.utils import (
    BangladeshHolidays,
    USHolidays,
    create_d_peak,
    create_mask_logistic,
    create_stan_data,
    fourier_design_matrix,
    get_holiday_dataframe,
)


class TestHolidayCalendars:
    def test_us_holidays_custom_additions_and_removals(self) -> None:
        us_hols = USHolidays(years=list((2022, 2023)), observed=False)
        names = set(us_hols.values())

        # Custom additions
        assert "Easter" in names
        assert "Mothers Day" in names
        assert "Fathers Day" in names
        assert "Halloween" in names
        assert "Presidents Day" in names
        assert us_hols[date(2023, 10, 31)] == "Halloween"

        # Federal holidays explicitly removed
        assert "Washington's Birthday" not in names
        assert "Memorial Day" not in names
        assert "Veterans Day" not in names
        assert date(2023, 6, 19) not in us_hols

    def test_bangladesh_holidays_populates(self) -> None:
        bd_hols = BangladeshHolidays(years=list((2022, 2023)), observed=False)
        assert len(bd_hols) > 0

    def test_get_holiday_dataframe_structure_and_ids(self) -> None:
        df_hol = get_holiday_dataframe(years=list((2022, 2023)), country="UnitedStates")

        expected_cols = {
            "HolidayDate",
            "HolidayName",
            "HolidayId",
            "days_behind_diff",
            "days_ahead_diff",
        }
        assert expected_cols.issubset(df_hol.columns)
        assert df_hol["HolidayDate"].is_monotonic_increasing

        # Same holiday across 2022 and 2023 must share the exact same HolidayId
        halloween_ids = df_hol.loc[
            df_hol["HolidayName"] == "Halloween", "HolidayId"
        ].to_numpy()
        assert len(halloween_ids) == 2
        assert np.all(halloween_ids == halloween_ids.item(0))

    def test_get_holiday_dataframe_invalid_country(self) -> None:
        with pytest.raises(ValueError, match="Unsupported country"):
            get_holiday_dataframe(years=list((2022, 2023)), country="Atlantis")


class TestFeatureEngineering:
    def test_fourier_design_matrix_values_and_shape(self) -> None:
        times = np.array((0.0, 1.0, 2.0, 3.0))
        period = 4.0
        num_modes = 2

        X = fourier_design_matrix(times, period=period, num_modes=num_modes)
        assert X.shape == (4, 4)

        # Compare against explicit phase-shifted cosine formula
        expected = np.vstack(
            tuple(
                np.cos(2.0 * np.pi * (m * times / period + shift))
                for m in range(1, num_modes + 1)
                for shift in (0.0, 0.25)
            )
        )
        np.testing.assert_allclose(X, expected, atol=1e-12)

    def test_fourier_design_matrix_zero_modes(self) -> None:
        times = np.arange(10, dtype=np.float64)
        X = fourier_design_matrix(times, period=52.0, num_modes=0)
        assert X.shape == (0, 10)

    def test_create_d_peak_nearest_holiday_distance(self) -> None:
        holiday_list = pd.DataFrame(
            {
                "HolidayId": np.array((1, 1, 2)),
                "HolidayName": ("HolA", "HolA", "HolB"),
                "HolidayDate": pd.to_datetime(
                    ("2023-01-10", "2023-01-31", "2023-01-20")
                ),
            }
        )
        times = pd.to_datetime(("2023-01-03", "2023-01-10", "2023-01-24", "2023-01-31"))
        d_peak = create_d_peak(times, holiday_list)

        assert d_peak.shape == (2, 4)
        # HolA dates: Jan 10 and Jan 31 -> distances in weeks: (-1.0, 0.0, -1.0, 0.0)
        np.testing.assert_allclose(d_peak[0], (-1.0, 0.0, -1.0, 0.0))
        # HolB date: Jan 20 -> Jan 24 is +4 days (+4/7 weeks)
        assert d_peak[1, 2] == pytest.approx(4.0 / 7.0)

    def test_create_mask_logistic_bounds_and_peak(self) -> None:
        holiday_list = pd.DataFrame(
            {
                "HolidayId": np.array((1,)),
                "HolidayName": ("TestHol",),
                "HolidayDate": pd.to_datetime(("2023-06-15",)),
                "days_behind_diff": pd.to_timedelta(np.array((10,)), unit="D"),
                "days_ahead_diff": pd.to_timedelta(np.array((10,)), unit="D"),
            }
        )
        times = pd.date_range("2023-06-01", "2023-06-30", freq="D")
        mask = create_mask_logistic(times, holiday_list)

        assert mask.shape == (1, len(times))

        before_win = np.asarray(times < pd.Timestamp("2023-06-05"), dtype=np.bool_)
        after_win = np.asarray(times > pd.Timestamp("2023-06-25"), dtype=np.bool_)
        assert np.all(mask[0, before_win] == 0.0)
        assert np.all(mask[0, after_win] == 0.0)

        # Peak on the holiday date itself should be close to 1.0
        hol_idx = int(np.flatnonzero(times == pd.Timestamp("2023-06-15")).item(0))
        assert mask[0, hol_idx] > 0.99

    def test_create_stan_data_shapes_and_contiguity(self) -> None:
        num_holidays, num_train, num_test, num_modes = 3, 20, 5, 2
        obs = np.arange(num_train, dtype=np.int64)
        X_all = np.ones((2 * num_modes, num_train + num_test))
        d_all = np.zeros((num_holidays, num_train + num_test))
        m_all = np.ones((num_holidays, num_train + num_test))

        # Pass non-contiguous slices to verify np.ascontiguousarray conversion
        stan_data = create_stan_data(
            observed=obs,
            num_modes_year=num_modes,
            X_year=X_all[:, :num_train],
            X_year_test=X_all[:, num_train:],
            d_peak=d_all[:, :num_train],
            d_peak_test=d_all[:, num_train:],
            hol_mask=m_all[:, :num_train],
            hol_mask_test=m_all[:, num_train:],
            use_seasonality=1,
            use_holidays=0,
        )

        assert stan_data["num_dates"] == num_train
        assert stan_data["num_test_dates"] == num_test
        assert stan_data["num_holidays"] == num_holidays
        assert stan_data["use_holidays"] == 0
        assert stan_data["X_year"].flags["C_CONTIGUOUS"]
        assert stan_data["d_peak_test"].flags["C_CONTIGUOUS"]
        assert stan_data["h_loc_prior_mu"].shape == (num_holidays,)
