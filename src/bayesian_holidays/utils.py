import contextlib
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from dateutil.easter import easter
from dateutil.relativedelta import MO, SU
from dateutil.relativedelta import relativedelta as rd
from holidays.constants import FEB, JUN, MAY, OCT
from holidays.countries import Bangladesh, UnitedStates
from scipy.special import expit

LOG2 = 0.6931471805599453


def create_stan_data(
    observed: np.ndarray | pd.Series,
    num_modes_year: int,
    X_year: np.ndarray,
    X_year_test: np.ndarray,
    d_peak: np.ndarray,
    d_peak_test: np.ndarray,
    hol_mask: np.ndarray,
    hol_mask_test: np.ndarray,
    use_seasonality: int = 1,
    use_holidays: int = 1,
    use_ar1: int = 1,
) -> dict[str, Any]:
    num_holidays, num_dates = d_peak.shape
    _, num_test_dates = d_peak_test.shape

    return {
        # Observations & Dimensions
        "num_dates": int(num_dates),
        "num_test_dates": int(num_test_dates),
        "num_holidays": int(num_holidays),
        "obs": np.asarray(observed, dtype=np.int64),
        # Seasonality
        "num_modes_year": int(num_modes_year),
        "X_year": np.ascontiguousarray(X_year, dtype=np.float64),
        "X_year_test": np.ascontiguousarray(X_year_test, dtype=np.float64),
        "use_seasonality": int(use_seasonality),
        # Holidays & Tight Priors (single-allocation np.zeros / np.full)
        "use_holidays": int(use_holidays),
        "h_loc_prior_mu": np.zeros(num_holidays, dtype=np.float64),
        "h_loc_prior_sig": np.full(num_holidays, 0.1, dtype=np.float64),
        "h_scale_prior_alpha": np.full(num_holidays, 0.1, dtype=np.float64),
        "h_scale_prior_beta": np.ones(num_holidays, dtype=np.float64),
        "h_shape_prior_mu": np.zeros(num_holidays, dtype=np.float64),
        "h_shape_prior_sig": np.full(num_holidays, 0.25, dtype=np.float64),
        "h_skew_prior_mu": np.zeros(num_holidays, dtype=np.float64),
        "h_skew_prior_sig": np.full(num_holidays, 0.1, dtype=np.float64),
        # Holiday Distance & Mask Matrices
        "d_peak": np.ascontiguousarray(d_peak, dtype=np.float64),
        "d_peak_test": np.ascontiguousarray(d_peak_test, dtype=np.float64),
        "hol_mask": np.ascontiguousarray(hol_mask, dtype=np.float64),
        "hol_mask_test": np.ascontiguousarray(hol_mask_test, dtype=np.float64),
        "use_ar1": int(use_ar1),

    }


class USHolidays(UnitedStates):
    """Custom US holiday calendar tailored for consumer search behavior."""

    def _populate(self, year: int) -> None:
        super()._populate(year)

        # Remove standard federal holidays that don't match target search patterns
        for name in ("Washington's Birthday", "Memorial Day", "Veterans Day"):
            with contextlib.suppress(KeyError):
                self.pop_named(name)

        if year > 2020:
            self.pop(date(year, JUN, 19), None)

        # Add retail/cultural holidays
        self[date(year, FEB, 1) + rd(weekday=MO(+3))] = "Presidents Day"
        self[easter(year)] = "Easter"
        self[date(year, MAY, 1) + rd(weekday=SU(+2))] = "Mothers Day"
        self[date(year, JUN, 1) + rd(weekday=SU(+3))] = "Fathers Day"
        self[date(year, OCT, 31)] = "Halloween"


class BangladeshHolidays(Bangladesh):
    """Custom Bangladesh holiday calendar."""

    def _populate(self, year: int) -> None:
        super()._populate(year)


def fourier_design_matrix(
    times: np.ndarray | pd.Series, period: float = 365.25, num_modes: int = 1
) -> np.ndarray:
    """
    Build a Fourier design matrix of shape (2 * num_modes, len(times))
    using vectorized 2D NumPy broadcasting.
    """
    if num_modes <= 0:
        return np.empty((0, len(times)), dtype=np.float64)

    t = np.asarray(times, dtype=np.float64)
    modes = np.arange(1, num_modes + 1, dtype=np.float64)[:, None]
    angles = (2.0 * np.pi / period) * modes * t[None, :]

    out = np.empty((2 * num_modes, t.shape[0]), dtype=np.float64)
    out[0::2] = np.cos(angles)
    out[1::2] = -np.sin(angles)
    return out


def create_mask_logistic(
    times: pd.Series | pd.DatetimeIndex | np.ndarray, holiday_list: pd.DataFrame
) -> np.ndarray:
    """
    Vectorized logistic window mask of shape (num_holidays, num_dates).
    Uses integer day conversions and 2D NumPy broadcasting instead of .iterrows().
    """
    times_dt = pd.to_datetime(times)
    num_dates = len(times_dt)
    num_holidays = int(holiday_list["HolidayId"].max())
    mask_array = np.zeros((num_holidays, num_dates), dtype=np.float64)

    # Filter to holidays inside the date range once
    dmin, dmax = times_dt.min(), times_dt.max()
    valid = holiday_list[
        (holiday_list["HolidayDate"] >= dmin) & (holiday_list["HolidayDate"] <= dmax)
    ].dropna(subset=["days_behind_diff", "days_ahead_diff"])

    if valid.empty:
        return mask_array

    # Convert dates and timedeltas to integer days since epoch for pure NumPy math
    times_days = times_dt.to_numpy(dtype="datetime64[D]").astype(np.int64)

    hol_days = valid["HolidayDate"].to_numpy(dtype="datetime64[D]").astype(np.int64)
    behind_days = valid["days_behind_diff"].dt.days.to_numpy(dtype=np.int64)
    ahead_days = valid["days_ahead_diff"].dt.days.to_numpy(dtype=np.int64)
    hol_ids = valid["HolidayId"].to_numpy(dtype=np.int64) - 1

    # Window boundaries per holiday occurrence: shape (num_occurrences,)
    win_start = hol_days - behind_days
    win_end = hol_days + ahead_days
    alpha = LOG2 / (0.01 * (ahead_days + behind_days))

    # Broadcast across (num_occurrences, num_dates)
    in_win = (times_days[None, :] >= win_start[:, None]) & (
        times_days[None, :] <= win_end[:, None]
    )

    # Note: In your original code, xL = times[ind] - days_behind_diff, so
    # (times[ind] - xL) simplifies identically to days_behind_diff, and
    # -(times[ind] - xU) simplifies identically to days_ahead_diff.
    # Preserving the exact date-relative formula below:
    t_rel_start = times_days[None, :] - win_start[:, None]
    t_rel_end = times_days[None, :] - win_end[:, None]

    weights = np.where(
        in_win,
        expit(alpha[:, None] * t_rel_start) * expit(-alpha[:, None] * t_rel_end),
        0.0,
    )

    # Accumulate occurrences by HolidayId
    np.add.at(mask_array, hol_ids, weights)
    return mask_array


def create_d_peak(
    times: pd.Series | pd.DatetimeIndex | np.ndarray, holiday_list: pd.DataFrame
) -> np.ndarray:
    """
    Vectorized distance (in weeks) from each date to the nearest occurrence
    of each HolidayId using binary search (np.searchsorted).
    """
    times_days = pd.to_datetime(times).to_numpy(dtype="datetime64[D]").astype(np.int64)

    # Sort holiday_list once so each group's dates are strictly ascending
    sorted_hols = holiday_list.sort_values("HolidayDate")
    unique_holiday_ids = sorted_hols["HolidayId"].unique()

    grouped_dates = {
        hid: grp["HolidayDate"].to_numpy(dtype="datetime64[D]").astype(np.int64)
        for hid, grp in sorted_hols.groupby("HolidayId", sort=False)
    }

    def nearest_week_diff(hol_days: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(hol_days, times_days)
        idx_right = np.clip(idx, 0, len(hol_days) - 1)
        idx_left = np.clip(idx - 1, 0, len(hol_days) - 1)

        diff_right = times_days - hol_days[idx_right]
        diff_left = times_days - hol_days[idx_left]

        nearest_diff = np.where(
            np.abs(diff_left) <= np.abs(diff_right), diff_left, diff_right
        )
        return np.asarray(nearest_diff / 7.0, dtype=np.float64)

    return np.vstack(
        [nearest_week_diff(grouped_dates[hid]) for hid in unique_holiday_ids]
    )


def get_holiday_dataframe(years: list[int], country: str) -> pd.DataFrame:
    """
    Builds the holiday dataframe using vectorized pd.factorize() instead of
    looping over unique holiday names.
    """
    holiday_cls = {
        "UnitedStates": USHolidays,
        "Bangladesh": BangladeshHolidays,
    }.get(country)

    if holiday_cls is None:
        raise ValueError(f"Unsupported country: {country}")

    hol_dict = holiday_cls(years=years, observed=False)
    df_holiday = (
        pd.DataFrame(list(hol_dict.items()), columns=["HolidayDate", "HolidayName"])
        .assign(HolidayDate=lambda df: pd.to_datetime(df["HolidayDate"]))
        .sort_values("HolidayDate", ignore_index=True)
    )

    df_holiday["HolidayId"] = pd.factorize(df_holiday["HolidayName"])[0] + 1
    df_holiday["days_behind_diff"] = df_holiday["HolidayDate"].diff(1)
    df_holiday["days_ahead_diff"] = -df_holiday["HolidayDate"].diff(-1)
    return df_holiday
