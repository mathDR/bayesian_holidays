import numpy as np
from scipy.special import expit
import pandas as pd
from typing import Dict, List

from datetime import date, timedelta
import holidays
from dateutil.easter import easter
from dateutil.relativedelta import relativedelta as rd, MO, SU, FR

from holidays.constants import JAN, FEB, MAR, APR, MAY, JUN, JUL, AUG, SEP
from holidays.constants import OCT, NOV, DEC
from holidays.countries import UnitedStates

LOG2 = 0.6931471805599453


def create_stan_data(
    observed: np.ndarray,
    num_modes_year: int,
    X_year: np.ndarray,
    X_year_test: np.ndarray,
    d_peak: np.ndarray,
    d_peak_test: np.ndarray,
    hol_mask: np.ndarray,
    hol_mask_test: np.ndarray,
    use_seasonality: int = 1,
    use_holidays: int = 1,
) -> Dict:
    num_holidays, num_dates = d_peak.shape
    _, num_test_dates = d_peak_test.shape
    stan_data = {
        "num_dates": num_dates,
        "num_test_dates": num_test_dates,
        "num_holidays": num_holidays,
        "obs": observed,
    }

    # Seasonality
    stan_data["num_modes_year"] = num_modes_year
    stan_data["X_year"] = X_year
    stan_data["X_year_test"] = X_year_test
    stan_data["use_seasonality"] = use_seasonality

    # Holidays - tight priors for holidays
    stan_data["use_holidays"] = use_holidays
    stan_data["h_loc_prior_mu"] = 0.0 * np.ones(num_holidays)
    stan_data["h_loc_prior_sig"] = 0.1 * np.ones(num_holidays)

    stan_data["h_scale_prior_alpha"] = 0.1 * np.ones(num_holidays)
    stan_data["h_scale_prior_beta"] = 1.0 * np.ones(num_holidays)
    stan_data["h_shape_prior_mu"] = 0.0 * np.ones(num_holidays)
    stan_data["h_shape_prior_sig"] = 0.25 * np.ones(num_holidays)
    stan_data["h_skew_prior_mu"] = 0.0 * np.ones(num_holidays)
    stan_data["h_skew_prior_sig"] = 0.1 * np.ones(num_holidays)

    stan_data["d_peak"] = d_peak
    stan_data["d_peak_test"] = d_peak_test
    stan_data["hol_mask"] = hol_mask
    stan_data["hol_mask_test"] = hol_mask_test

    return stan_data


class USHolidays(UnitedStates):
    def _populate(self, year):
        # Populate the holiday list with the default US holidays
        UnitedStates._populate(self, year)

        # Remove Washingtons Birthday
        try:
            self.pop_named("Washington's Birthday")
        except KeyError as e:
            pass
        # Remove Memorial Day
        try:
            self.pop_named("Memorial Day")
        except KeyError as e:
            pass

        # Add Presidents Day -- 3rd monday in Februray
        try:
            self[date(year, FEB, 1) + rd(weekday=MO(+3))] = "Presidents Day"
        except KeyError as e:
            pass

        # Add Easter
        self[easter(year)] = "Easter"

        # Add Mothers Day -- 2nd sunday in may
        self[date(year, MAY, 1) + rd(weekday=SU(+2))] = "Mothers Day"

        # remove Juneteenth
        if year > 2020:
            try:
                self.pop(date(year, JUN, 19))
            except KeyError as e:
                pass
        # Add Fathers Day -- 3rd sunday in june
        self[date(year, JUN, 1) + rd(weekday=SU(+3))] = "Fathers Day"

        # Add Halloween - Oct 31
        self[date(year, OCT, 31)] = "Halloween"

        # Remove Veterans/Armistice Day
        try:
            self.pop_named("Veterans Day")
        except KeyError as e:
            pass


def fourier_design_matrix(
    times: np.ndarray, period: float = 365.25, num_modes: int = 1
):
    """
    times: input times
    period:  the period of the fourier modes
    num_modes:  how many modes to include
    """
    columns = []
    columns.extend(
        np.cos(2.0 * np.pi * (m * times / period + x))
        for m in range(1, num_modes + 1)
        for x in [0, 0.25]
    )
    return np.asarray(np.stack(columns))


def create_mask_logistic(times: np.ndarray, holiday_list: pd.DataFrame):
    """
    This function produces a continuous "mask" that is the
    size of num_holidays x num_dates.

    For a given holiday, hol, assume the holiday is on date hol_date.
    Now assume the "closest" holidays near hol_date (both before and after)
    are on dates prev_hol_date and post_hol_date, respectively (note,
    unless there is only one holiday in the calendar, these will be different
    holidays; both from the given holiday, and each other).

    Then the mask is given by
      logistic(alpha*(dates-prev_hol_date)) * logistic(-alpha*(dates-post_hol_date))
    where alpha denotes how much probability is in the "tails" (i.e. outside of
        [prev_hol_date, post_hol_date])
      alpha ~ log(2) / (rho * (post_hol_date-prev_hol_date)), where
      rho = prob outside of tails.

    Note, in scipy, the logistic() function is called as expit()
    """
    num_holidays = holiday_list.HolidayId.max()
    num_dates = times.shape[0]
    mask_array = np.zeros((num_holidays, num_dates))
    dmin = times.min()
    dmax = times.max()

    for row in holiday_list.iterrows():
        series = row[1]
        if (series["HolidayDate"] >= dmin) and (series["HolidayDate"] <= dmax):
            ind = np.logical_and(
                times
                >= series["HolidayDate"]
                - pd.to_timedelta(series.days_behind_diff, unit="d"),
                times
                <= series["HolidayDate"]
                + pd.to_timedelta(series.days_ahead_diff, unit="d"),
            )
            xL = times[ind] - pd.to_timedelta(series.days_behind_diff, unit="d")
            xU = times[ind] + pd.to_timedelta(series.days_ahead_diff, unit="d")
            alpha = LOG2 / (0.01 * (xU - xL).dt.days)
            mask_array[
                series.HolidayId - 1,
                ind,
            ] = expit(
                alpha * ((times[ind] - xL)).dt.days.values
            ) * expit(-alpha * ((times[ind] - xU)).dt.days.values)

    return mask_array


def create_d_peak(times: np.ndarray, holiday_list: pd.DataFrame):
    """
    times: input times
    holiday_list: a dataframe consisting of columns:
        HolidayId (int) index of holiday 1,...,num_holidays
        holidayname: (str) names of holidays (Easter, Christmas, etc.)
        HolidayDate: (pd.datetime) the date of the holiday

    This function produces an array of the "distance" between each holiday
    date and every date in times.
    """
    unique_holiday_ids = holiday_list.sort_values(by=["HolidayDate"])[
        "HolidayId"
    ].unique()
    nearest = lambda x, v: abs(v - x).idxmin()
    daysdiff = lambda x, v: (x - v[nearest(x, v)]).days
    d_peak = []
    # get difference from official date in days for each holiday
    for hol_id in unique_holiday_ids:
        dfh = holiday_list[holiday_list["HolidayId"] == hol_id]
        i = [daysdiff(d, dfh["HolidayDate"]) for d in times]
        d_peak.append(i)
    return np.asarray(d_peak) / 7.0


def get_holiday_dataframe(years: List[int]):
    df_holiday = (
        pd.DataFrame.from_dict(
            USHolidays(years=years, observed=False),
            orient="index",
            columns=["HolidayName"],
        )
        .reset_index()
        .rename(columns={"index": "HolidayDate"})
        .sort_values(by=["HolidayDate"])
    )

    df_holiday["HolidayDate"] = pd.to_datetime(df_holiday["HolidayDate"])
    z = dict(
        zip(
            df_holiday.head(
                df_holiday.HolidayName.unique().shape[0]
            ).HolidayName.values,
            range(1, df_holiday.HolidayName.unique().shape[0] + 1),
        )
    )
    df_holiday["HolidayId"] = 0
    for hol, hol_id in z.items():
        df_holiday.loc[df_holiday.HolidayName == hol, "HolidayId"] = hol_id
    df_holiday["days_behind_diff"] = df_holiday.HolidayDate.diff(periods=1)
    df_holiday["days_ahead_diff"] = -1 * df_holiday.HolidayDate.diff(periods=-1)
    return df_holiday
