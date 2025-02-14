from numpy import array
import pandas as pd
from cmdstanpy import CmdStanModel
from datetime import date, timedelta

from .utils import (
    create_d_peak,
    create_mask_logistic,
    create_stan_data,
    fourier_design_matrix,
    get_holiday_dataframe,
)


def fit_holiday_model(
    search_term: str,
    geo="US",
    start_date: str = None,
    train_split: int = 80,
    num_chains: int = 4,
    max_treedepth=10,
    adapt_delta=0.8,
) -> None:
    assert search_term in [
        "chocolate",
        "ramadan",
    ], f"Search term {search_term} not supported. Only ramadan and chocolate are supported."
    if search_term == "chocolate":
        df = pd.read_csv("../bayesian_holidays/src/data/us_chocolate.csv")
        df = df.rename(columns={"Chocolate": "observed"})
        country = "UnitedStates"
    else:
        df = pd.read_csv("../bayesian_holidays/src/data/bangladesh_ramadan.csv")
        df = df.rename(columns={"ramadan": "observed"})
        country = "Bangladesh"
    df["date"] = pd.to_datetime(df["Week"])

    df["observed"] = df["observed"].replace(["<1"], "0").astype(int)
    if start_date is None:
        start_date = df["date"].min()
    else:
        assert pd.to_datetime(start_date) >= df["date"].min()
        start_date = pd.to_datetime(start_date)
    df = df[df["date"] >= pd.to_datetime(start_date)]
    end_date = df["date"].max()

    holiday_years = list(
        range(pd.to_datetime(start_date).year - 1, pd.to_datetime(end_date).year + 1)
    )
    holiday_list = (
        get_holiday_dataframe(years=holiday_years, country=country)
        .sort_values(by="HolidayDate")
        .reset_index()
    )

    train_date = df.date.iloc[int((train_split / 100) * df.date.shape[0])]

    df_train = df[df.date <= train_date]
    df_test = df[df.date > train_date]

    d_peak = create_d_peak(df_train.date, holiday_list)
    d_peak_test = create_d_peak(df_test.date, holiday_list)

    hol_mask = create_mask_logistic(df_train.date, holiday_list)
    hol_mask_test = create_mask_logistic(df_test.date, holiday_list)

    num_modes_year = 3
    T_PER_YEAR = 52.1429

    X_year = fourier_design_matrix(
        array(df_train.date.dt.isocalendar().week.values, dtype=int),
        period=T_PER_YEAR,
        num_modes=num_modes_year,
    )

    X_year_test = fourier_design_matrix(
        array(df_test.date.dt.isocalendar().week.values, dtype=int),
        period=T_PER_YEAR,
        num_modes=num_modes_year,
    )

    holiday_model = CmdStanModel(
        stan_file="../bayesian_holidays/src/holiday_model.stan",
    )

    stan_data = create_stan_data(
        df_train.observed,
        num_modes_year,
        X_year,
        X_year_test,
        d_peak,
        d_peak_test,
        hol_mask,
        hol_mask_test,
    )

    holiday_pathfinder = holiday_model.pathfinder(data=stan_data, seed=42)

    holiday_fit = holiday_model.sample(
        inits=holiday_pathfinder.create_inits(),
        chains=num_chains,
        iter_warmup=1000,
        iter_sampling=1000,
        data=stan_data,
        max_treedepth=max_treedepth,
        adapt_delta=adapt_delta,
        show_progress=True,
        output_dir="./data",
    )
    return df, holiday_fit
