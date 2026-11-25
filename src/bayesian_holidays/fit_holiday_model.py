"""Fitting a Holiday model to data."""

from collections.abc import Mapping
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanMCMC, CmdStanModel

from bayesian_holidays.utils import (
    create_d_peak,
    create_mask_logistic,
    create_stan_data,
    fourier_design_matrix,
    get_holiday_dataframe,
)

# Registry mapping search terms to packaged CSV filenames, target columns, and countries
DATASET_CONFIG: dict[str, tuple[str, str, str]] = {
    "chocolate": ("us_chocolate.csv", "Chocolate", "UnitedStates"),
    "ramadan": ("bangladesh_ramadan.csv", "ramadan", "Bangladesh"),
}

T_PER_YEAR = 365.25 / 7.0  # Exact weeks per year (~52.1786)
NUM_MODES_YEAR = 3


def fit_holiday_model(
    search_term: str,
    geo: str = "US",
    start_date: str | None = None,
    train_split_percentage: int = 80,
    num_chains: int = 4,
    max_treedepth: int = 10,
    adapt_delta: float = 0.8,
    seed: int = 42,
    output_dir: str | Path = "./data",
) -> tuple[pd.DataFrame, CmdStanMCMC]:
    if search_term not in DATASET_CONFIG:
        raise ValueError(
            f"Search term '{search_term}' not supported. "
            f"Supported terms: {list(DATASET_CONFIG)}"
        )

    csv_name, target_col, country = DATASET_CONFIG[search_term]

    # Resolve package paths dynamically instead of hardcoded user paths
    # Resolve package paths directly inside bayesian_holidays
    pkg_root = files("bayesian_holidays")
    csv_resource = pkg_root / "data" / csv_name
    stan_path = pkg_root / "holiday_model.stan"

    # Load and clean dataset in a single pass
    with as_file(csv_resource) as csv_path:
        raw_df = pd.read_csv(Path(csv_path))
        matched_cols = [
            col
            for col in raw_df.columns
            if col.strip().lower().startswith(target_col.lower())
        ]
        if not matched_cols:
            raise ValueError(
                f"Column starting with '{target_col}' not found in {csv_name}. "
                f"Available columns: {list(raw_df.columns)}"
            )

        df = (
            raw_df[["Time", matched_cols[0]]]
            .rename(columns={matched_cols[0]: "observed"})
            .assign(
                date=lambda d: pd.to_datetime(d["Time"]),
                observed=lambda d: pd.to_numeric(
                    d["observed"].replace("<1", "0"), errors="raise"
                ).astype(np.int64),
            )
            .sort_values("date", ignore_index=True)
        )


    min_date = df["date"].min()
    if start_date is not None:
        parsed_start = pd.to_datetime(start_date)
        if parsed_start < min_date:
            raise ValueError(
                f"start_date ({parsed_start}) precedes dataset min ({min_date})."
            )
        df = df[df["date"] >= parsed_start].reset_index(drop=True)

    # Fetch holiday calendar covering one year before start through one year after end
    start_year = df["date"].min().year - 1
    end_year = df["date"].max().year + 1
    holiday_list = get_holiday_dataframe(
        years=list(range(start_year, end_year + 1)),
        country=country,
    )

    # Compute d_peak, hol_mask, and Fourier features ONCE on the full timeline, then slice
    split_idx = int((train_split_percentage / 100.0) * len(df))
    all_dates = df["date"].to_numpy()

    d_peak_all = create_d_peak(all_dates, holiday_list)
    hol_mask_all = create_mask_logistic(all_dates, holiday_list)

    week_of_year = df["date"].dt.isocalendar().week.to_numpy(dtype=np.int64)

    X_year_all = fourier_design_matrix(
        week_of_year,
        period=T_PER_YEAR,
        num_modes=NUM_MODES_YEAR,
    )

    stan_data = create_stan_data(
        df["observed"].iloc[:split_idx],
        NUM_MODES_YEAR,
        X_year_all[:, :split_idx],
        X_year_all[:, split_idx:],
        d_peak_all[:, :split_idx],
        d_peak_all[:, split_idx:],
        hol_mask_all[:, :split_idx],
        hol_mask_all[:, split_idx:],
    )

    # CmdStanModel automatically caches the compiled binary if the .stan file hasn't changed
    holiday_model = CmdStanModel(
        stan_file=str(stan_path),
        stanc_options={"O1": True},
    )


    Path(output_dir).mkdir(parents=True, exist_ok=True)

    holiday_fit = holiday_model.sample(
        inits=0.5,
        chains=num_chains,
        parallel_chains=num_chains,
        iter_warmup=1000,
        iter_sampling=1000,
        data=stan_data,
        max_treedepth=max_treedepth,
        adapt_delta=adapt_delta,
        seed=seed,
        show_progress=True,
        output_dir=str(output_dir),
    )

    return df, holiday_fit
