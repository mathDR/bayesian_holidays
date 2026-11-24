"""Run end-to-end Bayesian holiday model fitting, diagnostics, and plotting."""

import argparse
import cmdstanpy
from cmdstanpy.utils import cmdstan_path

from bayesian_holidays.fit_holiday_model import DATASET_CONFIG, fit_holiday_model
from bayesian_holidays.plot_utils import (
    get_individual_holidays,
    plot_decomposition,
    plot_individual_holidays,
)

PLOT_TITLES: dict[str, str] = {
    "chocolate": "US Chocolate Search Volume Decomposition",
    "ramadan": "Bangladesh Ramadan Search Volume Decomposition",
}


def ensure_cmdstan() -> None:
    """Install CmdStan only if no existing installation is detected."""
    try:
        cmdstan_path()
    except ValueError:
        cmdstanpy.install_cmdstan()


def main(
    search_term: str = "ramadan",
    train_split_percentage: int = 80,
    num_chains: int = 4,
    max_treedepth: int = 13,
    adapt_delta: float = 0.95,
) -> None:
    if search_term not in DATASET_CONFIG:
        raise ValueError(
            f"Search term '{search_term}' not supported. "
            f"Supported terms: {list(DATASET_CONFIG)}"
        )

    ensure_cmdstan()

    _, _, country = DATASET_CONFIG[search_term]

    df_holiday, fit_holiday = fit_holiday_model(
        search_term=search_term,
        start_date=None,
        train_split_percentage=train_split_percentage,
        num_chains=num_chains,
        max_treedepth=max_treedepth,
        adapt_delta=adapt_delta,
    )

    print(fit_holiday.diagnose())

    holiday_list, hols_train, _hols_test, df_train, _df_test = get_individual_holidays(
        df_holiday,
        fit_holiday,
        country=country,
        train_split=train_split_percentage,
    )

    # Extract unique holiday names ordered by HolidayId (1..num_holidays)
    hol_names = (
        holiday_list.sort_values("HolidayId")
        .drop_duplicates("HolidayId")["HolidayName"]
        .tolist()
    )

    # 1. Multi-panel grid of individual holidays with 50% & 95% credible intervals
    plot_individual_holidays(
        df_train["date"],
        hols_train,
        hol_names,
        ncols=3,
        exp_scale=True,
    )

    # 2. Full 4-panel model decomposition across train + out-of-sample test
    plot_decomposition(
        df_holiday,
        fit_holiday,
        name=PLOT_TITLES[search_term],
        train_split_percentage=train_split_percentage,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit and plot the Bayesian Holiday Model."
    )
    parser.add_argument(
        "--search-term",
        type=str,
        default="ramadan",
        choices=list(DATASET_CONFIG),
        help="Dataset search term to fit (default: ramadan).",
    )
    parser.add_argument(
        "--train-split",
        type=int,
        default=80,
        help="Percentage of timeline used for training (default: 80).",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=4,
        help="Number of MCMC chains (default: 4).",
    )
    parser.add_argument(
        "--max-treedepth",
        type=int,
        default=13,
        help="NUTS max tree depth (default: 13).",
    )
    parser.add_argument(
        "--adapt-delta",
        type=float,
        default=0.95,
        help="NUTS target acceptance rate (default: 0.95).",
    )
    args = parser.parse_args()

    main(
        search_term=args.search_term,
        train_split_percentage=args.train_split,
        num_chains=args.chains,
        max_treedepth=args.max_treedepth,
        adapt_delta=args.adapt_delta,
    )
