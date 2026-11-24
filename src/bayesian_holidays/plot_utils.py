import math
from collections.abc import Sequence
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanMCMC
from scipy.special import expit

from bayesian_holidays.utils import (
    create_d_peak,
    create_mask_logistic,
    get_holiday_dataframe,
)


def _subsample_draws(
    arr: np.ndarray, max_draws: int = 300, seed: int = 42
) -> np.ndarray:
    """Subsample rows (posterior draws) for fast Matplotlib rendering."""
    if arr.shape[0] <= max_draws:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(arr.shape[0], size=max_draws, replace=False)
    return arr[idx]


def _get_train_test_split(
    df: pd.DataFrame, train_split_percentage: int = 80
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice dataframe identically to fit_holiday_model."""
    split_idx = int((train_split_percentage / 100.0) * len(df))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def plot_posteriors(
    df: pd.DataFrame,
    df_fit: CmdStanMCMC,
    name: str | None = None,
    train_split_percentage: int = 80,
    plot_train: bool = True,
    plot_test: bool = True,
    max_draws: int = 300,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot posterior expected trajectories (train & out-of-sample test) against observed data.
    """
    df_train, df_test = _get_train_test_split(df, train_split_percentage)

    fig, ax = plt.subplots(figsize=(18, 10))

    if plot_train:
        mu_train = np.exp(df_fit.stan_variable("log_obs_mean"))
        mu_train_sub = _subsample_draws(mu_train, max_draws=max_draws)
        ax.plot(df_train["date"], mu_train_sub.T, color="steelblue", alpha=0.04, lw=0.8)
        ax.plot(
            df_train["date"],
            mu_train.mean(axis=0),
            color="cyan",
            lw=2.2,
            label="Train Posterior Mean",
        )

    if plot_test and len(df_test) > 0:
        mu_test = np.exp(df_fit.stan_variable("test_log_obsmean"))
        mu_test_sub = _subsample_draws(mu_test, max_draws=max_draws)
        ax.plot(df_test["date"], mu_test_sub.T, color="salmon", alpha=0.04, lw=0.8)
        ax.plot(
            df_test["date"],
            mu_test.mean(axis=0),
            color="firebrick",
            lw=2.2,
            label="OOS Posterior Mean",
        )

    # Plot observed series across the requested window
    if plot_train and plot_test:
        plot_df = df
    elif plot_train:
        plot_df = df_train
    elif plot_test:
        plot_df = df_test
    else:
        plot_df = df.iloc[0:0]

    if not plot_df.empty:
        ax.plot(
            plot_df["date"], plot_df["observed"], label="Observed", lw=2, color="black"
        )
        ax.set_xlim(plot_df["date"].min(), plot_df["date"].max())

    if plot_test and len(df_test) > 0:
        test_start = df_test["date"].min()
        ax.axvspan(test_start, df_test["date"].max(), facecolor="orange", alpha=0.15)
        ax.axvline(
            test_start, color="orange", lw=2.5, linestyle="--", label="Train/Test Split"
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Observed")
    ax.set_ylim(0.8 * df["observed"].min(), 1.1 * df["observed"].max())
    ax.legend(loc="upper left")

    if name is not None:
        ax.set_title(name)

    fig.tight_layout()
    plt.show()
    return fig, ax


def plot_components(
    df: pd.DataFrame,
    df_fit: CmdStanMCMC,
    name: str | None = None,
    start_date: str | None = None,
    train_split_percentage: int = 80,
    plot_train: bool = True,
    plot_test: bool = True,
    max_draws: int = 300,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """
    Plot multiplicative seasonality and holiday lift components alongside observed counts.
    """
    df_train, df_test = _get_train_test_split(df, train_split_percentage)

    fig, ax = plt.subplots(figsize=(18, 10))

    if plot_train:
        seas_train = np.exp(df_fit.stan_variable("log_seasonality"))
        hol_train = np.exp(df_fit.stan_variable("holiday_effect"))

        ax.plot(
            df_train["date"],
            _subsample_draws(seas_train, max_draws).T,
            color="tab:blue",
            alpha=0.04,
            lw=0.8,
        )
        ax.plot(
            df_train["date"],
            seas_train.mean(axis=0),
            color="tab:blue",
            lw=2,
            label="Seasonality (Train Mean)",
        )

        ax.plot(
            df_train["date"],
            _subsample_draws(hol_train, max_draws).T,
            color="tab:orange",
            alpha=0.04,
            lw=0.8,
        )
        ax.plot(
            df_train["date"],
            hol_train.mean(axis=0),
            color="tab:orange",
            lw=2,
            label="Holiday Lift (Train Mean)",
        )

    if plot_test and len(df_test) > 0:
        seas_test = np.exp(df_fit.stan_variable("test_log_seasonality"))
        hol_test = np.exp(df_fit.stan_variable("test_holiday_effect"))

        ax.plot(
            df_test["date"],
            _subsample_draws(seas_test, max_draws).T,
            color="tab:green",
            alpha=0.04,
            lw=0.8,
        )
        ax.plot(
            df_test["date"],
            seas_test.mean(axis=0),
            color="tab:green",
            lw=2,
            label="Seasonality (OOS Mean)",
        )

        ax.plot(
            df_test["date"],
            _subsample_draws(hol_test, max_draws).T,
            color="tab:red",
            alpha=0.04,
            lw=0.8,
        )
        ax.plot(
            df_test["date"],
            hol_test.mean(axis=0),
            color="tab:red",
            lw=2,
            label="Holiday Lift (OOS Mean)",
        )

    ax1 = ax.twinx()
    if plot_train and plot_test:
        plot_df = df
    elif plot_train:
        plot_df = df_train
    elif plot_test:
        plot_df = df_test
    else:
        plot_df = df.iloc[0:0]

    xmin = (
        pd.to_datetime(start_date) if start_date is not None else plot_df["date"].min()
    )
    xmax = plot_df["date"].max()

    if not plot_df.empty:
        ax1.plot(
            plot_df["date"],
            plot_df["observed"],
            label="Observed",
            lw=1.8,
            color="black",
            alpha=0.85,
        )
        xmin_num = float(mdates.date2num(xmin))
        xmax_num = float(mdates.date2num(xmax))
        ax.set_xlim(xmin_num, xmax_num)
        ax1.set_xlim(xmin_num, xmax_num)

    # Dynamically shade alternating calendar years across the full date range
    year_boundaries = pd.date_range(
        start=xmin.to_period("Y").start_time, end=xmax, freq="YS"
    )
    for idx in range(0, len(year_boundaries), 2):
        y_start = float(mdates.date2num(year_boundaries[idx]))
        y_end = float(mdates.date2num(year_boundaries[idx] + pd.DateOffset(years=1)))
        ax.axvspan(y_start, y_end, facecolor="gray", alpha=0.15)

    ax.set_xlabel("Date")
    ax.set_ylabel("Multiplicative Effect (exp scale)")
    ax1.set_ylabel("Observed")

    # Combine legends from both axes cleanly
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax1.get_legend_handles_labels()
    ax.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    if name is not None:
        ax.set_title(name)

    fig.tight_layout()
    plt.show()
    return fig, (ax, ax1)


def get_holiday_lift(
    h_skew: np.ndarray,
    h_shape: np.ndarray,
    h_scale: np.ndarray,
    h_loc: np.ndarray,
    intensity: np.ndarray,
    d_peak: np.ndarray,
    hol_mask: np.ndarray,
    return_sum: bool = True,
) -> np.ndarray:
    """
    Compute posterior holiday lift across MCMC draws.
    Evaluates only active (non-zero) mask entries to save memory and CPU time.
    """
    num_draws, num_holidays = intensity.shape
    num_dates = d_peak.shape[1]

    tdd = np.zeros((num_draws, num_holidays, num_dates), dtype=np.float64)

    for h in range(num_holidays):
        active = hol_mask[h] != 0.0
        if not np.any(active):
            continue

        z = (d_peak[h, active][None, :] - h_loc[:, h, None]) / h_scale[:, h, None]
        tdd[:, h, active] = (
            (2.0 * intensity[:, h, None])
            * np.exp(-((np.square(z) + 1e-12) ** h_shape[:, h, None]))
            * expit(h_skew[:, h, None] * z)
            * hol_mask[h, active][None, :]
        )

    return tdd.sum(axis=1) if return_sum else tdd


def get_individual_holidays(
    df: pd.DataFrame,
    df_fit: CmdStanMCMC,
    country: str,
    train_split: int = 80,
    return_all: bool = False,
) -> tuple[Any, ...]:
    start_date = df["date"].min()
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

    h_skew = df_fit.stan_variable("h_skew")
    h_shape = df_fit.stan_variable("h_shape")
    h_scale = df_fit.stan_variable("h_scale")
    h_loc = df_fit.stan_variable("h_loc")
    intensity = df_fit.stan_variable("intensity")

    hols_train = get_holiday_lift(
        h_skew, h_shape, h_scale, h_loc, intensity, d_peak, hol_mask, return_sum=False
    )
    hols_test = get_holiday_lift(
        h_skew,
        h_shape,
        h_scale,
        h_loc,
        intensity,
        d_peak_test,
        hol_mask_test,
        return_sum=False,
    )

    if return_all:
        return (
            holiday_list,
            h_skew,
            h_shape,
            h_scale,
            h_loc,
            intensity,
            hols_train,
            hols_test,
            df_train,
            df_test,
        )
    else:
        return holiday_list, hols_train, hols_test, df_train, df_test


def plot_individual_holidays(
    times: pd.Series | pd.DatetimeIndex | np.ndarray,
    tdd: np.ndarray,
    hol_names: Sequence[str] | np.ndarray,
    ncols: int = 3,
    exp_scale: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot each holiday's posterior trajectory in a multi-panel subplot grid
    with 95% and 50% credible interval ribbons and posterior mean.

    Parameters
    ----------
    times : array-like of datetime
        Dates corresponding to the last axis of `tdd`.
    tdd : np.ndarray of shape (num_draws, num_holidays, num_dates)
        Posterior holiday effect draws from `get_individual_holidays`.
    hol_names : sequence of str
        Names of the holidays ordered by HolidayId (1..num_holidays).
    ncols : int, default 3
        Number of columns in the subplot grid.
    exp_scale : bool, default False
        If True, plots multiplicative lift `exp(tdd)` instead of additive log-lift.
    """
    data = np.exp(tdd) if exp_scale else tdd
    _, num_holidays, _ = data.shape
    nrows = math.ceil(num_holidays / ncols)

    # Vectorized quantile computation across all draws simultaneously
    q025, q25, q75, q975 = np.percentile(data, [2.5, 25.0, 75.0, 97.5], axis=0)
    post_mean = data.mean(axis=0)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.5 * ncols, 3.0 * nrows),
        sharex=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()

    baseline_ref = 1.0 if exp_scale else 0.0
    ylabel = "Multiplicative Lift" if exp_scale else "Log Lift"

    for h in range(num_holidays):
        ax = axes_flat[h]
        ax.axhline(baseline_ref, color="gray", linestyle=":", linewidth=1.0, alpha=0.8)
        ax.fill_between(
            times, q025[h], q975[h], color="orange", alpha=0.22, label="95% CI"
        )
        ax.fill_between(
            times, q25[h], q75[h], color="darkorange", alpha=0.38, label="50% CI"
        )
        ax.plot(times, post_mean[h], color="firebrick", linewidth=1.8, label="Mean")

        ax.set_title(str(hol_names[h]), fontsize=11, fontweight="bold")
        if h % ncols == 0:
            ax.set_ylabel(ylabel)

    # Hide unused trailing subplot cells
    for idx in range(num_holidays, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Format x-axis dates cleanly on visible bottom plots
    fig.autofmt_xdate(rotation=30)
    # Add a single shared legend at the top of the figure
    handles, labels = axes_flat[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()
    return fig, axes


def plot_decomposition(
    df: pd.DataFrame,
    df_fit: CmdStanMCMC,
    name: str | None = None,
    train_split_percentage: int = 80,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot a 4-panel time-series decomposition across train and test windows:
      1. Observed vs. Full Posterior Expected Mean & 95% Predictive Interval
      2. Baseline Trend (exp(log_baseline_real))
      3. Yearly Seasonality Multiplier (exp(log_seasonality))
      4. Combined Holiday Lift Multiplier (exp(holiday_effect))
    """
    _df_train, df_test = _get_train_test_split(df, train_split_percentage)
    has_test = len(df_test) > 0
    all_dates = df["date"]

    # Extract and concatenate train + test posterior draws along time axis
    baseline_draws = df_fit.stan_variable("log_baseline_real")[:, None]
    baseline_full = np.repeat(np.exp(baseline_draws), len(df), axis=1)

    seas_train = df_fit.stan_variable("log_seasonality")
    hol_train = df_fit.stan_variable("holiday_effect")
    mu_train = df_fit.stan_variable("log_obs_mean")

    if has_test:
        seas_full = np.exp(
            np.concatenate(
                [seas_train, df_fit.stan_variable("test_log_seasonality")], axis=1
            )
        )
        hol_full = np.exp(
            np.concatenate(
                [hol_train, df_fit.stan_variable("test_holiday_effect")], axis=1
            )
        )
        mu_full = np.exp(
            np.concatenate([mu_train, df_fit.stan_variable("test_log_obsmean")], axis=1)
        )
    else:
        seas_full = np.exp(seas_train)
        hol_full = np.exp(hol_train)
        mu_full = np.exp(mu_train)

    panels = [
        (
            "Observed vs. Expected Mean (95% CI)",
            mu_full,
            "steelblue",
            "Count",
            True,
        ),
        (
            "Baseline Level (exp(α))",
            baseline_full,
            "slategray",
            "Baseline Rate",
            False,
        ),
        (
            "Yearly Seasonality Multiplier",
            seas_full,
            "teal",
            "Multiplier",
            False,
        ),
        (
            "Combined Holiday Lift Multiplier",
            hol_full,
            "darkorange",
            "Multiplier",
            False,
        ),
    ]

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(16, 13),
        sharex=True,
        gridspec_kw={"height_ratios": [1.5, 0.8, 1.0, 1.0]},
    )

    test_start = df_test["date"].min() if has_test else None
    test_end = df_test["date"].max() if has_test else None

    for ax, (panel_title, draws, color, ylabel, show_obs) in zip(
        axes, panels, strict=False
    ):
        q025, q25, q75, q975 = np.percentile(draws, [2.5, 25.0, 75.0, 97.5], axis=0)
        mean_curve = draws.mean(axis=0)

        ax.fill_between(all_dates, q025, q975, color=color, alpha=0.20, label="95% CI")
        ax.fill_between(all_dates, q25, q75, color=color, alpha=0.35, label="50% CI")
        ax.plot(
            all_dates, mean_curve, color=color, linewidth=2.0, label="Posterior Mean"
        )

        if show_obs:
            ax.plot(
                all_dates,
                df["observed"],
                color="black",
                linewidth=1.5,
                alpha=0.85,
                label="Observed",
            )
        else:
            ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)

        if has_test:
            ax.axvspan(test_start, test_end, facecolor="orange", alpha=0.10)
            ax.axvline(
                test_start,
                color="orange",
                linestyle="--",
                linewidth=1.8,
                label="Train/Test Split" if show_obs else None,
            )

        ax.set_title(panel_title, loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper left", ncol=4, frameon=True, framealpha=0.85, fontsize=9)

    axes[-1].set_xlabel("Date")
    axes[-1].set_xlim(all_dates.min(), all_dates.max())
    fig.autofmt_xdate(rotation=30)

    if name is not None:
        fig.suptitle(name, fontsize=14, fontweight="bold", y=1.005)

    fig.tight_layout()
    plt.show()
    return fig, axes
