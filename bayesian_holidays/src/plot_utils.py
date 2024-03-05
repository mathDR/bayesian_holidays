import matplotlib.pyplot as plt
from pandas import offsets, to_datetime
from numpy import exp, expand_dims, mean, square, sum
from scipy.special import expit
from ..src.utils import create_d_peak, create_mask_logistic, get_holiday_dataframe


def plot_posteriors(df, df_fit, name=None, plot_train=True, plot_test=True):
    alpha = df_fit.stan_variable("log_baseline_real")
    seasonality = df_fit.stan_variable("log_seasonality")
    holiday_effect = df_fit.stan_variable("holiday_effect")
    log_mu = alpha.reshape(-1, 1) + seasonality + holiday_effect

    test_seasonality = df_fit.stan_variable("test_log_seasonality")
    test_holiday_effect = df_fit.stan_variable("test_holiday_effect")
    test_log_mu = alpha.reshape(-1, 1) + test_seasonality + test_holiday_effect

    train_date = df.date.iloc[int(0.8 * df.shape[0])]

    df_train = df[df.date <= train_date]
    df_test = df[df.date > train_date]

    start_date = df.date.min()

    fig, ax = plt.subplots(figsize=(18, 12))
    if plot_train:
        p = ax.plot(df_train.date, exp(log_mu[0, :]), alpha=0.01)
        clr = p[0].get_color()
        for i in range(1, alpha.shape[0]):
            ax.plot(df_train.date, exp(log_mu[i, :]), color=clr, alpha=0.01)
        ax.plot(
            df_train.date,
            exp(mean(log_mu, axis=0)),
            color="cyan",
            label="Mean of Posterior",
        )
    if plot_test:
        p = ax.plot(df_test.date, exp(test_log_mu[0, :]), alpha=0.01)
        clr = p[0].get_color()
        for i in range(1, alpha.shape[0]):
            ax.plot(df_test.date, exp(test_log_mu[i, :]), color=clr, alpha=0.01)
        ax.plot(
            df_test.date,
            exp(mean(test_log_mu, axis=0)),
            color="firebrick",
            label="OOS Posterior Mean",
        )
    if plot_train and plot_test:
        ax.plot(df.date, df.observed, label="Observed", lw=2, color="black")
        ax.set_xlim(to_datetime(start_date), df.date.max())
    elif plot_train:
        ax.plot(df_train.date, df_train.observed, label="Observed", lw=2, color="black")
        ax.set_xlim(to_datetime(start_date), df_train.date.max())
    elif plot_test:
        ax.plot(df_test.date, df_test.observed, label="Observed", lw=2, color="black")

    ax.set_xlabel("Date")
    ax.set_ylabel("Observed")

    ax.legend(loc="upper left")

    plt.axvspan(df_test.date.min(), df_test.date.max(), facecolor="orange", alpha=0.15)
    plt.axvline(df_test.date.min(), color="orange", lw=3)
    if name is not None:
        plt.title(name)
    plt.ylim(0.8 * df.observed.min(), 1.1 * df.observed.max())
    plt.xlim(to_datetime(df.date.min()), df.date.max())

    plt.show()
    return None


def plot_components(
    df, df_fit, name=None, start_date="2016-01-01", plot_train=True, plot_test=True
):
    log_baseline = df_fit.stan_variable("log_baseline")
    log_seasonality = df_fit.stan_variable("log_seasonality")
    holiday_effect = df_fit.stan_variable("holiday_effect")

    test_log_baseline = df_fit.stan_variable("test_log_baseline")
    test_log_seasonality = df_fit.stan_variable("test_log_seasonality")
    test_holiday_effect = df_fit.stan_variable("test_holiday_effect")

    train_date = df.date.iloc[int(0.8 * df.shape[0])]

    df_train = df[df.date <= train_date]
    df_test = df[df.date > train_date]

    fig, ax = plt.subplots(figsize=(18, 12))
    if plot_train:
        p = ax.plot(df_train.date, exp(log_seasonality[0, :]), alpha=0.05)
        clr = p[0].get_color()
        for i in range(1, seasonality.shape[0]):
            ax.plot(df_train.date, exp(log_seasonality[i, :]), color=clr, alpha=0.05)
        ax.plot(
            df_train.date,
            exp(mean(seasonality, axis=0)),
            label="Mean of Posterior Seasonality",
        )
        p = ax.plot(df_train.date, exp(holiday_effect[0, :]), alpha=0.05)
        clr = p[0].get_color()
        for i in range(1, holiday_effect.shape[0]):
            ax.plot(df_train.date, exp(holiday_effect[i, :]), color=clr, alpha=0.05)
        ax.plot(
            df_train.date,
            np.exp(np.mean(holiday_effect, axis=0)),
            label="Mean of Posterior Holiday Effect",
        )
    if plot_test:
        p = ax.plot(df_test.date, exp(test_log_seasonality[0, :]), alpha=0.05)
        clr = p[0].get_color()
        for i in range(1, test_seasonality.shape[0]):
            ax.plot(
                df_test.date, exp(test_log_seasonality[i, :]), color=clr, alpha=0.05
            )
        ax.plot(
            df_test.date,
            exp(mean(test_seasonality, axis=0)),
            label="Mean of Posterior Seasonality OOS",
        )
        p = ax.plot(df_test.date, exp(test_holiday_effect[0, :]), alpha=0.05)
        clr = p[0].get_color()
        for i in range(1, test_holiday_effect.shape[0]):
            ax.plot(df_test.date, exp(test_holiday_effect[i, :]), color=clr, alpha=0.05)
        ax.plot(
            df_test.date,
            exp(mean(test_holiday_effect, axis=0)),
            label="Mean of Posterior Holiday Effect OOS",
        )
    ax1 = ax.twinx()
    if plot_train and plot_test:
        ax1.plot(df.date, df.observed, label="Observed", lw=2, color="black")
        ax1.set_xlim(to_datetime(start_date), df.date.max())
    elif plot_train:
        ax1.plot(
            df_train.date, df_train.observed, label="Observed", lw=2, color="black"
        )
        ax1.set_xlim(to_datetime(start_date), df_train.date.max())
    elif plot_test:
        ax1.plot(df_test.date, df_test.observed, label="Observed", lw=2, color="black")
    ax.set_xlabel("Date")
    ax.set_ylabel("Observed")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    for i in range(0, 5, 2):
        plt.axvspan(
            to_datetime(start_date) + i * offsets.DateOffset(years=1),
            to_datetime(start_date) + (i + 1) * offsets.DateOffset(years=1),
            facecolor="gray",
            alpha=0.25,
        )
    if name is not None:
        plt.title(name)
    plt.show()
    return None


def get_holiday_lift(
    h_skew, h_shape, h_scale, h_loc, intensity, d_peak, hol_mask, return_sum=True
):
    # z = (t - loc) / scale
    z = (expand_dims(d_peak, axis=0) - expand_dims(h_loc, axis=2)) / expand_dims(
        h_scale, axis=2
    )
    # tdd = intensity * exp(-square(z)** shape) * expit(z* skew) * mask
    tdd = (
        2.0
        * expand_dims(intensity, axis=2)
        * exp(-square(z) ** expand_dims(h_shape, axis=2))
        * expit(expand_dims(h_skew, axis=2) * z)
        * expand_dims(hol_mask, axis=0),
    )

    if return_sum:
        return sum(tdd, axis=1)
    else:
        return tdd[0]


def get_individual_holidays(df, df_fit, country=None, train_split=80, return_all=False):
    start_date = df["date"].min()
    end_date = df["date"].max()

    holiday_years = list(
        range(to_datetime(start_date).year - 1, to_datetime(end_date).year + 1)
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


def plot_individual_holidays(times, tdd, hol_names):
    for h in range(tdd.shape[1]):
        for j in range(tdd.shape[0]):
            plt.plot(times, tdd[j, h, :], color="orange", alpha=0.1)
        plt.plot(times, mean(tdd[:, h, :], axis=0), color="firebrick", lw=2)
        plt.title(hol_names[h])
        plt.show()
