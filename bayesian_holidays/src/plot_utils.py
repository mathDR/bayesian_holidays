from utils import create_d_peak, create_holiday_mask, get_holiday_dataframe


def plot_posteriors(df, df_fit, filename, plot_train=True, plot_test=True):
    alpha = df_fit.stan_variable("alpha")
    seasonality = df_fit.stan_variable("seasonality")
    holiday_effect = df_fit.stan_variable("holiday_effect")
    log_mu = alpha.values + seasonality.values + holiday_effect.values
    # log_mu = df_fit.stan_variable('obs_mean');
    test_seasonality = df_fit.stan_variable("test_seasonality")
    test_holiday_effect = df_fit.stan_variable("test_holiday_effect")
    test_log_mu = alpha.values + test_seasonality.values + test_holiday_effect.values
    # test_log_mu = df_fit.stan_variable('test_obsmean')

    train_date = df.date.iloc[int(0.8 * df.shape[0])]

    df_train = df[df.date <= train_date]
    df_test = df[df.date > train_date]

    fig, ax = plt.subplots(figsize=(18, 12))
    if plot_train:
        p = ax.plot(df_train.date, np.exp(log_mu[0, :]), alpha=0.01)
        clr = p[0].get_color()
        for i in range(1, alpha.shape[0]):
            ax.plot(df_train.date, np.exp(log_mu[i, :]), color=clr, alpha=0.01)
        ax.plot(
            df_train.date,
            np.exp(np.mean(log_mu, axis=0)),
            color="cyan",
            label="Mean of Posterior",
        )
    if plot_test:
        p = ax.plot(df_test.date, np.exp(test_log_mu[0, :]), alpha=0.01)
        clr = p[0].get_color()
        for i in range(1, alpha.shape[0]):
            ax.plot(df_test.date, np.exp(test_log_mu[i, :]), color=clr, alpha=0.01)
        ax.plot(
            df_test.date,
            np.exp(np.mean(test_log_mu, axis=0)),
            color="firebrick",
            label="OOS Posterior Mean",
        )
    if plot_train and plot_test:
        ax.plot(df.date, df.observed, label="Observed", lw=2, color="black")
        ax.set_xlim(pd.to_datetime(start_date), df.date.max())
    elif plot_train:
        ax.plot(df_train.date, df_train.observed, label="Observed", lw=2, color="black")
        ax.set_xlim(pd.to_datetime(start_date), df_train.date.max())
    elif plot_test:
        ax.plot(df_test.date, df_test.observed, label="Observed", lw=2, color="black")
    ax.set_xlabel("Date")
    ax.set_ylabel("Observed")
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left');
    ax.legend(loc="upper left")
    first_date = "2016-01-01"
    for i in range(0, 5, 2):
        plt.axvspan(
            pd.to_datetime(first_date) + i * pd.offsets.DateOffset(years=1),
            pd.to_datetime(first_date) + (i + 1) * pd.offsets.DateOffset(years=1),
            facecolor="gray",
            alpha=0.15,
        )
        # plt.axvline(pd.to_datetime(first_date) + i*pd.offsets.DateOffset(years=1),
        #            color='red', lw=2)
    plt.axvspan(
        df_train.date.min(), df_train.date.max(), facecolor="powderblue", alpha=0.15
    )
    plt.axvspan(df_test.date.min(), df_test.date.max(), facecolor="orange", alpha=0.15)
    # plt.axvline(df_test.date.min(),
    #            color='orange', lw=3)
    for tt in ax.get_xticklabels()[-2:]:
        tt.set_color("orange")
    name = filename.split("/")[-1].split(".csv")[0]
    if "fireworks" in name:
        name = name.split("_")[0]
    plt.title(name)
    plt.ylim(0.8 * df.observed.min(), 1.1 * df.observed.max())
    plt.xlim(pd.to_datetime(first_date), df.date.max())
    l = [
        (pd.to_datetime(first_date) + pd.offsets.DateOffset(months=5)).date(),
        (pd.to_datetime("2017-01-01") + pd.offsets.DateOffset(months=5)).date(),
        (pd.to_datetime("2018-01-01") + pd.offsets.DateOffset(months=5)).date(),
        (pd.to_datetime("2019-01-01") + pd.offsets.DateOffset(months=5)).date(),
        (pd.to_datetime("2020-01-01") + pd.offsets.DateOffset(months=5)).date(),
    ]
    ax.set_xticks(l)
    ax.set_xticklabels([k.year for k in l])
    plt.show()
    return None


def plot_components(
    df, df_fit, filename, start_date="2016-01-01", plot_train=True, plot_test=True
):
    baseline = df_fit.stan_variable("baseline").values
    seasonality = df_fit.stan_variable("seasonality").values
    holiday_effect = df_fit.stan_variable("holiday_effect").values

    test_baseline = df_fit.stan_variable("test_baseline").values
    test_seasonality = df_fit.stan_variable("test_seasonality").values
    test_holiday_effect = df_fit.stan_variable("test_holiday_effect").values

    train_date = df.date.iloc[int(0.8 * df.shape[0])]

    df_train = df[df.date <= train_date]
    df_test = df[df.date > train_date]

    fig, ax = plt.subplots(figsize=(18, 12))
    if plot_train:
        #         p = ax.plot(df_train.date, np.exp(baseline[0,:]), alpha=0.05)
        #         clr = p[0].get_color()
        #         for i in range(1,baseline.shape[0]):
        #             ax.plot(df_train.date, np.exp(baseline[i,:]), color=clr, alpha=0.05)
        #         ax.plot(df_train.date, np.exp(np.mean(baseline,axis=0)),label='Mean of Posterior Baseline');
        p = ax.plot(df_train.date, np.exp(seasonality[0, :]), alpha=0.05)
        clr = p[0].get_color()
        for i in range(1, seasonality.shape[0]):
            ax.plot(df_train.date, np.exp(seasonality[i, :]), color=clr, alpha=0.05)
        ax.plot(
            df_train.date,
            np.exp(np.mean(seasonality, axis=0)),
            label="Mean of Posterior Seasonality",
        )
        p = ax.plot(df_train.date, np.exp(holiday_effect[0, :]), alpha=0.05)
        clr = p[0].get_color()
        for i in range(1, holiday_effect.shape[0]):
            ax.plot(df_train.date, np.exp(holiday_effect[i, :]), color=clr, alpha=0.05)
        ax.plot(
            df_train.date,
            np.exp(np.mean(holiday_effect, axis=0)),
            label="Mean of Posterior Holiday Effect",
        )
    if plot_test:
        #         p = ax.plot(df_test.date, np.exp(test_baseline[0,:]), alpha=0.05)
        #         clr = p[0].get_color()
        #         for i in range(1,test_baseline.shape[0]):
        #             ax.plot(df_test.date, np.exp(test_baseline[i,:]), color=clr, alpha=0.05)
        #         ax.plot(df_test.date, np.exp(np.mean(test_baseline,axis=0)),label='Mean of Posterior Baseline OOS');
        p = ax.plot(df_test.date, np.exp(test_seasonality[0, :]), alpha=0.05)
        clr = p[0].get_color()
        for i in range(1, test_seasonality.shape[0]):
            ax.plot(df_test.date, np.exp(test_seasonality[i, :]), color=clr, alpha=0.05)
        ax.plot(
            df_test.date,
            np.exp(np.mean(test_seasonality, axis=0)),
            label="Mean of Posterior Seasonality OOS",
        )
        p = ax.plot(df_test.date, np.exp(test_holiday_effect[0, :]), alpha=0.05)
        clr = p[0].get_color()
        for i in range(1, test_holiday_effect.shape[0]):
            ax.plot(
                df_test.date, np.exp(test_holiday_effect[i, :]), color=clr, alpha=0.05
            )
        ax.plot(
            df_test.date,
            np.exp(np.mean(test_holiday_effect, axis=0)),
            label="Mean of Posterior Holiday Effect OOS",
        )
    ax1 = ax.twinx()
    if plot_train and plot_test:
        ax1.plot(df.date, df.observed, label="Observed", lw=2, color="black")
        ax1.set_xlim(pd.to_datetime(start_date), df.date.max())
    elif plot_train:
        ax1.plot(
            df_train.date, df_train.observed, label="Observed", lw=2, color="black"
        )
        ax1.set_xlim(pd.to_datetime(start_date), df_train.date.max())
    elif plot_test:
        ax1.plot(df_test.date, df_test.observed, label="Observed", lw=2, color="black")
    ax.set_xlabel("Date")
    ax.set_ylabel("Observed")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    for i in range(0, 5, 2):
        plt.axvspan(
            pd.to_datetime(start_date) + i * pd.offsets.DateOffset(years=1),
            pd.to_datetime(start_date) + (i + 1) * pd.offsets.DateOffset(years=1),
            facecolor="gray",
            alpha=0.25,
        )
    plt.title(filename.split("/")[-1].split(".csv")[0])
    plt.show()
    return None


def inv_logit(u):
    return 1.0 / (1.0 + np.exp(-u))


def get_holiday_lift(h_skew, h_shape, h_scale, h_loc, intensity, d_peak, lb, ub):
    num_holidays, num_dates = d_peak.shape
    tdd = np.zeros((h_loc.shape[0], h_loc.shape[1], d_peak[0, :].shape[0]))

    for t in range(num_dates):
        for h in range(num_holidays):
            if (d_peak[h, t] > lb[h]) and (d_peak[h, t] < ub[h]):
                z = (d_peak[h, t] - h_loc[:, h]) / h_scale[:, h]
                tdd[:, h, t] += (
                    2.0
                    * intensity[:, h]
                    * np.exp(-np.abs(z) ** h_shape[:, h])
                    * inv_logit(h_skew[:, h] * z)
                    * np.exp(-1.0 / ((ub[h] - d_peak[h, t]) * (d_peak[h, t] - lb[h])))
                )
    return tdd


def get_individual_holidays(df, df_fit, train_split=80, return_all=False):
    start_date = df["date"].min()
    end_date = df["date"].max()

    holiday_years = list(
        range(pd.to_datetime(start_date).year - 1, pd.to_datetime(end_date).year + 1)
    )
    holiday_list = (
        get_holiday_dataframe(years=holiday_years)
        .sort_values(by="HolidayDate")
        .reset_index()
    )

    train_date = df.date.iloc[int((train_split / 100) * df.date.shape[0])]

    df_train = df[df.date <= train_date]
    df_test = df[df.date > train_date]

    d_peak = create_d_peak(df_train.date, holiday_list)
    d_peak_test = create_d_peak(df_test.date, holiday_list)

    h_skew = df_fit.stan_variable("h_skew").values
    h_shape = df_fit.stan_variable("h_shape").values
    h_scale = df_fit.stan_variable("h_scale").values
    h_loc = df_fit.stan_variable("h_loc").values
    intensity = df_fit.stan_variable("intensity").values

    df_bounds = (
        holiday_list.groupby(by="HolidayName")
        .agg({"days_behind_diff": min, "days_ahead_diff": min})
        .reindex(
            holiday_list.head(holiday_list.HolidayName.unique().shape[0]).HolidayName
        )
        .rename(
            columns={
                "days_behind_diff": "lower_bounds",
                "days_ahead_diff": "upper_bounds",
            }
        )
    )
    lb = -df_bounds.lower_bounds.dt.days.values
    ub = df_bounds.upper_bounds.dt.days.values

    hols_train = get_holiday_lift(
        h_skew, h_shape, h_scale, h_loc, intensity, d_peak, lb, ub
    )
    hols_test = get_holiday_lift(
        h_skew, h_shape, h_scale, h_loc, intensity, d_peak_test, lb, ub
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
        plt.plot(times, np.mean(tdd[:, h, :], axis=0), color="firebrick", lw=2)
        plt.title(hol_names[h])
        plt.show()
