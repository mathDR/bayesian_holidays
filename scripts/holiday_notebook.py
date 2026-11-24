import cmdstanpy

cmdstanpy.install_cmdstan()

from bayesian_holidays.fit_holiday_model import fit_holiday_model
from bayesian_holidays.plot_utils import (
    get_individual_holidays,
    plot_decomposition,
    plot_individual_holidays,
)

search_term = "ramadan"
assert search_term in ["chocolate", "ramadan"], (
    f"Search term {search_term} not supported. Only ramadan and chocolate are supported."
)

if search_term == "ramadan":
    country = "Bangladesh"
else:
    country = "UnitedStates"

train_split_percentage = 80

df_holiday, fit_holiday = fit_holiday_model(
    search_term,
    start_date=None,
    train_split_percentage=train_split_percentage,
    num_chains=4,
    max_treedepth=13,
    adapt_delta=0.95,
)

print(fit_holiday.diagnose())

holiday_list, hols_train, hols_test, df_train, df_test = get_individual_holidays(
    df_holiday, fit_holiday, country=country, train_split=train_split_percentage
)

# Get unique holiday names ordered by HolidayId (1..num_holidays)
hol_names = (
    holiday_list.sort_values("HolidayId")
    .drop_duplicates("HolidayId")["HolidayName"]
    .tolist()
)

# 2. Multi-panel grid of individual holidays with 50% & 95% credible intervals
plot_individual_holidays(
    df_train["date"], hols_train, hol_names, ncols=3, exp_scale=True
)

# 3. Full 4-panel model decomposition across train + out-of-sample test
plot_decomposition(
    df_holiday, fit_holiday, name="US Chocolate Search Volume Decomposition"
)
