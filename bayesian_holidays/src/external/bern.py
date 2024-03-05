import cmdstanpy
import logging

cmdstanpy.install_cmdstan()
# cmdstanpy.rebuild_cmdstan(verbose=True)

from cmdstanpy import CmdStanModel

# cmdstanpy_logger = logging.getLogger("cmdstanpy")
# cmdstanpy_logger.disabled = False
# cmdstanpy_logger.handlers = []
# cmdstanpy_logger.setLevel(logging.DEBUG)
# handler = logging.FileHandler("all.log")
# handler.setLevel(logging.DEBUG)
# handler.setFormatter(
#    logging.Formatter(
#        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#        "%H:%M:%S",
#    )
# )

# cmdstanpy_logger.addHandler(handler)
cpp_options = {
    "CXXFLAGS": "-Wno-deprecated-declarations",
    "CXX": "arch -arch arm64e clang++",
}
stan_model = CmdStanModel(
    stan_file="/Users/daniel.marthaler/dev/bayesian_holidays/bayesian_holidays/src/external/bernoulli.stan",
    user_header="/Users/daniel.marthaler/dev/bayesian_holidays/bayesian_holidays/src/external/make_odds.hpp",
    # cpp_options=cpp_options,
    stanc_options={"allow-undefined": True},
)

fit = stan_model.sample(data={"N": 10, "y": [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]})
print(fit.stan_variable("odds"))
