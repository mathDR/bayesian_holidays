#include <stan/model/model_header.hpp>
#include <ostream>

namespace bernoulli_model_namespace {

  template <typename T0__, stan::require_stan_scalar_t<T0__>* = nullptr>
  stan::promote_args_t<T0__>
  make_odds_marthaler(const T0__& theta, std::ostream* pstream__) {
    return theta / (1 - theta);
  }
}