functions {
  /*
    Our holiday effect function: get_holiday_lift describes the 
    effect of the holiday at date t as:
      h(t) = 2*lambda * exp(−|z(t)|^h_shape) / (1+exp(−h_skew * z(t))
    with
      z(t) = (t−h_loc) / h_scale
    where
    * h_loc is the location parameter - it denotes how “offset” the effect 
      is from the actual holiday date
    * h_scale is the scale parameter - it denotes how broad the effect 
      of the holiday is over time
    * h_shape is the shape parameter - it denotes how “peaky” the effect 
      is in time
    * h_skew is the skew parameter - it denotes how asymmetrical the 
      holiday effect is around h_loc
    * lambda is the intensity parameter - this denotes the magnitude of
      the holiday effect.
    The model is then "masked" so that the effect of any holiday can only persist
    within a time window between the previous holiday and the next holiday.  (So
    for example, Christmas cannot persist back before Thanksgiving, nor can 
    it persist beyond New Year's Day)
  */
  
  row_vector get_holiday_lift(
    vector h_skew, 
    vector h_shape,
    vector h_scale,
    vector h_loc,
    vector intensity,
    matrix d_peak,
    matrix hol_mask
    ) 
  {
    int num_holidays = dims(d_peak)[1];
    int num_dates = dims(d_peak)[2];

    row_vector[num_dates] z;
    row_vector[num_dates] tdd = zeros_row_vector(num_dates);
    
    for (h in 1:num_holidays) {
      z = (d_peak[h, :] - h_loc[h]) ./ h_scale[h];
      tdd += (2.0 * intensity[h] * exp(-abs(z) .^ h_shape[h]) .* 
        inv_logit(h_skew[h] * z)
        ) .* hol_mask[h,:];
    }

    return tdd;

  }

}

data {
  // OBSERVATIONS
  int<lower=1> num_dates; // number of dates
  int<lower=1> num_test_dates; // number of dates
  int<lower=0> num_holidays; // number of holidays
  int<lower=0> obs[num_dates];

  matrix[num_holidays, num_dates] d_peak; // distance (in time) from holiday
  matrix[num_holidays, num_test_examples] d_peak_test; // distance (in time) from holiday

  matrix[num_holidays, num_dates] hol_mask;
  matrix[num_holidays, num_test_examples] hol_mask_test;

  int<lower=0> num_modes_year;              // Number of fourier modes
  matrix[2*num_modes_year, num_dates] X_year;  // one each for cosine and sine
  matrix[2*num_modes_year, num_test_dates] X_year_test;

  vector[num_holidays] h_loc_prior_mu;
  vector<lower=0>[num_holidays] h_loc_prior_sig;

  vector<lower=0>[num_holidays] h_scale_prior_alpha;
  vector<lower=0>[num_holidays] h_scale_prior_beta;

  vector[num_holidays] h_shape_prior_mu;
  vector<lower=0>[num_holidays] h_shape_prior_sig;
  
  vector[num_holidays] h_skew_prior_mu;
  vector<lower=0>[num_holidays] h_skew_prior_sig;

}

transformed data {
  // HOLIDAY Prior parameters
  real expected_num_holidays = 3.0; // Expected number of activated holidays
  real slab_scale = 2.0;    // Scale for large slopes - s parameter in HS paper
  real slab_scale2 = square(slab_scale); // s^2 in HS paper
  real slab_df = 25.0;      // Effective degrees of freedom for large slopes - nu in HS paper
  real half_slab_df = 0.5 * slab_df; // nu/2 in regularized HS paper
  real tau0 = (expected_num_holidays / (num_holidays - expected_num_holidays)) * (1.0 / sqrt(1.0 * num_dates));

}

parameters {
  real alpha;

  row_vector[2*num_modes_year] fourier_coefficients;
  
  // Holiday Parameters  
  vector[num_holidays] lambda_tilde;
  real<lower=0> c2_tilde;
  real<lower = 0, upper = pi()/2> tau_tilde_unif;
  vector[num_holidays] h_locZ;
  vector<lower=0>[num_holidays] h_scale_raw;
  vector[num_holidays] h_shapeZ;
  vector[num_holidays] h_skewZ;
  vector<lower = 0, upper = pi()/2>[num_holidays] lambda_m_unif;
}

transformed parameters {
  row_vector[num_examples] log_obs_mean;
  row_vector[num_examples] log_baseline;
  row_vector[num_examples] holiday_effect;

  row_vector[num_examples] seasonality;

  vector[num_holidays] h_skew;
  vector<lower=0>[num_holidays] h_shape;
  vector<lower=0>[num_holidays] h_scale;
  vector[num_holidays] h_loc;
  vector[num_holidays] intensity;

  vector<lower=0>[num_holidays] lambda_m = tan(lambda_m_unif);
  real<lower=0> tau_tilde = tan(tau_tilde_unif);
  real tau = tau0 * tau_tilde; // tau ~ cauchy(0, tau0)
  real c2 = slab_scale2 * c2_tilde;
  vector<lower=0>[num_holidays] lambda_tilde_m = (
    sqrt( c2 * square(lambda_m) ./ (c2 + square(tau) * square(lambda_m)) )
  );

  intensity = sqrt(tau_squared) * lambda_tilde_h .* lambda_tilde;
    
  // PRIOR REPARAMETRIZATION
  seasonality = fourier_coefficients * X_year;

  intensity = tau * lambda_tilde_m .* lambda_tilde;
  h_loc = h_loc_prior_mu + h_loc_prior_sig .* h_locZ;
  h_shape = exp(h_shape_prior_mu + h_shape_prior_sig .* h_shapeZ); //non-centered lognormal
  h_skew = h_skew_prior_mu + h_skew_prior_sig .* h_skewZ;
  h_scale = h_scale_raw ./ h_scale_prior_beta; 

  profile("compute holiday") {
    holiday_effect = get_holiday_lift(
      h_skew, h_shape, h_scale, h_loc, intensity, d_peak, hol_mask
    );
  }

  log_baseline = rep_row_vector(alpha, num_dates);
  if (use_holidays) {
    if (use_seasonality) {
      log_obs_mean = (log_baseline + holiday_effect + seasonality);
    }
    else {
      log_obs_mean = (log_baseline + holiday_effect);
    }
  } else if (use_seasonality) {
    log_obs_mean = (log_baseline + seasonality);
  }
  else {
    log_obs_mean = log_baseline;
  }
    
}

model {

  // PRIORS
  profile("priors") {
    fourier_coefficients ~ std_normal();
    
    alpha ~ beta(alpha_prior_alpha, alpha_prior_beta);
    
    lambda_tilde ~ std_normal();
    lambda_m_unif ~ uniform(0, pi()/2);  // not necessary but pedantic
    tau_tilde_unif ~ uniform(0, pi()/2);  // not necessary but pedantic
    c2_tilde ~ inv_gamma(half_slab_df, half_slab_df);
    h_locZ ~ std_normal();
    h_scale_raw ~ gamma(h_scale_prior_alpha, 1.0);
    h_shapeZ ~ std_normal();
    h_skewZ ~ std_normal();
  }
    
  // LIKELIHOOD
  target += poisson_log_lupmf(obs| log_obs_mean)
}

generated quantities {
  row_vector[num_test_dates] test_obs;
  row_vector[num_test_dates] test_log_obsmean;
  row_vector[num_test_dates] test_log_baseline;
  row_vector[num_test_dates] test_holiday_effect;

  row_vector[num_test_dates] test_seasonality;

  test_baseline = rep_row_vector(alpha,num_test_dates);
  test_holiday_effect = get_holiday_lift(
      h_skew, h_shape, h_scale, h_loc, intensity, d_peak_test, hol_mask_test
  );

  test_seasonality = fourier_coefficients * X_year_test;

  if (use_holidays) {
    if (use_seasonality) {
      test_log_obsmean = (test_log_baseline + test_holiday_effect + test_seasonality);
    }
    else {
      test_log_obsmean = (test_log_baseline + test_holiday_effect);
    }
  } else if (use_seasonality) {
    test_log_obsmean = (test_log_baseline + test_seasonality);
  }
  else {
    test_log_obsmean = test_log_baseline;
  }
  
  for(n in 1:num_test_dates) {
    test_obs[n] = poisson_log_rng(test_log_obsmean[n]);
  }
}