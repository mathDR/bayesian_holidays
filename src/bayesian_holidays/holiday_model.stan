functions {
  int count_mask_runs(matrix hol_mask) {
    int num_holidays = rows(hol_mask);
    int num_dates = cols(hol_mask);
    int total_runs = 0;

    for (h in 1:num_holidays) {
      int in_run = 0;
      for (t in 1:num_dates) {
        if (hol_mask[h, t] != 0.0) {
          if (in_run == 0) {
            total_runs += 1;
            in_run = 1;
          }
        } else {
          in_run = 0;
        }
      }
    }
    return total_runs;
  }

  array[,] int get_mask_runs(matrix hol_mask, int total_runs) {
    int num_holidays = rows(hol_mask);
    int num_dates = cols(hol_mask);
    array[total_runs, 3] int runs;
    int run_idx = 0;

    for (h in 1:num_holidays) {
      int in_run = 0;
      for (t in 1:num_dates) {
        if (hol_mask[h, t] != 0.0) {
          if (in_run == 0) {
            run_idx += 1;
            runs[run_idx, 1] = h;
            runs[run_idx, 2] = t;
            in_run = 1;
          }
          runs[run_idx, 3] = t;
        } else {
          in_run = 0;
        }
      }
    }
    return runs;
  }

  row_vector get_holiday_lift(
    vector h_skew, 
    vector h_shape,
    vector h_scale,
    vector h_loc,
    vector intensity,
    matrix d_peak,
    matrix hol_mask,
    array[,] int mask_runs
  ) {
    int num_dates = cols(d_peak);
    int total_runs = size(mask_runs);
    row_vector[num_dates] tdd = rep_row_vector(0.0, num_dates);
    vector[rows(d_peak)] two_intensity = 2.0 * intensity;
    
    for (r in 1:total_runs) {
      int h = mask_runs[r, 1];
      int s = mask_runs[r, 2];
      int e = mask_runs[r, 3];
      int win_len = e - s + 1;
      
      row_vector[win_len] z = (d_peak[h, s:e] - h_loc[h]) / h_scale[h];
      
      tdd[s:e] += (
        two_intensity[h] * exp(-((square(z) + 1e-12) .^ h_shape[h])) 
        .* inv_logit(h_skew[h] * z)
      ) .* hol_mask[h, s:e];
    }

    return tdd;
  }
}

data {
  // OBSERVATIONS
  int<lower=1> num_dates;
  int<lower=1> num_test_dates;
  int<lower=0> num_holidays;
  array[num_dates] int<lower=0> obs;

  // FEATURE TOGGLES
  int<lower=0, upper=1> use_seasonality;
  int<lower=0, upper=1> use_holidays;

  matrix[num_holidays, num_dates] d_peak;
  matrix[num_holidays, num_test_dates] d_peak_test;

  matrix[num_holidays, num_dates] hol_mask;
  matrix[num_holidays, num_test_dates] hol_mask_test;

  int<lower=0> num_modes_year;
  matrix[2 * num_modes_year, num_dates] X_year;
  matrix[2 * num_modes_year, num_test_dates] X_year_test;

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
  real expected_num_holidays = 3.0;
  real slab_scale = 2.0;
  real slab_scale2 = square(slab_scale);
  real slab_df = 25.0;
  real half_slab_df = 0.5 * slab_df;

  real tau0 = (expected_num_holidays / (num_holidays - expected_num_holidays)) * inv_sqrt(num_dates);

  int total_runs = use_holidays ? count_mask_runs(hol_mask) : 0;
  array[total_runs, 3] int mask_runs;
  if (total_runs > 0) {
    mask_runs = get_mask_runs(hol_mask, total_runs);
  }

  int total_runs_test = use_holidays ? count_mask_runs(hol_mask_test) : 0;
  array[total_runs_test, 3] int mask_runs_test;
  if (total_runs_test > 0) {
    mask_runs_test = get_mask_runs(hol_mask_test, total_runs_test);
  }

}

parameters {
  real log_baseline_real;
  row_vector[2 * num_modes_year] fourier_coefficients;
  
  vector[num_holidays] lambda_tilde;
  real<lower=0> c2_tilde;
  real<lower=0, upper=pi() / 2> tau_tilde_unif;
  vector[num_holidays] h_locZ;
  vector<lower=0>[num_holidays] h_scale_raw;
  vector[num_holidays] h_shapeZ;
  vector[num_holidays] h_skewZ;
  vector<lower=0, upper=pi() / 2>[num_holidays] lambda_m_unif;
}

transformed parameters {
  vector<lower=0>[num_holidays] lambda_m = tan(lambda_m_unif);
  real<lower=0> tau_tilde = tan(tau_tilde_unif);
  real<lower=0> tau = tau0 * tau_tilde;
  real<lower=0> c2 = slab_scale2 * c2_tilde;
  vector<lower=0>[num_holidays] lambda_tilde_m = 
    sqrt(c2 * square(lambda_m) ./ (c2 + square(tau) * square(lambda_m)));

  vector[num_holidays] intensity = tau * lambda_tilde_m .* lambda_tilde;
  vector[num_holidays] h_loc = h_loc_prior_mu + h_loc_prior_sig .* h_locZ;
  vector<lower=0>[num_holidays] h_shape = exp(h_shape_prior_mu + h_shape_prior_sig .* h_shapeZ);
  vector[num_holidays] h_skew = h_skew_prior_mu + h_skew_prior_sig .* h_skewZ;
  vector<lower=0>[num_holidays] h_scale = h_scale_raw ./ h_scale_prior_beta; 

  row_vector[num_dates] log_seasonality = use_seasonality
    ? (fourier_coefficients * X_year)
    : rep_row_vector(0.0, num_dates);

  row_vector[num_dates] holiday_effect;
  profile("compute holiday") {
    holiday_effect = use_holidays
      ? get_holiday_lift(h_skew, h_shape, h_scale, h_loc, intensity, d_peak, hol_mask, mask_runs)
      : rep_row_vector(0.0, num_dates);
  }

  row_vector[num_dates] log_obs_mean = log_baseline_real + holiday_effect + log_seasonality;
}

model {
  profile("priors") {
    fourier_coefficients ~ std_normal();
    log_baseline_real ~ std_normal();
    
    lambda_tilde ~ std_normal();
    c2_tilde ~ inv_gamma(half_slab_df, half_slab_df);
    h_locZ ~ std_normal();
    h_scale_raw ~ gamma(h_scale_prior_alpha, 1.0);
    h_shapeZ ~ std_normal();
    h_skewZ ~ std_normal();
  }
    
  target += poisson_log_lupmf(obs | log_obs_mean);
}

generated quantities {
  row_vector[num_test_dates] test_log_baseline = rep_row_vector(log_baseline_real, num_test_dates);
  row_vector[num_test_dates] test_log_seasonality = use_seasonality
    ? (fourier_coefficients * X_year_test)
    : rep_row_vector(0.0, num_test_dates);

  row_vector[num_test_dates] test_holiday_effect = use_holidays
    ? get_holiday_lift(h_skew, h_shape, h_scale, h_loc, intensity, d_peak_test, hol_mask_test, mask_runs_test)
    : rep_row_vector(0.0, num_test_dates);

  row_vector[num_test_dates] test_log_obsmean = test_log_baseline + test_holiday_effect + test_log_seasonality;
  array[num_test_dates] int test_obs = poisson_log_rng(test_log_obsmean);
}
