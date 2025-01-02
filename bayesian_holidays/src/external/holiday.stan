functions {

  row_vector get_holiday_lift(
    vector h_skew, 
    vector h_shape,
    vector h_scale,
    vector h_loc,
    vector intensity,
    matrix d_peak,
    matrix hol_mask
  );
  /*{
    int num_holidays = dims(d_peak)[1];
    int num_dates = dims(d_peak)[2];

    row_vector[num_dates] z;
    row_vector[num_dates] tdd = zeros_row_vector(num_dates);
    
    for (h in 1:num_holidays) {
      z = (d_peak[h, :] - h_loc[h]) ./ h_scale[h];
      tdd += (2.0 * intensity[h] * exp(-pow(square(z),h_shape[h])) .* 
        inv_logit(h_skew[h] * z)
        ) .* hol_mask[h,:];
    }

    return tdd;

  }*/

}

data {

  // OBSERVATIONS
  int<lower=1> num_dates; // number of dates
  int<lower=0> num_holidays; // number of holidays
  array [num_dates] int<lower=0> obs;

  matrix[num_holidays, num_dates] d_peak; // distance (in time) from holiday

  matrix[num_holidays, num_dates] hol_mask;

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
  real expected_num_holidays = 3.0;  // Expected number of activated holidays
  real slab_scale = 2.0;    // Scale for large slopes
  real slab_scale2 = square(slab_scale);
  real slab_df = 25.0;      // Effective degrees of freedom for large slopes
  real half_slab_df = 0.5 * slab_df;

  real tau0 = (expected_num_holidays / (num_holidays - expected_num_holidays)) * (1.0 / sqrt(1.0 * num_dates));

}

parameters {
  
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
  row_vector[num_dates] holiday_effect;

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

  intensity = tau * lambda_tilde_m .* lambda_tilde;

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

    
}

model {

  // PRIORS
  profile("priors") {
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
  target += poisson_log_lupmf(obs| holiday_effect);
}