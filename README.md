# bayesian_holidays
A Bayesian Holiday Time Series Model


---
title: "Tis the Season...to be Bayesian!"
date: 2020-12-16 09:00:00 -7000
layout: post
linkedinimage: ""
facebookimage: ""
twitterimage: ""
author:
- Daniel Marthaler
graphics:
- Brian Coffey
published: true
location: "San Francisco, CA"
excerpt: <p>Bayesian Holiday Modeling for Time Series Inference</p>
tags:
  - algorithms
---

<style>

@font-face {
  font-family: 'brandontext-svg-bold';
  src: url('/assets/posts/2020-06-02-large-scale-experimentation/fonts/brandontext-svg/31E0E1_0_0.eot');
  src: url('/assets/posts/2020-06-02-large-scale-experimentation/fonts/brandontext-svg/31E0E1_0_0.eot?#iefix') format('embedded-opentype'),
     url('/assets/posts/2020-06-02-large-scale-experimentation/fonts/brandontext-svg/31E0E1_0_0.woff') format('woff'),
     url('/assets/posts/2020-06-02-large-scale-experimentation/brandontext-svg/31E0E1_0_0.ttf') format('truetype'),
     url('/assets/posts/2020-06-02-large-scale-experimentation/brandontext-svg/31E0E1_0_0.svg') format('svg');
  font-weight: normal;
}

@font-face {
  font-family: 'brandontext-svg-regular';
  src: url('/assets/posts/2020-06-02-large-scale-experimentation/fonts/brandontext-svg/31E0E1_2_0.eot');
  src: url('/assets/posts/2020-06-02-large-scale-experimentation/fonts/brandontext-svg/31E0E1_2_0.eot?#iefix') format('embedded-opentype'),
     url('/assets/posts/2020-06-02-large-scale-experimentation/fonts/brandontext-svg/31E0E1_2_0.woff') format('woff'),
     url('/assets/posts/2020-06-02-large-scale-experimentation/brandontext-svg/31E0E1_2_0.ttf') format('truetype'),
     url('/assets/posts/2020-06-02-large-scale-experimentation/brandontext-svg/31E0E1_2_0.svg') format('svg');
  font-weight: normal;
}

</style>

As Mark Twain may have said:  “Prediction is very difficult, especially about the future” [[1](#1)].  Alas, at Stitch Fix we do this all the time. Specifically, we need to predict upcoming demand for our inventory. This allows us to determine which clothes to stock, so we can order the correct number of units for each item, and to allocate those units to the right warehouses, allowing us to deliver delightful fixes for our clients!

As the year unfolds, our demand fluctuates. Two big drivers of that fluctuation are seasonality and holidays. With the holiday season upon us, it's a great time to describe how both seasonality and holiday effects can be estimated, and how you can use this formulation in a predictive time series model.

In this post, we describe the difference between seasonality and holiday effects, posit a general Bayesian Holiday Model, and show how that model performs on some Google Trends data.

If you want to skip to the examples, click [here](#examples_section).

## Seasonality vs Holidays

For the purpose of this post, we are using the term “holiday” to denote an important yearly date that affects your data observations.  This usually means days like Christmas and New Year’s Eve, but it could also refer to dates like the Superbowl, the first day of school, or when hunting season starts.

Holidays are in contrast to "seasonality," which is a longer term effect that can persist over multiple weeks or months.  We will see that both contribute to our data observations, and distinguishing between them is important when we want to make decisions based on the outcome of our model.

Holiday effects show up in many different sorts of time series, including retail sales [[2](#2), [3](#3), [4](#4)], electrical load [[5]](#5), social media events [[6]](#6), along with many others.

Unfortunately, holiday effect data is sparse, since most holidays occur only once a year.  Furthermore, many holidays are entwined within certain "seasonal" time periods. This makes modeling the effect especially difficult.

Most people think of holiday impacts as a spike in time, but this isn't always the case. For example: in many retail stores across the globe, the entire month leading up to Christmas means increased sales. So we would want a holiday effect to reflect this ramp up. Conversely, if you are modeling the number of customers who will visit your brunch spot in late March/early April, you would expect a spike right on Easter Sunday. We would also want our model to reflect this phenomenon. If we are interested in demand for big screen T.V.s at our electronics store, we would expect a drop-off in sales after Super Bowl Sunday.  Each of these examples lead to very different holiday effect profiles.  Therefore, we want a model that is flexible enough to cover these cases (and many others!).

To robustly capture holiday effects within time series data, a model should meet several criteria:

* Flexibility in general shape: Holiday effects may suddenly spike, then disappear, or they may have long ascent/descent effects or sustained elevation over time.
* The effect of the holiday can be positive or negative.
* Discovery of peak location: Holiday effects may peak close to, but not exactly on, a holiday date.
* Sparsity: Holiday effects should be used sparingly, so as not to explain too much variation in the data.

Standard strategies for modeling holidays incorporate them as additive components into a model, and focus on the use of indicator variables for each holiday [[6](#6), [7](#7)]. This strategy places a "yes" on holiday dates and a "no" on the other days, "indicating" which dates are active. To do this, one has to know all of the holiday dates on which to place a "yes." The dates must include each occurrence of the holiday going back in time over all of the training data, and going forward, over the dates you wish to forecast against.  An extension to this idea is having additional indicator variables within a user-specified range of dates before or after the holiday  [[2](#2),[4](#4),[5](#6)]. However, if the holiday effect is long-lasting, growth in the number of indicator variables can lead the model to overfit, especially when the data is sparse.

Below, we outline an approach that simultaneously minimizes overfitting with holidays, while also allowing for determining the contribution of individual holidays to the overall effect of the time series.  

Being able to extract the contribution of each holiday to the observations is important when we want to make decisions based on the outcome of the model. Marketing teams would love to know that an increase in sales is due to Christmas coming up, and not due to Thanksgiving having passed, allowing them to run promotions targeting the correct holiday.


## Bayesian Holiday Modeling
We begin with a list of holidays provided by the user (denoted the *holiday calendar*).  These are the dates that one expects will influence the time series.  For example, if you are a CPA, then you would expect your workload to increase after tax day (April 15th in the United States) so you would have April 15 in your holiday calendar.  Retail marketers, on the other hand, would not necessarily have that date in their holiday calendar, but they might have the date that kids go back to school.

To generate a flexible model that encompasses all of the criteria above, we borrow from the space of probability distributions:  We utilize an unnormalized skew-normal distribution function (where we approximate the normal CDF with an inverse logit). Probability distributions are not typically viewed as "functions" for typical regressions, but we find their flexibility in conforming to parameters exactly what we need for modeling holidays.

The effect from an individual holiday is modeled as a function of the difference between the date of the observation, and the closest date of that holiday. For example, if only Christmas and Easter are included in your holiday calendar, then the observation on February 12 would have holiday features [-49, 60], since it is 49 days (previous) since the nearest Christmas, and 60 days until the nearest Easter. The holiday effect for February 12 would then be a function of these features.

If there are $$H$$ holidays considered in the model, then each observation has $$H$$ holiday features as input to the model ($$H=2$$ in the above example).

### The Holiday Effect Function
Generating a model that encompasses all of the criteria we outlined above involves combining a lot of ingredients.

Our holiday effect function $$h(t)$$ describes the effect of the holiday at date $$t$$ as:  
<div align="center">
$$
h(t) = 2 \lambda \frac{\exp\left(-|z(t)|^\omega\right)}{1+\exp\left(-\kappa z(t)\right)}
$$
</div>
with
<div align="center">
$$
z(t) = \frac{t - \mu}{\sigma} 
$$
</div>

where
+ $$\mu$$ is the location parameter - it denotes how “offset” the effect is from the actual holiday date
+ $$\sigma$$ is the scale parameter - it denotes how broad the effect of the holiday is over time
+ $$\omega$$ is the shape parameter - it denotes how “peaky” the effect is in time
+ $$\kappa$$ is the skew parameter - it denotes how asymmetrical the holiday effect is around $$\mu$$
+ $$\lambda$$ is the intensity parameter - this denotes the magnitude of the holiday effect.

To get a feel for the types of effects able to be modeled, play with the sliders below to generate possible ranges from the Holiday Effect Function.


<svg viewBox="0 0 684 234" enable-background="new 0 0 684 234" xml:space="preserve" style="padding-bottom:30px;padding-top:10px">
<g id="static">
    <line fill="none" stroke="#000000" stroke-width="0.25" stroke-miterlimit="10" x1="242.8" y1="27.7" x2="27.5" y2="27.7"/>
    <line fill="none" stroke="#000000" stroke-width="0.25" stroke-miterlimit="10" x1="242.8" y1="135.6" x2="27.5" y2="135.6"/>
    <line fill="none" stroke="#000000" stroke-width="0.25" stroke-miterlimit="10" x1="242.8" y1="171.5" x2="27.5" y2="171.5"/>
    <line fill="none" stroke="#000000" stroke-width="0.25" stroke-miterlimit="10" x1="63.5" y1="9" x2="63.5" y2="189"/>
    <line fill="none" stroke="#000000" stroke-width="0.25" stroke-miterlimit="10" x1="170.8" y1="9" x2="170.8" y2="189"/>
    <line fill="none" stroke="#000000" stroke-width="0.25" stroke-miterlimit="10" x1="206.8" y1="9" x2="206.8" y2="189"/>
    <line fill="none" stroke="#000000" stroke-width="0.25" stroke-miterlimit="10" x1="242.8" y1="9" x2="242.8" y2="189"/>
    <text transform="matrix(1 0 0 1 11.7402 31)" font-family="'brandontext-svg-regular'" font-size="10px">4</text>
    <text transform="matrix(1 0 0 1 12.4702 67)" font-family="'brandontext-svg-regular'" font-size="10px">2</text>
    <text transform="matrix(1 0 0 1 11.5801 103)" font-family="'brandontext-svg-regular'" font-size="10px">0</text>
    <text transform="matrix(1 0 0 1 8.6401 139)" font-family="'brandontext-svg-regular'" font-size="10px">-2</text>
    <text transform="matrix(1 0 0 1 7.9102 175)" font-family="'brandontext-svg-regular'" font-size="10px">-4</text>
    <line fill="none" stroke="#000000" stroke-width="0.25" stroke-miterlimit="10" x1="242.8" y1="8.5" x2="27.5" y2="8.5"/>
    <text transform="matrix(1 0 0 1 19.1099 207)" font-family="'brandontext-svg-regular'" font-size="10px">-20</text>
    <text transform="matrix(1 0 0 1 56.1899 207)" font-family="'brandontext-svg-regular'" font-size="10px">-10</text>
    <text transform="matrix(1 0 0 1 95.79 207)" font-family="'brandontext-svg-regular'" font-size="10px">0</text>
    <text transform="matrix(1 0 0 1 130.105 207)" font-family="'brandontext-svg-regular'" font-size="10px">10</text>
    <text transform="matrix(1 0 0 1 165.0249 207)" font-family="'brandontext-svg-regular'" font-size="10px">20</text>
    <text transform="matrix(1 0 0 1 200.9751 207)" font-family="'brandontext-svg-regular'" font-size="10px">30</text>
    <text transform="matrix(1 0 0 1 236.6602 207)" font-family="'brandontext-svg-regular'" font-size="10px">40</text>

    <text transform="matrix(1 0 0 1 441 31)" font-family="'brandontext-svg-regular'" font-size="12px">Skew (&#954;)</text>

    <text transform="matrix(1 0 0 1 441 67)" font-family="'brandontext-svg-regular'" font-size="12px">Shape (&#969;)</text>

    <text transform="matrix(1 0 0 1 441 103)" font-family="'brandontext-svg-regular'" font-size="12px">Scale (&#963;)</text>

    <text transform="matrix(1 0 0 1 441 139)" font-family="'brandontext-svg-regular'" font-size="12px">Loc (&#956;)</text>

    <text transform="matrix(1 0 0 1 441 175)" font-family="'brandontext-svg-regular'" font-size="12px">Intensity (&#955;)</text>

    <text transform="matrix(1 0 0 1 596.5986 54)" font-family="'brandontext-svg-regular'" font-size="12px">Examples</text>
    <text transform="matrix(1 0 0 1 23.2217 225)" font-family="'brandontext-svg-regular'" font-size="12px">number of days away from the holiday date</text>
</g>
<g id="y-0">
    <line fill="none" stroke="#000000" stroke-width="0.25" stroke-miterlimit="10" x1="242.8" y1="99.6" x2="27.5" y2="99.6"/>
</g>
<g id="y-2">
    <line fill="none" stroke="#000000" stroke-width="0.25" stroke-miterlimit="10" x1="242.8" y1="63.7" x2="27.5" y2="63.7"/>
</g>
<g id="x-0">
    <line fill="none" stroke="#000000" stroke-width="0.25" stroke-miterlimit="10" x1="99.4" y1="9" x2="99.4" y2="189"/>
</g>
<g id="x-10">
    <line fill="none" stroke="#000000" stroke-width="0.25" stroke-miterlimit="10" x1="135.4" y1="9" x2="135.4" y2="189"/>
</g>
<g id="axes">
    <line fill="none" stroke="#000000" stroke-miterlimit="10" x1="27.5" y1="9" x2="27.5" y2="189"/>
    <line fill="none" stroke="#000000" stroke-miterlimit="10" x1="242.8" y1="189.5" x2="27.5" y2="189.5"/>
</g>
<g id="slider-lines">
    <line fill="none" stroke="#000000" stroke-width="0.25" stroke-miterlimit="10" x1="279" y1="27" x2="387" y2="27"/>
    <line fill="none" stroke="#000000" stroke-width="0.25" stroke-miterlimit="10" x1="279" y1="63" x2="387" y2="63"/>
    <line fill="none" stroke="#000000" stroke-width="0.25" stroke-miterlimit="10" x1="279" y1="99" x2="387" y2="99"/>
    <line fill="none" stroke="#000000" stroke-width="0.25" stroke-miterlimit="10" x1="279" y1="135" x2="387" y2="135"/>
    <line fill="none" stroke="#000000" stroke-width="0.25" stroke-miterlimit="10" x1="279" y1="171" x2="387" y2="171"/>
</g>
<g id="slider-skew">
    <circle fill="#FFFFFF" stroke="#000000" stroke-width="0.25" stroke-miterlimit="10" cx="351" cy="27" r="9"/>
</g>
<g id="slider-shape">
    <circle fill="#FFFFFF" stroke="#000000" stroke-width="0.25" stroke-miterlimit="10" cx="297" cy="63" r="9"/>
</g>
<g id="slider-scale">
    <circle fill="#FFFFFF" stroke="#000000" stroke-width="0.25" stroke-miterlimit="10" cx="342" cy="99" r="9"/>
</g>
<g id="slider-loc">
    <circle fill="#FFFFFF" stroke="#000000" stroke-width="0.25" stroke-miterlimit="10" cx="324" cy="135" r="9"/>
</g>
<g id="slider-intensity">
    <circle fill="#FFFFFF" stroke="#000000" stroke-width="0.25" stroke-miterlimit="10" cx="306" cy="171" r="9"/>
</g>
<g id="text-skew">
    <text transform="matrix(1 0 0 1 432 31)" font-family="'brandontext-svg-bold'" font-size="12px">5.04</text>
</g>
<g id="text-shape">
    <text transform="matrix(1 0 0 1 432 67)" font-family="'brandontext-svg-bold'" font-size="12px">0.80</text>
</g>
<g id="text-scale">
    <text transform="matrix(1 0 0 1 432 103)" font-family="'brandontext-svg-bold'" font-size="12px">4.20</text>
</g>
<g id="text-loc">
    <text transform="matrix(1 0 0 1 432 139)" font-family="'brandontext-svg-bold'" font-size="12px">-0.20</text>
</g>
<g id="text-intensity">
    <text transform="matrix(1 0 0 1 432 175)" font-family="'brandontext-svg-bold'" font-size="12px">2.80</text>
</g>
<g id="button-christmas">
    <path fill="#FFFFFF" stroke="#000000" stroke-width="0.25" stroke-miterlimit="10" d="M663,99h-84c-6.6,0-12-5.4-12-12v-3
        c0-6.6,5.4-12,12-12h84c6.6,0,12,5.4,12,12v3C675,93.6,669.6,99,663,99z"/>
    <text transform="matrix(1 0 0 1 621 90)" font-family="'brandontext-svg-regular'" font-size="12px" text-anchor="middle">Santa Claus</text>
</g>
<g id="button-tax">
    <path fill="#FFFFFF" stroke="#000000" stroke-width="0.25" stroke-miterlimit="10" d="M663,135h-84c-6.6,0-12-5.4-12-12v-3
        c0-6.6,5.4-12,12-12h84c6.6,0,12,5.4,12,12v3C675,129.6,669.6,135,663,135z"/>
    <text transform="matrix(1 0 0 1 621 126)" font-family="'brandontext-svg-regular'" font-size="12px" text-anchor="middle">Tax Return</text>
</g>
<g id="button-defaults">
    <path fill="#FFFFFF" stroke="#000000" stroke-width="0.25" stroke-miterlimit="10" d="M663,171h-84c-6.6,0-12-5.4-12-12v-3
        c0-6.6,5.4-12,12-12h84c6.6,0,12,5.4,12,12v3C675,165.6,669.6,171,663,171z"/>
    <text transform="matrix(1 0 0 1 621 162)" font-family="'brandontext-svg-regular'" font-size="12px" text-anchor="middle">Defaults</text>
</g>
<g id="graph-circles">
</g>
</svg>

As the sliders show, our holiday effect function allows for extreme flexibility in both shape and location.

Two examples for the model are included for illustrative purposes.  When pressing the "Santa Claus" button, we see a holiday effect corresponding to the ramp-up of searches for "Santa Claus" in Google Trends data. The effect starts to ramp about 7 days before Christmas, reaching its peak on the evening of the 24<sup>th</sup> of December, then abruptly crashing after the holiday.

The "Tax Return" example shows the effect of searches for "Tax Return" in Google Trends data.  Searches do not begin until after April 15th (Tax Day in the United States) and slowly taper off throughout April and into early May.
 

### Priors
Typically, we embed the holiday effect function within a larger model, like as a component for the mean of a count distribution.  Using strict priors on the relatively few parameters, the function is able to express a wide variety of holiday effects without overfitting.

Setting priors for the parameters of the holiday effect function is done similarly to other Bayesian workflows [[8](#8)]:  Prior predictive checks should be completed to determine relevant scales and default values.

For the parameters in the model, the distributions chosen were:
+ $$\mu \sim \mathcal{N}\left(0, \sigma_{\mu}\right)$$.
+ $$\sigma \sim \textrm{Gamma}\left(k_{\sigma}, \theta_{\sigma}\right)$$.
+ $$\omega  \sim \textrm{Gamma}\left(k_{\omega},\theta_{\omega}\right)$$.
+ $$\kappa \sim \mathcal{N}\left(0, \sigma_{\kappa}\right)$$.
+ $$\lambda$$ is drawn from a regularized horseshoe [[9](#9)]

Setting the prior mean for $$\mu$$ at zero implies we expect the effect to peak on the official holiday date. The standard deviation for the location, $$\sigma_{\mu},$$ should be chosen such that each holiday doesn’t drift too far away from its original location.  The coefficients of $$\sigma$$ are chosen such that the locality of each holiday is controlled; preventing the effects of one holiday from exceeding the dates of its closest holiday neighbors.  The scale and skew can be tuned to implement beliefs in how the effect of the holiday persists about the holiday date.  Of course, using weakly informative priors here is a good alternative strategy if no prior information about your data are known (but this is rarely the case!).

The regularized horseshoe prior [[9](#9)] for $$\lambda$$ can be considered as a continuous extension of the spike-and-slab prior. This is a way of merging a prior at zero (the "spike") with a prior away from zero (the "slab"), and letting the model decide if the coefficient should have a non-zero posterior.  This encourages sparsity and resists using holidays to explain minor variation in the data.


#### Regularized Horseshoe
The coefficient for the intensity of each holiday (the $$\lambda$$ parameter) is determined by the following prior:
<div align="center">
$$\begin{aligned}
\lambda &\sim \mathcal{N}\left(0,\tilde{\lambda_h}^2 \tau^2\right) \\
\tilde{\lambda_h}^2 &= \frac{c^2 \lambda_h^2}{c^2 + \tau^2\lambda_h^2} \\
\lambda_h &\sim \textrm{Cauchy}^{+}\left(0,1\right) \\
c^2 &\sim \textrm{Inv-Gamma}\left(\frac{\nu}{2},s^2\frac{\nu}{2}\right) \\
\tau^2 &\sim \textrm{Cauchy}^{+}\left(0,\tau_0\right) \\
\tau_0 &= \frac{h_0}{H-h_0}
\end{aligned}$$
</div>

where $$h_0$$ is the expected number of active holidays, and $$H$$ is the total number of unique holidays in the holiday calendar.

The prior on $$c^2$$ translates to a Student-$$t_{\nu}\left(0, s^2\right)$$ slab for the coefficients far from zero. We choose $$s = 3$$ and $$\nu = 25$$ and are typically good default choices for a weakly informative prior [[9](#9)].

Having this prior on the holiday intensity is imperative for the model to work at all.  Having no sparsity-preserving prior leads to each holiday being activated to overfit the observed data.  Enforcing regularization via the horseshoe allows for realistic intensity values to be inferred from the data.

Immediate extensions to the model would be to have correlated structure among the holiday intensities, that would allow for prior beliefs of how holidays act together.  You might know, for instance, that the effects of Labor Day and Memorial Day covary. Therefore, you would want to tie the intensities of each to each other.


## Modeling Seasonality
Along with holidays, another critical component of most time series forecasting is to determine the effect of "seasonality" on the observed data.  We assume the seasonality has a long extent, but could consist of multiple seasonal periods.

To this end we utilize *Dynamic Harmonic Regression*: utilizing Fourier modes to model the seasonality of our data.  The model is a linear combination of sines and cosines, with period chosen to represent the estimated period (yearly, quarterly, weekly, etc.).  Advantages of this approach include:
* allowing for any length of seasonality
* mixing different seasonal periods (modeling both yearly and quarterly for instance) 
* controlling how smooth the seasonal pattern becomes by varying the number of Fourier modes. 

The only real disadvantage of this method is the assumption that the seasonality is fixed; it cannot change over time.  In practice, seasonality is fairly constant, so this is not a big problem, except perhaps for long time series [[7](#7)].

### Seasonal Model
The seasonality model, $$S(t)$$ for a single seasonal period is 
<div align="center">
$$
S(t) = \sum\limits_{n=1}^{N} a_n \sin\left(\frac{2\pi n t}{T}\right) + b_n \cos\left(\frac{2\pi n t}{T}\right)
$$
</div>
where $$N$$ denotes the number of Fourier modes, and $$T$$ denotes the seasonal period.

We expect only a few modes of the Fourier series are required to capture the seasonal trends. Adding many modes would lead to overfitting in the model, which is another good reason to keep the number of modes low.

## <a name="examples_section"></a>Examples
To demonstrate the utility of the Bayesian Holiday model, we show how it generalizes on Google Trends data. Google Trends is a feature that shows how often a given term is searched on Google, relative to all other words over a given period of time.

For our examples below, we select the previous five years of data for the given search term, and fit a model using the first four years as training data.  We then use this learned model to predict the timeseries for the fifth year and compare it to the actual search data.

We use a Poisson likelihood, as the Trends data is represented in counts. The mean of the Poisson distribution is modeled with a log-link function comprising three components: a baseline, seasonality features, and the holiday effect.

The baseline is an inferred constant, while seasonality and the holiday effects are the models described above. 

So the full model becomes:
<div align="center">
$$
y(t) \sim \mathrm{Poisson}\left(\exp\left[\alpha + S(t) + H(t)\right]\right)
$$
</div>

with

<div align="center">
$$
H(t) = \sum\limits_{n=1}^H h_n(t)
$$
</div>

Here $$y(t)$$ denotes the number of times the search term appears at time $$t$$. $$S(t)$$ is the seasonal component of the model. We use $$N=3$$ Fourier models throughout and set the period to $$T=52.1429,$$ since the data is weekly. Also, $$h_n(t)$$ is a single holiday instance, and $$H$$ is the number of holidays in the holiday calendar.

This model gets at the crux of the difference between seasonality and holiday effects: seasonality is a global concept, while the holiday effect is localized in time.   

### Holiday Calendar
The holidays for the examples were selected from an American calendar:
* New Year's Day
* Martin Luther King Jr. Day
* Valentine's Day
* Easter
* Mother's Day
* Memorial Day
* Father's Day
* Independence Day (4<sup>th</sup> of July)
* Labor Day
* Columbus Day/Indigenous Peoples' Day
* Halloween
* Thanksgiving
* Christmas Day
 
### Fireworks [[10](#10)]
Our first example shows the time series of counts for the search term “fireworks” (in black).  The blue lines denote the posterior samples from fitting the model, and the orange lines are the posterior draws for the holdout time period.  Each year is shaded, as are the training (blue) and holdout (orange) time periods.

We see that the model fits the data extremely well!

<figure style="margin-bottom: 1rem;margin-left: 2rem;margin-right: 2rem;">
  <img alt="The posterior samples (and means) for both in and out of sample inference for the search term fireworks." src="/assets/posts/2020-12-16-tis-the-season-to-be-bayesian/fireworks_worldwide.png"/>
  <figcaption style="text-align:center"><i>The posterior samples (and means) for in-sample (blue) and out of sample (orange) draws for the search term "fireworks".</i></figcaption>
</figure>
<br/>  


Inspecting the fit, we see the expected spikes for the Fourth of July and New Year's. 

Since our model is additive, we can observe the effect of individual holidays on the observations.  The following figure shows the individual posterior traces for each holiday in the last two years of training and the full holdout set. 

<figure style="margin-bottom: 1rem;margin-left: 2rem;margin-right: 2rem;">
  <img alt="The posterior samples (and means) of sample inference for individual holidays for the search term fireworks." src="/assets/posts/2020-12-16-tis-the-season-to-be-bayesian/individual_fireworks.png"/>
  <figcaption style="text-align:center"><i>The posterior samples (and means) of sample inference for individual holidays for the search term "fireworks".</i></figcaption>
</figure>
<br/>

We see that the three spikes in searches are made up of 
* A Combinations of New Year's Day and Christmas
* A Combination of Father's Day and Independence Day
* Halloween (seemingly)

with a small amount of seasonality to model the fit.

The spikes about Independence Day and New Year's are expected, but at least for those living in the United States, the activation of Halloween seems very odd! It is not a traditional holiday to expect fireworks.  To those living in the Commonwealth, however, there is an obvious explanation:  Guy Fawkes Day.

Guy Fawkes day occurs on November 5<sup>th</sup>. It marks the anniversary of the discovery of a plot to blow up the Houses of Parliament in London in 1605. Within the UK it is customary to have both fireworks and bonfires to celebrate the holiday.

This example shows the importance of having the correct holiday calendar for the data that you are modeling.  Having an American calendar to model global data can lead to nonsensical results (as happened in this case).  An important caveat is that our model fits the data to a close holiday (Halloween), and this behavior is a good posterior check to see if your calendar is appropriate.

Another interesting insight is the activation of Father's Day. Father's Day occurs on the third Sunday of June each year. This is another holiday where we would not expect to see prevalent fireworks. In this instance, by looking at the values of the intensities between Father's Day and Independence Day, the model attributes searches for fireworks before July 4<sup>th</sup> *both* to Independence Day *and* Father's Day. 

Given the weakly-informative priors for the baseline model, we do not *a priori* encode any difference between the two holidays, so the model is free to select from either. Knowing we do not expect to see fireworks on Father's Day, we should update our priors to make it search-term specific (i.e. put a much smaller prior on the intensity for Father's Day than that of the 4<sup>th</sup> of July).

The same phenomenon occurs between Christmas and New Year's:  the model estimates the search term counts are due to a split between the two holidays, since no prior information was given to distinguish between them.


### Chocolate [[11](#11)]
The plot below shows a time series of counts for the search term “chocolate” (again in black).  Again, the blue lines denote the posterior samples from fitting the model, and the orange lines are the posterior draws for the holdout time period.

The model does a great job of fitting the time series data. In particular, capturing the spikes near Thanksgiving, Christmas and Valentine's Day. The low-frequency seasonality and baseline capture the remaining counts throughout the year.

<figure style="margin-bottom: 1rem;margin-left: 2rem;margin-right: 2rem;">
  <img alt="The in-sample (blue) and out of sample (orange) posterior samples (and means) for the search term chocolate." src="/assets/posts/2020-12-16-tis-the-season-to-be-bayesian/chocolate.png"/>
  <figcaption style="text-align:center"><i>The in-sample (blue) and out of sample (orange) posterior samples (and means) for the search term "chocolate".</i></figcaption>
</figure> 
<br/>  

An interesting artifact of the holdout time period, is that the model completely misses the lift in searches around March of 2020.  This is due to the model extrapolating from historical data, and there has never historically been a lot of searches for chocolate in March.  One thing that also has never occured in March is a large number of people sheltering-in-place due to COVID-19. Perhaps staying at home during a global pandemic makes people want chocolate?


### Pumpkin Spice [[12](#12)]
Our last example shows the power of the model to fit "holidays" that do not fit a typical calendar definition.  The search term "Pumpkin Spice" aligns with the (re-)introduction of the Pumpkin Spice Latte (PSL) from Starbucks each autumn.  

The fit against the searches below shows a few interesting observations:  The first is that the searches occur over a long period of time (early September through late November).  The second is that the overall lift is punctuated by spikes of searches in two areas (seemingly early September and mid-November).

<figure style="margin-bottom: 1rem;margin-left: 2rem;margin-right: 2rem;">
  <img alt="The posterior samples (and means) of sample inference for individual holidays for the search term pumpkin spice." src="/assets/posts/2020-12-16-tis-the-season-to-be-bayesian/pumpkin_spice.png"/>
  <figcaption style="text-align:center"><i>The posterior samples (and means) of sample inference for individual holidays for the search term "pumpkin spice".</i></figcaption>
</figure>
<br/>

So how is the model fitting this data?  Through a combination of seasonality and holiday effects. As seen in the following figure, coupled with the seasonal component modeling the long duration lift, a combination of Labor Day and Thanksgiving lead to the fit we expect for people searching for their PSLs. 

<figure style="margin-bottom: 1rem;margin-left: 2rem;margin-right: 2rem;">
  <img alt="The posterior samples (and means) of sample inference for seasonality and individual holidays for the search term pumpkin spice." src="/assets/posts/2020-12-16-tis-the-season-to-be-bayesian/individual_ps.png"/>
  <figcaption style="text-align:center"><i>The posterior samples (and means) of sample inference for seasonality and individual holidays for the search term "pumpkin spice".</i></figcaption>
</figure>
<br/>

In this example, Labor Day is doing the work of fitting the spike at the beginning of the PSL promotion at Starbucks.  Our simplistic model has limited power to capture this dynamic, so it utilizes the closest holiday to the start of the promotion, namely Labor Day.  In reality, a Starbuck's model would probably also include data features like marketing spend and other components that would allow for robust inference in the true domain.

Furthermore, the posterior samples for Labor Day expose a weakness of the model:  there is no hard constraint that prevents any holiday from having support for long periods of time.  Therefore, Labor Day, which occurs in early September, has support over a full two months leading up to the holiday. An extension of the model would be to enforce compact support for each holiday to be within the time period between the previous to the next holiday. So in this instance, the support for Labor Day could not surpass Independence Day previously, nor exceed Columbus Day/Indiginous People's Day going forward.


## Conclusion
Having a principled holiday effect function embedded within your time series model is crucial to be able to extract knowledge about how holidays affect your observed data.  

Being able to decompose individual effects is a powerful tool to determine if your strategies or decisions are doing what you expect.  Furthermore, having a flexible model let's you determine if you are even using the right holiday calendar!

Happy Holidays from all of us at Stitch Fix!!


## Attributions
Part of this work was done with Alex Braylan while the author was at Revionics (now Atios), and presented at StanCon Helsinki in August 2018. 


## References
<a id="1">[1]</a>
[The Perils of Prediction](https://www.economist.com/letters-to-the-editor-the-inbox/2007/07/15/the-perils-of-prediction-june-2nd)

<a id="2">[2]</a> 
Tucker S. McElroy, Brian C. Monsell, and
Rebecca Hutchinson.  “Modeling of Holiday
Effects and Seasonality in Daily Time Series.”
Center for Statistical Research and Methodology
Report Series RRS2018/01. Washington: US Census Bureau (2018).

<a id="3">[3]</a> 
Usha Ramanathan and Luc Muyldermana. "Identifying demand factors for promotional
planning and forecasting: A case of a soft drink company in the UK". International Journal of
Production Economics, 128 (2), 538-545 (2010)

<a id="4">[4]</a>
Nari S. Arunraj and Diane Ahrens. "Estimation of non-catastrophic weather
impacts for retail industry". International Journal of Retail and Distribution
Management, 44:731–753 (2016)

<a id="5">[5]</a> 
Florian Ziel. "Modeling public holidays in load forecasting: a German case study". Journal of Modern Power Systems and Clean Energy 6(2):191–207 (2018).

<a id="6">[6]</a> 
Sean J. Taylor and Benjamin Letham. "Forecasting at scale". The American Statistician (2017).

<a id="7">[7]</a>
Rob J. Hyndman and George Athanasopoulos. Forecasting: Principles and Practice,
online: https://otexts.com/fpp2/ (2012)

<a id="8">[8]</a>
Andrew Gelman, Aki Vehtari, Daniel Simpson, Charles C. Margossian, 
Bob Carpenter, Yuling Yao, Lauren Kennedy, Jonah Gabry, Paul-Christian Bürkner, 
Martin Modrák. Bayesian Workflow. https://dpsimpson.github.io/pages/talks/Bayesian_Workflow.pdf

<a id="9">[9]</a>
Juho Piironen and Aki Vehtari. “Sparsity information and regularization in the horseshoe and other shrinkage priors.” Electronic Journal of Statistics, 11(2): 5018–5051 (2017)

<a id="10">[10]</a>
[Fireworks Google Trends](https://trends.google.com/trends/explore?date=2015-11-02%202020-11-02&geo=US&q=fireworks)

<a id="11">[11]</a>
[Chocolate Google Trends](https://trends.google.com/trends/explore?date=2015-11-02%202020-11-02&geo=US&q=chocolate)

<a id="12">[12]</a>
[Pumpkin Spice Google Trends](https://trends.google.com/trends/explore?date=2015-11-02%202020-11-02&geo=US&q=pumpkin%20spice)

<script src="https://d3js.org/d3.v4.min.js"></script>

<script>

// config

var slider_config = {
  skew: {min: -5, max: 5, val: 0.0},
  shape: {min: 0.5, max: 10, val: 1.0},
  scale: {min: 0.1, max: 10, val: 1.0},
  loc: {min: -10, max: 20, val: 0.0},
  intensity: {min: -2.0, max: 2.0, val: 1.0},
}

var presets = {
  'christmas': {
    skew: -4.89,
    shape: 1.1,
    scale: 2.3,
    loc: -0.5,
    intensity: 1.68,
  },
  'tax': {
    skew: 5.04,
    shape: 0.8,
    scale: 4.2,
    loc: -0.2,
    intensity: 2.0,
  },
  'defaults': {
    skew: 0.0,
    shape: 1.0,
    scale: 1.0,
    loc: 0.0,
    intensity: 1.0,
  }
}

// internal variables
var slider_x_min = 279
var slider_x_max = 387
var y0 = 99.6
var y1 = 63.7
var dy = y1 - y0
var x0 = 99.4
var x10 = 135.4
var dx = (x10 - x0) / 10
var holiday_graph_data

// compute the graph data
function compute_graph(slider_config) {

  const h_skew = slider_config.skew.val
  const h_shape = slider_config.shape.val
  const h_scale = slider_config.scale.val
  const h_loc = slider_config.loc.val
  const lam = slider_config.intensity.val

  var data_out = []
  for (var i = 0; i < 1200; i++) {
    var x = -20 + 0.05*i
    var z = (x-h_loc) / h_scale
    var y = 2.0 * lam * Math.exp(-1.0*Math.pow(Math.abs(z),h_shape)) /
             (1.0 + Math.exp(-h_skew * z))
    data_out.push({x: x, y: y})
  }

  return data_out
}

// update the graph (including calling compute)
function update_graph(duration){

  holiday_graph_data = compute_graph(slider_config)

  var valueline = d3.line()
    .x(d => x0 + dx * d.x)
    .y(d => y0 + dy * d.y)

  d3.select("#graph-circles").selectAll("path").data([holiday_graph_data])
    .transition().duration(duration)
      .attr("d", valueline)
  
}

// update sliders (and call graph update)
function update_slider(key, new_val, duration){

  // slider value
  slider_config[key].val = new_val

  // slider location
  var k = (new_val - slider_config[key].min) / (slider_config[key].max - slider_config[key].min)
  var new_cx = slider_x_min + k * (slider_x_max - slider_x_min)
  d3.select("#slider-" + key).select("circle")
    .transition().duration(duration)
      .attr("cx", new_cx)

  // slider text
  d3.select("#text-" + key).select("text").text("" + new_val.toFixed(2))

  // update graph
  update_graph(duration)

}


// initializations on page load
function on_load(){

  // sliders
  Object.keys(slider_config).forEach(function(key){

    // right-align text outputs
    d3.select("#text-" + key).select("text").attr("text-anchor", "end")

    // initialize slider locations
    update_slider(key, slider_config[key].val)

    // slider action
    d3.select("#slider-" + key).select("circle")
      .style("cursor", "pointer")
      .call(d3.drag().on("drag", function() {
        var new_cx = Math.max(slider_x_min, Math.min(slider_x_max, d3.event.x))
        var k = (new_cx - slider_x_min) / (slider_x_max - slider_x_min)
        var new_val = slider_config[key].min + k * (slider_config[key].max - slider_config[key].min)
        update_slider(key, new_val, 0)
      }))

  })

  // buttons
  Object.keys(presets).forEach(function(key){
    d3.select("#button-" + key).selectAll("path, text").style("cursor", "pointer")
      .on("click", function() {
        Object.keys(presets[key]).forEach(function(slider_key){
          update_slider(slider_key,  presets[key][slider_key], 750)
        })
      })
  })

  // graph
  d3.select("#graph-circles").selectAll("path").data([holiday_graph_data])
    .enter().append("path")
      .style("stroke-width", 2)
      .style("stroke", "#ff5c61")
      .style("fill", "none")

  update_graph()

}

on_load()

</script>
