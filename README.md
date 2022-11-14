# bayesian_holidays
A Bayesian Holiday Time Series Model

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

$$
h(t) = 2 \lambda \frac{\exp\left(-|z(t)|^\omega\right)}{1+\exp\left(-\kappa z(t)\right)}
$$

with

$$
z(t) = \frac{t - \mu}{\sigma} 
$$


where
+ $$\mu$$ is the location parameter - it denotes how “offset” the effect is from the actual holiday date
+ $$\sigma$$ is the scale parameter - it denotes how broad the effect of the holiday is over time
+ $$\omega$$ is the shape parameter - it denotes how “peaky” the effect is in time
+ $$\kappa$$ is the skew parameter - it denotes how asymmetrical the holiday effect is around $$\mu$$
+ $$\lambda$$ is the intensity parameter - this denotes the magnitude of the holiday effect.

 
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
