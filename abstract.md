# cherry blossom prediction abstract
frankiethull

### abstract

This submission presents a novel approach for predicting the peak bloom
date of cherry blossoms using an ensemble of weak Support Vector
Machines (SVMs) and February external regressors. The method was applied
to five locations: Kyoto, Liestal, Washington D.C., New York City, and
Vancouver. For the first three locations, a variety of specialty kernels
were employed with SVMs using the experimental R package {maize}, which
were then stacked to create an ensemble model. These SVMs utilized a
carefully selected set of predictors based on February data, including
total precipitation, total relative humidity, maximum temperature,
carbon levels, year, and an additional variable called “last freeze
day.” The last freeze day required an additional wavelet SVM for
accurate prediction. For New York City and Vancouver, where historical
data was limited, a latitude-based ensemble was created using the SVM
stacks from the other three locations, with a bias adjustment based on
the previous year’s bloom date.

### additional framework details

#### zorghum zakura method

The historical weather dataset used in this study was obtained from
NASAPOWER, a reliable source for climate data. This dataset was merged
with bloom date data spanning from 2001 to 2024, providing a
comprehensive foundation for the analysis. The combined dataset was then
divided into multiple subsets to facilitate the training of various SVM
models.

Nine distinct kernels were employed in training the SVMs: laplacian,
hyperbolic tangent, cosine similarity, cauchy, t-student, RBF-ANOVA,
tanimoto, wavelet, and fourier. This diverse set of kernels allowed for
capturing different patterns and relationships within the data. The
training process resulted in over 100 potential SVMs, providing a rich
pool of models for stacking.

The stacking method utilized the {stacks} library, which ensembles the
top-performing models. Various stacking approaches were tested,
including elastic-net and LASSO ensembling techniques. The final member
stack was carefully selected to create the most accurate bloom day of
year prediction.

One key innovation in this approach was the inclusion of the “last
freeze day” as a predictor. Since this day typically occurs in March, a
separate SVM with a wavelet kernel was developed to estimate this
variable before incorporating it into the member stack. This additional
step improved the overall accuracy of the predictions.

To provide a measure of uncertainty, prediction intervals were generated
based on the spread of point forecasts within the stack. This approach
offers insights into the potential range of bloom dates, accounting for
variability in the predictions.

For the 2025 predictions, temperature and humidity data were sourced
from openmeteo, as NASAPOWER data is subject to a delay. However, due to
significant discrepancies between NASAPOWER-corrected precipitation and
openmeteo precipitation data, February 2025 precipitation was predicted
using an SVM with a sorensen kernel. This ensured consistency and
accuracy in the precipitation input for the model.

#### elote method

To address the challenge of data-poor situations, such as Vancouver and
New York City where insufficient historical data was available for
training SVMs, an alternative approach called the elote method was
developed. This method assumes that these locations can be modeled based
on their latitudinal relationship to Washington D.C. and Liestal.

The elote method creates predictions for Vancouver and New York City
using a latitude-weighted blend of the ensembled models from Washington
D.C. and Liestal. To account for local variations and recent trends,
these predictions were then bias-adjusted based on a five-day spread
observed in the bloom date for 2024.

It’s worth noting that despite its geographical proximity, Kyoto was not
included in the elote method due to its distinct weather patterns, which
differ significantly from the North American and European locations in
the study.

<div id="sqdgyfkfqc" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
  &#10;  

| peak cherry blossom bloom predictions |            |       |       |
|:--------------------------------------|-----------:|------:|------:|
| novel SVM ensemble technique          |            |       |       |
| location                              | prediction | lower | upper |
| kyoto                                 |       92.6 |  90.7 | 124.4 |
| liestal                               |       87.6 |  75.3 | 115.8 |
| washingtondc                          |       88.5 |  74.6 |  91.6 |
| newyorkcity                           |       89.4 |  76.2 | 103.8 |
| vancouver                             |       87.0 |  73.8 | 101.4 |

</div>

This novel approach, combining SVMs with many kernels & few external
regressors is designed as a creative solutions for cherry bloom
prediction. Providing a framework of stacking many SVMs, which would
typically not be the ideal model of choice for this competition.
