# maize & cherry blossoms


# nyc

### get

``` r
raw_nyc <- "https://raw.githubusercontent.com/GMU-CherryBlossomCompetition/peak-bloom-prediction/refs/heads/main/data/nyc.csv"

nyc_df <- data.table::fread(raw_nyc)
```

``` r
library(nasapower)
library(dplyr)
```


    Attaching package: 'dplyr'

    The following objects are masked from 'package:stats':

        filter, lag

    The following objects are masked from 'package:base':

        intersect, setdiff, setequal, union

``` r
library(ggplot2)
# library(openmeteo)

# via perplexity
carbon_data <- dput(data.frame(
  year = 2001:2025,
  co2 = c(371.13, 373.22, 375.77, 377.49, 379.80, 381.90, 383.76, 385.59, 387.37, 389.85, 
          391.63, 393.82, 396.48, 398.65, 400.83, 404.24, 406.55, 408.52, 411.44, 414.24, 
          416.45, 418.56, 419.30, 424.61, 426.50)
))
```

    structure(list(year = 2001:2025, co2 = c(371.13, 373.22, 375.77, 
    377.49, 379.8, 381.9, 383.76, 385.59, 387.37, 389.85, 391.63, 
    393.82, 396.48, 398.65, 400.83, 404.24, 406.55, 408.52, 411.44, 
    414.24, 416.45, 418.56, 419.3, 424.61, 426.5)), class = "data.frame", row.names = c(NA, 
    -25L))

``` r
nyc_info <- nyc_df |> head(1)

# weather_df <- nasapower::get_power(community = "ag", 
#                                    pars = c("T2M", "RH2M", "PRECTOTCORR"),
#                                    temporal_api = "hourly", 
#                                    lonlat = c(nyc_info$long, nyc_info$lat),
#                                    dates = c("2001-01-01", "2024-04-10")
#                                    )
```

### FE EDA

``` r
# 
# february_metrics  <- weather_df |>
#                      dplyr::filter(MO == 2) |>
#                      dplyr::group_by(YEAR) |>
#                      dplyr::summarize(
#                        total_feb_precip = sum(PRECTOTCORR),
#                        total_feb_rh     = sum(RH2M),
#                        maxim_feb_temp   = max(T2M)
#                      ) |>
#                       janitor::clean_names()
# 
# last_freeze_df <- 
#           weather_df |>
#               dplyr::mutate(
#                   freeze = ifelse(T2M < 0, 1, 0)
#               ) |>
#               dplyr::filter(MO <= 3) |>
#               dplyr::group_by(YEAR) |> 
#               dplyr::filter(freeze == 1) |> 
#               slice_tail(n = 1) |>
#               dplyr::mutate(
#                 freeze_date = lubridate::make_date(YEAR, MO, DY)
#               ) |>
#             janitor::clean_names() |>
#             dplyr::select(year, freeze_date)
# 
# 
# analysis_df <-
# nyc_df |>
#   dplyr::mutate(
#     bloom_date = as.Date(bloom_date)
#   ) |>
#   dplyr::left_join(february_metrics, by = c('year')) |>
#   dplyr::left_join(last_freeze_df, by = c('year')) |>
#   dplyr::select(-c(lat, long, alt, location)) |> 
#   na.omit() |>
#   dplyr::mutate(
#     distance_from_freeze = (as.numeric(bloom_date - freeze_date))
#   ) |>
#   dplyr::select(-bloom_date, -freeze_date)
# 
# analysis_df |> 
#   gt::gt() |>
#   gt::as_raw_html()
```

##### elote method

data-poor locations will get a simple ensemble. NYC will be similar to
dc but a bit further north, will ensemble our locations (w.o kyoto).

``` r
# get lat, long, altitude information --------------------------------------------
liestal_info <- readr::read_csv("../liestal/liestal_info.csv") |> select(location:alt)
```

    Rows: 1 Columns: 7
    ── Column specification ────────────────────────────────────────────────────────
    Delimiter: ","
    chr  (1): location
    dbl  (5): lat, long, alt, year, bloom_doy
    date (1): bloom_date

    ℹ Use `spec()` to retrieve the full column specification for this data.
    ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
washdc_info  <- readr::read_csv("../washington_dc/washingtondc_info.csv") |> select(location:alt) |> 
                mutate(alt = alt + 1)
```

    Rows: 1 Columns: 7
    ── Column specification ────────────────────────────────────────────────────────
    Delimiter: ","
    chr  (1): location
    dbl  (5): lat, long, alt, year, bloom_doy
    date (1): bloom_date

    ℹ Use `spec()` to retrieve the full column specification for this data.
    ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
geo_info <- nyc_info |> 
            select(location:alt) |> 
            bind_rows(liestal_info, washdc_info)

# create scalars ----------------------------------------------------------------
nyc_lat <- geo_info$lat[geo_info$location == "newyorkcity"]

lat_weights <- nyc_lat / geo_info$lat[2:3]

total_weights <- sum(lat_weights)

# get preds and weight them ----------------------------------------------------
liestal_pred <- readr::read_csv("../liestal/liestal_preds.csv") |>
  mutate(
    across(where(is.numeric), ~ . * lat_weights[1])
  )
```

    Rows: 1 Columns: 4
    ── Column specification ────────────────────────────────────────────────────────
    Delimiter: ","
    chr (1): location
    dbl (3): prediction, lower, upper

    ℹ Use `spec()` to retrieve the full column specification for this data.
    ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
washdc_pred  <- readr::read_csv("../washington_dc/washingtondc_preds.csv") |>
  mutate(
    across(where(is.numeric), ~ . * lat_weights[2])
  )           
```

    Rows: 1 Columns: 4
    ── Column specification ────────────────────────────────────────────────────────
    Delimiter: ","
    chr (1): location
    dbl (3): prediction, lower, upper

    ℹ Use `spec()` to retrieve the full column specification for this data.
    ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
# ensemble & bias  - 
# 2024 - vancouver-to-nyc 
# had a 5 day spread (nyc behind vancouver)
nyc_preds <- liestal_pred |> 
            rbind(washdc_pred) |> 
            summarize_if(is.numeric, sum) |>
    mutate(
    across(where(is.numeric), ~ . +2.5),
    across(where(is.numeric), ~ . / total_weights)
  ) |>
  mutate(location = "newyorkcity")

readr::write_csv(nyc_preds, "nyc_preds.csv")

nyc_preds |>
  gt::gt() |>
  gt::as_raw_html()
```

<div id="tcwriyrdas" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
  &#10;  

| prediction |    lower |    upper | location    |
|-----------:|---------:|---------:|:------------|
|   89.37484 | 76.21847 | 103.8159 | newyorkcity |

</div>
