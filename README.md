# Electricity Production Forecasting

**Course: Forecasting** | Johnny CHREIM

---

## Overview

This project forecasts monthly U.S. electricity production using a suite of time series models implemented in R. Multiple approaches are compared — ETS, ARIMA, TBATS, and Neural Network Autoregression (NNAR) — and the best-performing model is used to generate a 36-month forward forecast.

---

## Data

| Field | Detail |
|---|---|
| Source file | `Electric_Production.csv` |
| Series | `IPG2211A2N` — U.S. Electric Power Production Index |
| Frequency | Monthly |
| Start | January 1985 |
| Format | Date column (`MM/DD/YYYY`) + production index |

---

## Train / Test Split

| Split | Period | Share |
|---|---|---|
| Training | Jan 1985 – Dec 2009 | ~75% |
| Test | Jan 2010 – end of series | ~25% |

`tsclean()` is applied to the training set to remove outliers and fill missing values before modelling.

---

## Exploratory Analysis

The following plots are produced to understand the series before modelling:

- **Line plot** — overall trend and level
- **Seasonal plot** — year-over-year overlay by month
- **Monthplot (subseries plot)** — average seasonal pattern per month
- **ACF / PACF** — autocorrelation structure to guide ARIMA order selection

---

## Transformation

A **Box-Cox transformation** is applied to stabilise variance before fitting all models. The optimal lambda (λ) is estimated automatically via `BoxCox.lambda()`. All forecasts are back-transformed to the original scale using `InvBoxCox()` before evaluation.

---

## Models

### ETS (Exponential Smoothing State Space)

Three specifications are fitted and compared by AIC:

| Model | Error | Trend | Seasonality |
|---|---|---|---|
| `f1` | Additive | Additive | Additive |
| `f2` | Multiplicative | Additive | Additive |
| `f3` (auto) | Auto (`ZZZ`) | Auto | Auto |

The best model (`f3`, selected by AIC) is used for evaluation against the test set.

### ARIMA

ACF and PACF plots on the transformed training series guide manual order selection. Three models are fitted:

| Model | Order | Seasonal |
|---|---|---|
| `arima1` | (1,1,1) | (0,1,1) |
| `arima2` | (2,1,0) | (0,1,1) |
| `arima3` | auto via `auto.arima()` | auto |

All forecasts are back-transformed before computing test-set accuracy.

### TBATS

Handles complex seasonality, trend, and Box-Cox transformation internally. Fitted on the transformed training series; forecasts are back-transformed for evaluation.

### NNAR (Neural Network Autoregression)

`nnetar()` fits a feed-forward neural network with lagged inputs. Fitted on the transformed training series; forecasts are back-transformed for evaluation.

---

## Model Evaluation

Models are compared on the **test set** using `accuracy()`, which reports:

- **ME** — Mean Error
- **RMSE** — Root Mean Squared Error
- **MAE** — Mean Absolute Error
- **MAPE** — Mean Absolute Percentage Error
- **MASE** — Mean Absolute Scaled Error

Residual diagnostics are run on the two best models (ETS and auto-ARIMA) using `checkresiduals()` and the **Ljung-Box test** (lag = 10) to assess whether residuals resemble white noise.

---

## Final Forecast

The best ARIMA specification (`order = (2,1,0)`, `seasonal = (0,1,1)`) is refit on the **full dataset** (train + test) and used to generate a **36-month ahead forecast**.

---

## Output

| Output | Description |
|---|---|
| Forecast vs. test set plots | Visualises each model's predictions against actual test values |
| Accuracy tables | Train and test metrics for all models |
| Residual diagnostic plots | ACF of residuals + Ljung-Box p-value |
| Final 36-month forecast plot | Forward-looking projection from the full series |

---

## Requirements

- **R** (≥ 4.0)
- **Package:** `fpp2` (includes `forecast`, `ggplot2`, and supporting utilities)
- Input file `Electric_Production.csv` placed in the working directory
