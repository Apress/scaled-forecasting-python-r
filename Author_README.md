# Scaled Forecasting with Python
Builds the Forecaster object to create easily accessed forecasts for many time series

## Scenario
- Our company is wanting to expand into a new state
- As part of this decision, our role is to forecast the economic outlooks of all 50 states
- For each state, we will attempt to forecast a leading indicator, coincidental indicator, and unemployment rate (150 total time series of varying lengths, each with its own peculiarities)
- All data obtained from [FRED](fred.stlouisfed.org)

## Other uses
- In the real world you may use this approach to predict metrics specific to a company, such as daily sales or monthly web interactions
- This framework can be expanded to such use cases with minimal adjustments--instead of modeling based on your beliefs about a recession, you may add a vector of holidays or recurring dates that you believe affect the metrics you are attempting to forecast
- In this forecast, there isn't much consideration of seasonality (two of the models account for that automatically), but that is something you need to think about

## Installation (Windows Specific)
- Run all commands in requirements.txt on command line
- You may get an error when trying to install rpy2--This can be for a variety of reasons. Google is your friend!