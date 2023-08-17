
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.api import Holt,SimpleExpSmoothing,ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from matplotlib.ticker import MaxNLocator
import pandas as pd

def autoArima(df_test, test_train_ratio = 0.9, ticker="", base_price="Close"):
  df_test = df_test[df_test["Ticker"]==ticker]
  per_cols = ['Close', 'High', 'Low', 'Open']
  for col in per_cols:
      df_test[f'{col}_Per'] = df_test[col].pct_change(1)*100
  df_test = df_test.set_index('Date')
  df_test.dropna(inplace=True)
  df_test = df_test[[f"{base_price}_Per"]]
  train_data, test_data = df_test[3:int(len(df_test)*test_train_ratio)], df_test[int(len(df_test)*test_train_ratio):]
  model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
                        test='adf',       # use adftest to find optimal 'd'
                        max_p=3, max_q=3, # maximum p and q
                        m=1,              # frequency of series
                        d=None,           # let model determine 'd'
                        seasonal=False,   # No Seasonality
                        start_P=0,
                        D=0,
                        trace=True,
                        error_action='ignore',
                        suppress_warnings=True,
                        stepwise=True)
  print(model_autoARIMA.summary())
  model_autoARIMA.plot_diagnostics(figsize=(15,8))
  plt.show()
  model = ARIMA(train_data, order=(model_autoARIMA.order[0],model_autoARIMA.order[1],model_autoARIMA.order[2]))
  fitted = model.fit()
  forecast_values = fitted.get_forecast(steps=321, alpha=0.05)
  fc = forecast_values.predicted_mean
  se = forecast_values.se_mean

  n_len = len(test_data.index)

  lower_series = forecast_values.conf_int()[f"lower {base_price}_Per"][:n_len]
  upper_series = forecast_values.conf_int()[f"upper {base_price}_Per"][:n_len]

  #fc_series = pd.Series(fc, index=test_data.index)
  fc_series_2 = fc[:n_len]
  fc_series_2.index = test_data.index

  se_series = pd.Series(se, index=test_data.index)
  lower_series.index = test_data.index
  upper_series.index = test_data.index

  # Plot
  plt.figure(figsize=(10,5), dpi=100)
  plt.plot(train_data, label='training data')
  plt.plot(test_data, color = 'blue', label='Actual Stock Price')
  plt.plot(fc_series_2, color = 'orange',label='Predicted Stock Price')
  plt.fill_between(lower_series.index, lower_series, upper_series,
                  color='k', alpha=.10)
  plt.title(f'{ticker} Stock Price Prediction')
  plt.xlabel('Time')
  plt.ylabel(f'{ticker} Stock Price')
  plt.legend(loc='upper left', fontsize=8)

  plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=30))
  plt.xticks(rotation=90)

  plt.show()
