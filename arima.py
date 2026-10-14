import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kaggle_install import path
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

csv_path = path + f"\\b2c_sales_data.csv"
data_frame = pd.read_csv(csv_path)

data_frame['Order_Date'] = pd.to_datetime(data_frame['Order_Date'], format='%Y-%m-%d', utc=True)
df = data_frame.set_index('Order_Date').sort_index()

q_ts = df.Quantity.resample('W').sum()
q_ts = q_ts[ (q_ts.index > '2021-01-03') & (q_ts.index < '2023-12-25') ]

def check_stationarity(series):
    # Copied from https://machinelearningmastery.com/time-series-data-stationary-python/

    result = adfuller(series.values, result_object=False)

    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    if (result[1] <= 0.05) & (result[4]['5%'] > result[0]):
        print("\u001b[32mStationary\u001b[0m")
    else:
        print("\x1b[31mNon-stationary\x1b[0m")

def fetch_auto_correlation_lags(
        series,
        output_path:str|None = None,
        lag_acf:int = 15,
        lag_pacf:int = 15,
        alpha:float=0.05
) -> tuple[int,int]:
    if not 0 < alpha < 1:
        raise ValueError('alpha must be between 0 and 1')

    height = 4
    width = 12
    f, ax = plt.subplots(nrows=2, ncols=1, figsize=(width, 2 * height))
    plot_acf(series, lags=lag_acf, ax=ax[0], alpha=alpha)
    plot_pacf(series, lags=lag_pacf, ax=ax[1], method='ols', alpha=alpha)
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close()
    acf_result = acf(series, nlags=lag_acf, alpha=alpha, result_object=True)
    pacf_result = pacf(series, nlags=lag_pacf, alpha=alpha, result_object=True, method='ols')
    print('ACF Result:')
    print(acf_result)
    print('PACF Result:')
    print(pacf_result)
    highest_acf_lag = 0
    highest_pacf_lag = 0
    for i in range(len(acf_result.acf)):
        acf_interval = acf_result.confint[i]
        if acf_interval[0] > 0 or acf_interval[1] < 0:
            highest_acf_lag = i
        pacf_interval = pacf_result.confint[i]
        if pacf_interval[0] > 0 or pacf_interval[1] < 0:
            highest_pacf_lag = i
    return highest_acf_lag, highest_pacf_lag

def graph_time_series_with_rolling_mean(series, rolling_window:int, output_path:str|None = None) -> None:
    rolling_mean = series.rolling(rolling_window).mean()
    plt.figure(figsize=(12, 6))
    plt.plot(series, label='Original Time Series')
    plt.plot(rolling_mean, label=f'{rolling_window}-Week Rolling Mean', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Sale Quantity')
    plt.title('Time Series Data with Rolling Mean')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close()

def autoreg_test(series, nlags:int, train_pct:float=0.8) -> None:
    if train_pct < 0 or train_pct > 1:
        raise ValueError('train_len must be between 0 and 1')

    num_samples = len(series)
    train_len = int(num_samples * train_pct)
    train = series[:train_len]
    ar_model = AutoReg(train, lags=nlags).fit()

    print(ar_model.summary())
    pred = ar_model.predict(start=train_len, end=num_samples-1, dynamic=False)

    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))
    sns.lineplot(x=series.index[train_len:num_samples], y=series.values[train_len:num_samples], marker='o', label='test',
                 color='grey')
    sns.lineplot(x=series.index[:train_len], y=train, marker='o', label='train')
    sns.lineplot(x=series.index[train_len:num_samples], y=pred, marker='o', label='pred')
    ax.set_xlim([series.index[0], series.index[-1]])
    ax.set_title(f'Auto Regression Model (lags = {nlags})')
    plt.tight_layout()
    plt.show()

def arima_test(series,
               order:tuple[int,int,int],
               train_pct:float = 0.8,
               output_path:str|None = None
) -> None:
    if train_pct < 0 or train_pct > 1:
        raise ValueError('train_len must be between 0 and 1')

    num_samples = len(series)
    train_len = int(num_samples * train_pct)
    train = series[:train_len]
    arima_model = ARIMA(train, order=order).fit()

    print(arima_model.summary())
    pred = arima_model.predict(start=train_len, end=num_samples-1, dynamic=False)

    actual = np.array(series.values[train_len:num_samples], dtype=float)
    predicted = np.array(pred, dtype=float)
    model_sse = np.sum((actual - predicted) ** 2)
    mean_baseline_sse = np.sum((actual - train.mean()) ** 2)
    oos_r2 = 1 - model_sse / mean_baseline_sse
    print(f"Out-of-sample R²: {oos_r2:.3f}")

    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))
    sns.lineplot(x=series.index[train_len:num_samples], y=series.values[train_len:num_samples], marker='o', label='test',
                 color='grey')
    sns.lineplot(x=series.index[:train_len], y=train, marker='o', label='train')
    sns.lineplot(x=series.index[train_len:num_samples], y=pred, marker='o', label='pred')
    ax.set_xlim([series.index[0], series.index[-1]])
    ax.set_title(f"ARIMA Model {order}")
    metrics_text = (
        f"AIC: {arima_model.aic:.2f}\n"
        f"OOS $R^2$: {oos_r2:.3f}"
    )

    ax.text(
        0.02,
        0.95,
        metrics_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        fontsize=11,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "grey",
            "alpha": 0.85
        }
    )
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close()

graph_time_series_with_rolling_mean(q_ts, rolling_window=13, output_path=f'media/time_series_w_rolling_mean.jpg')
check_stationarity(q_ts)
autoreg_test(q_ts, nlags=1)
p, q = fetch_auto_correlation_lags(q_ts, alpha=0.05, output_path='media/autocorrelation_lags.jpg')
print("p = ", p)
print("q = ", q)
d = 0
arima_test(q_ts, order=(p,d,q), output_path=f'media/arima_model_{p}_{d}_{q}.jpg')
arima_test(q_ts, order=(2,d,q), output_path=f'media/arima_model_{2}_{d}_{q}.jpg')
arima_test(q_ts, order=(10,d,q), output_path=f'media/arima_model_{10}_{d}_{q}.jpg')
arima_test(q_ts, order=(p,d,2), output_path=f'media/arima_model_{p}_{d}_{2}.jpg')
arima_test(q_ts, order=(2,d,2), output_path=f'media/arima_model_{2}_{d}_{2}.jpg')
arima_test(q_ts, order=(10,d,2), output_path=f'media/arima_model_{10}_{d}_{2}.jpg')
arima_test(q_ts, order=(p,d,0), output_path=f'media/arima_model_{p}_{d}_{0}.jpg')
arima_test(q_ts, order=(2,d,0), output_path=f'media/arima_model_{2}_{d}_{0}.jpg')
arima_test(q_ts, order=(10,d,0), output_path=f'media/arima_model_{10}_{d}_{0}.jpg')

print("mean: ", q_ts.mean())
print("std: ", q_ts.std())
