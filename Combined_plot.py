import pandas as pd
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import numpy as np
import pmdarima as pm  # Auto ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing  # SES
from statsmodels.tsa.holtwinters import ExponentialSmoothing  # State space model ETS
from fbprophet import Prophet  # Prophet


def prophet(data, Pre_day):
    # 添加假日數據：
    NewYear = pd.DataFrame({  # 元旦
        'holiday': 'NewYear',
        'ds': pd.to_datetime(['2014-01-01','2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01']),
        'lower_window': -2,
        'upper_window': 2,
    })
    SpringFestival = pd.DataFrame({  # 春節
        'holiday': 'SpringFestival',
        'ds': pd.to_datetime(['2014-01-31','2015-02-19', '2016-02-08', '2017-01-28', '2018-02-16']),
        'lower_window': -2,
        'upper_window': 4,
    })
    LaborDay = pd.DataFrame({  # 勞動節
        'holiday': 'LaborDay',
        'ds': pd.to_datetime(['2014-05-01', '2015-05-01', '2016-05-01', '2017-05-01', '2018-05-01']),
        'lower_window': -2,
        'upper_window': 2,
    })
    NationalDay = pd.DataFrame({  # 國慶節
        'holiday': 'NationalDay',
        'ds': pd.to_datetime(['2014-10-01', '2015-10-01', '2016-10-01', '2017-10-01', '2018-10-01']),
        'lower_window': -1,
        'upper_window': 3,
    })
    holidays = pd.concat((NewYear, SpringFestival, LaborDay, NationalDay))  # 結合為節假日因子
    m = Prophet(holidays=holidays)
    m.fit(data)
    future = m.make_future_dataframe(periods=Pre_day)
    forecast = m.predict(future)
    print("======================prophet========================")
    # print(forecast)
    # register_matplotlib_converters()
    # fig1 = m.plot(forecast)
    # fig1.show()
    result = forecast['yhat']
    # result = result.iloc[-Pre_day:]
    # print(result)
    return result


def autoarima(data, Pre_day):
    stepwise_fit = pm.auto_arima(data, m=12, seasonal=True, error_action='ignore', suppress_warnings=True,
                                 stepwise=True)
    print("======================autoarima========================")
    # result =stepwise_fit.predict(n_periods=Pre_day)
    result = stepwise_fit.predict(n_periods=data.shape[0])
    # 繪圖
    '''
    x = np.arange(data.shape[0])
    plt.scatter(x, data, marker='x')
    plt.plot(x, stepwise_fit.predict(n_periods=data.shape[0]))
    plt.title('Actual test samples vs. forecasts')
    plt.show()
    '''
    return result


def SES(data, Pre_day):
    model = SimpleExpSmoothing(data)
    model_fit = model.fit()
    # yhat = model_fit.predict(row_num - Pre_day, row_num -1)
    yhat = model_fit.predict(0, row_num - Pre_day-1)
    print("======================SES========================")
    '''
    x = np.arange(data.shape[0])
    plt.scatter(x, data, marker='x')
    plt.plot(x, model_fit.predict(0, row_num - Pre_day-1))
    plt.title('Actual test samples vs. forecasts')
    plt.show()
    '''
    result = yhat
    return result


def ETS(data, Pre_day):
    # fit model
    model = ExponentialSmoothing(data)
    model_fit = model.fit()
    # make prediction
    # yhat = model_fit.predict(row_num - Pre_day, row_num -1)
    yhat = model_fit.predict(0, row_num - Pre_day - 1)
    print("======================ETS========================")
    '''
    x = np.arange(data.shape[0])
    plt.scatter(x, data, marker='x')
    plt.plot(x, model_fit.predict(0, row_num - Pre_day - 1))
    plt.title('Actual test samples vs. forecasts')
    plt.show()
    '''
    result = yhat
    return result


def cal_err(data_real, data_pre):
    error = 0
    for i in range(0, Pre_num):
        error += (data_pre.loc[row_num -Pre_num + i] - data_real.loc[row_num -Pre_num + i]) ** 2
        print("==================")
        print(data_pre.loc[row_num -Pre_num + i])
        print(error)
    error = np.sqrt(error).astype(float)
    return error


def cal_err2(data_real, data_pre):
    error = 0
    for i in range(0, Pre_num):
        error += (data_pre[i] - data_real.loc[row_num -Pre_num + i]) ** 2
        # print("==================")
        # print(error)
    error = np.sqrt(error).astype(float)
    return error


def cal_err3(data_real, data_pre):
    error = 0
    len = data_real.shape[0]
    for i in range(0, len):
        error += (data_pre.loc[i] - data_real.loc[i]) ** 2
        # print("==================")
        # print(data_pre.loc[i])
        # print(error)
    error = np.sqrt(error).astype(float)
    return error


def cal_err4(data_real, data_pre):
    error = 0
    len = data_real.shape[0]
    for i in range(0, len):
        error += (data_pre[i] - data_real.loc[i]) ** 2
        # print("==================")
        # print(error)
    error = np.sqrt(error).astype(float)
    return error

if __name__=='__main__':
    # 讀入數據
    df = pd.read_csv('class1.csv')
    df = df.fillna(df.bfill())
    df_y = df['y']
    row_num = df.shape[0]
    Pre_num = 12
    error_of_prophet = 0
    data_train = df.iloc[:-Pre_num]
    data_tr_y = data_train['y']
    data_test = df.iloc[-Pre_num:]
    data_test = data_test['y']
    data_pre = df.iloc[:-1]
    data_pre_y = data_pre['y']
    # Prophet
    P_prophet = prophet(data_train, Pre_num)
    # err_pro = cal_err(data_test, P_prophet)
    print("============plot==============")
    # print(data_train.shape[0])
    # print(P_prophet.shape[0])
    err_pro = cal_err3(data_tr_y, P_prophet.iloc[:-Pre_num])

    # auto_arima
    P_auar = autoarima(data_tr_y, Pre_num)
    # err_auar = cal_err2(data_test, P_auar)
    err_auar = cal_err4(data_tr_y, P_auar)

    # SES
    P_SES = SES(data_tr_y, Pre_num)
    # err_SES = cal_err(data_test, P_SES)
    err_SES = cal_err3(data_tr_y, P_SES)

    # ETS
    P_ETS = ETS(data_tr_y, Pre_num)
    # err_ETS = cal_err(data_test, P_ETS)
    err_ETS = cal_err3(data_tr_y, P_ETS)

    # 計算權重
    weight = [0, 0, 0, 0]
    error_all = [err_pro, err_auar, err_SES, err_ETS]
    Total_error = err_pro + err_auar + err_SES + err_ETS
    for i in range(0, 4):
        weight[i] = error_all[i] / Total_error

    print(weight)
    # 預測結果
    '''
    Pre_value = 0
    Pre_value += weight[0] * prophet(data_pre, 1)
    Pre_value += weight[1] * autoarima(data_pre_y, 1)
    Pre_value += weight[2] * SES(data_pre_y, 1)
    Pre_value += weight[3] * ETS(data_pre_y, 1)
    print(Pre_value)
    '''
    Pre_value = weight[0] * P_prophet
    Pre_value = Pre_value.iloc[:-Pre_num]
    Pre_value += weight[1] * P_auar
    Pre_value += weight[2] * P_SES
    Pre_value += weight[3] * P_ETS
    
    x = np.arange(data_tr_y.shape[0])
    plt.scatter(x, data_tr_y, marker='x')
    plt.plot(x, Pre_value)
    plt.title('Actual test samples vs. forecasts')
    plt.show()
