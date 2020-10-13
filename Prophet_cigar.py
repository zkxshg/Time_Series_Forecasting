// FB Prophet for cigar sale

import pandas as pd
from fbprophet import Prophet
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import numpy as np

if __name__=='__main__':
    # 讀入數據
    df = pd.read_csv('season.csv')

    # 可選：進行log變換
    # df['y'] = np.log(df['y'])

    # 添加假日數據：
    # 元旦
    NewYear = pd.DataFrame({
        'holiday': 'NewYear',
        'ds': pd.to_datetime(['2014-01-01','2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01']),
        'lower_window': -2,
        'upper_window': 2,
    })
    # 春節
    SpringFestival = pd.DataFrame({
        'holiday': 'SpringFestival',
        'ds': pd.to_datetime(['2014-01-31','2015-02-19', '2016-02-08', '2017-01-28', '2018-02-16']),
        'lower_window': -2,
        'upper_window': 4,
    })
    # 勞動節
    LaborDay = pd.DataFrame({
        'holiday': 'NewYear',
        'ds': pd.to_datetime(['2014-05-01', '2015-05-01', '2016-05-01', '2017-05-01', '2018-05-01']),
        'lower_window': -2,
        'upper_window': 2,
    })
    # 國慶節
    NationalDay = pd.DataFrame({
        'holiday': 'NewYear',
        'ds': pd.to_datetime(['2014-10-01', '2015-10-01', '2016-10-01', '2017-10-01', '2018-10-01']),
        'lower_window': -1,
        'upper_window': 3,
    })
    # 結合為節假日因子
    holidays = pd.concat((NewYear, SpringFestival, LaborDay, NationalDay))

    # 創建並訓練模型
    m = Prophet(holidays=holidays)
    m.fit(df)

    # 構建未來一年日期
    future = m.make_future_dataframe(periods=365)
    # print(future.tail())

    # 預測未來一年銷售中的銷售量
    forecast = m.predict(future)
    # print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    # 繪製銷售折線圖
    register_matplotlib_converters()
    fig1 = m.plot(forecast)
    fig1.show()
    # 繪製因子折線圖
    fig2 = m.plot_components(forecast)
    fig2.show()
