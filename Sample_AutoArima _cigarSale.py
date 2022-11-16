# Weekly Auto-Arima for cigar sale data

import itertools
import pandas as pd
import numpy as np
import statsmodels as sm
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
import warnings

def data_preprocess(data_real):

    # 按星期划分
    Mon = data_real.loc[data_real['week'].str.contains('周一')]
    Tues = data_real.loc[data_real['week'].str.contains('周二')]
    Wed = data_real.loc[data_real['week'].str.contains('周三')]
    Thur = data_real.loc[data_real['week'].str.contains('周四')]
    Fri = data_real.loc[data_real['week'].str.contains('周五')]

    # 处理缺失值并重新排列
    Mon = Mon.dropna()
    Mon.reset_index(drop=True, inplace=True)
    Tues = Tues.dropna()
    Tues.reset_index(drop=True, inplace=True)
    Wed = Wed.dropna()
    Wed.reset_index(drop=True, inplace=True)
    Thur = Thur.dropna()
    Thur.reset_index(drop=True, inplace=True)
    Fri = Fri.dropna()
    Fri.reset_index(drop=True, inplace=True)

    # 预测周一至周五各日销量
    print("周一：")
    print(ts_para(Mon['num']))
    print("周二：")
    print(ts_para(Tues['num']))
    print("周三：")
    print(ts_para(Wed['num']))
    print("周四：")
    print(ts_para(Thur['num']))
    print("周五：")
    print(ts_para(Fri['num']))
    return 1


def ts_para(data_per):
    # 读入数据
    data = data_per
    # print(data.head())

    # 生成待检验参数组合
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    pdq.remove((0, 0, 0))

    # specify to ignore warning messages
    warnings.filterwarnings("ignore")

    # 划分为 train_set 和 test_set
    data_train = data.iloc[:-2]
    data_test = data.iloc[-2:]
    # print(data_test)
    tup_num = data_train.shape[0]

    # 判断是否为0向量
    sum_train = 0 
    for s in range(0, tup_num):
        sum_train += data_train.loc[s].astype(float)
    if sum_train < 0.1:
        pre_value = 0
    else:
        # 找出预测误差最小的参数
        min_error = 999
        min_para = []
        min_pre = 0
        
        # 由於有些參數可能無法正常建立模型，故需try catch處理
        for param in pdq:
            try:
                model = sm.tsa.arima_model.ARIMA(data_train, param)
                results = model.fit(disp=0)
                # pred = results.get_prediction(start=tup_num, end=tup_num+1)
                pred = results.predict(tup_num, tup_num + 2, dynamic=True)
                # print(pred)
                error = 0
                error += (pred.loc[tup_num] - data_test.loc[tup_num]) ** 2
                error += (pred.loc[tup_num + 1] - data_test.loc[tup_num + 1]) ** 2
                error = np.sqrt(error).astype(float)
                e = error
                if e < min_error:
                    min_error = e
                    min_para = param
                    min_pre = pred.loc[tup_num+2].astype(float)
            except:
                continue
        pre_value = min_pre
    return pre_value

# 读入数据
data_all = pd.read_excel('待处理_real.xlsx', skiprows=1 )
# print(data_all)
real_num = data_all.shape[1]

# 逐列进行预测
for i in range(1, real_num):
    data_t = data_all.iloc[:, i:i + 1]
    data_week = data_all.iloc[:, 0:1]
    data_wt = pd.concat([data_week, data_t], axis=1)
    data_wt.columns = ['week', 'num']
    data_pro = data_preprocess(data_wt)
