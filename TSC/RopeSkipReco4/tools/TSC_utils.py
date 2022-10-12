import numpy as np
import pandas as pd
from dtw import dtw
from numpy.linalg import norm
from numpy import array

# 计算时间序列距离
def DTW(x, y):
    dist, cost, acc, path = dtw(x, y, dist=lambda x, y: norm(x - y, ord=1))
    return dist

# dataframe标准化
def normalize(df):
    df = (df - df.mean())/(df.std())  
    return df

# 字段转为 float
def str2num(df, ind):
    ser_x = df[ind].apply(lambda x : float(x.split(',')[0][1 : -1]))
    ser_y = df[ind].apply(lambda x : float(x.split(',')[1][1 : -2]))
    pair = pd.concat([ser_x, ser_y], axis=1)
    pair.columns = ['x' + ind, 'y' + ind]
    pair['date'] = pair.index
    return pair

# 读取17个关节点
def readJoin(df):
    inds = df.columns[:17]
    joins = []
    for ind in inds:
        joins.append(str2num(df, ind))
    return joins

# 统一抽样为：30帧/秒
def sample_30(df, ori_frame):
    rows = df.shape[0]
    ratio = 30.0 / ori_frame
    tem = 0
    inds = []
    for i in range(0, rows):
        tem += ratio
        if tem >= 1:
            inds.append(i)
            tem -= 1
    
    df_30 = df.iloc[inds]
    df_30.index = df.index[0 : df_30.shape[0]]
    return df_30

# 根据前后判断是否为高峰
def isPeak(arr, pos):
    length = arr.shape[0]
    
    back = pos - 1
    while back >= 0:
        if arr[back] > arr[pos]:
            return False
        elif arr[back] < arr[pos]:
            break
        back -= 1
            
    forw = pos + 1
    while forw <= length:
        if arr[forw] > arr[pos]:
            return False
        elif arr[forw] < arr[pos]:
            break
        forw += 1
    
    # print("arr[", pos, "] = " , arr[pos])
    return True

# 根据指定关节计算跳跃个数
def countPeakByJoin(df, pos):
    arr = df[pos].iloc[:,0] 
    
    peak_arr = []
    left_df = arr
    
    length = left_df.shape[0]
    for i in range(1, length - 1):
        if (isPeak(left_df, i)):
            if len(peak_arr) < 1:
                peak_arr.append(i)
            elif (i - peak_arr[-1] >= 9):
                peak_arr.append(i)
    
    return peak_arr

# 提取特征序列: 17 * 2 * fre
def getSeq_17(df, arr):
    fea = []
    
    for ind in df:
        pair = []
        
        peak_str = []
        peak_str.append(ind.iloc[0 : arr[1], 0].tolist())
        num = len(arr)
        for i in range(1, num - 1):
            peak_str.append(ind.iloc[arr[i - 1] : arr[i + 1] + 1, 0].tolist())    
        pair.append(peak_str)
        
        y_str = []
        y_str.append(ind.iloc[0 : arr[1], 1].tolist())
        num = len(arr)
        for i in range(1, num - 1):
            y_str.append(ind.iloc[arr[i - 1] : arr[i + 1] + 1, 1].tolist())
        pair.append(peak_str)
        
        fea.append(pair)
    
    return fea

# 提取特征序列
def getSeq(df, arr):
    peak_str = []
    peak_str.append(df.iloc[0 : arr[1]])
    
    num = len(arr)
    for i in range(1, num - 1):
        peak_str.append(df.iloc[arr[i - 1] : arr[i + 1] + 1])       
    return peak_str

# 读入并抽样参考序列
def readRef(path, rpath):
    epath  = rpath + path
    edf = pd.read_csv(epath,encoding='mbcs')
    edf = sample_30(edf, 61)
    joins = readJoin(edf)
    parr = countPeak(joins)
    # feature = getSeq_17(joins, parr)
    feature = getSeq(edf, parr)[1:-2]
    return feature

# 计算序列相似度
def calDist(target, refer):
    dist = 0
    
    for i in range(0, 17):
        p1 = str2num(target, target.columns[i])
        x1 = normalize(p1.iloc[:,0]) 
        y1 = normalize(p1.iloc[:,1]) 
        
        p2 = str2num(refer, refer.columns[i])
        x2 = normalize(p2.iloc[:,0]) 
        y2 = normalize(p2.iloc[:,1]) 
        
        dist += DTW(np.array(x1).reshape(-1, 1), np.array(x2).reshape(-1, 1))
        dist += DTW(np.array(y1).reshape(-1, 1), np.array(y2).reshape(-1, 1))
    
    return dist

# 计算序列相似度
def calErr1(target, refer, j0, j1):
    dist = 0
    
    for i in range(j0, j1):
        p1 = str2num(target, target.columns[i])
        x1 = normalize(p1.iloc[:,0]) 
        y1 = normalize(p1.iloc[:,1]) 
        
        min_dist = 99999999
        for err in refer:
            tem_dist = 0
            p2 = str2num(err, err.columns[i])
            x2 = normalize(p2.iloc[:,0]) 
            y2 = normalize(p2.iloc[:,1]) 
            
            tem_dist += DTW(np.array(x1).reshape(-1, 1), np.array(x2).reshape(-1, 1))
            tem_dist += DTW(np.array(y1).reshape(-1, 1), np.array(y2).reshape(-1, 1))
            
            if tem_dist < min_dist:
                min_dist = tem_dist
        
        dist = max(min_dist, dist)
    
    return dist

# 基于序列相似度判断错误次数
def ifErr1(feature, st, e1,j0,j1):
    cou = 0
    c = 0
    r = calErr1(feature[0], e1,j0,j1) / calErr1(feature[0], st,j0,j1)
    r += calErr1(feature[1], e1,j0,j1) / calErr1(feature[1], st,j0,j1)
    r += calErr1(feature[2], e1,j0,j1) / calErr1(feature[2], st,j0,j1)
    r += calErr1(feature[3], e1,j0,j1) / calErr1(feature[3], st,j0,j1)
    r /= 4
    r *= 0.9
    r = min(r, 0.9)

    err_set = []
    for seq in feature:
        c += 1
        st_dist = calErr1(seq, st)
        err_dist = calErr1(seq, e1)
        # print("st_dist = ", st_dist, ", err_dist = ", err_dist, "in c = ", c)
        if (not c in err_set) and err_dist < st_dist * r:
            cou += 1
            err_set.append(c)
    return cou 