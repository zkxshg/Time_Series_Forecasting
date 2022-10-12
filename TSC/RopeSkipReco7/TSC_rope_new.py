#!/usr/bin/env python
# coding: utf-8
import utils.TSC_utils as uil
import utils.TSC_rope as rope

import numpy as np
import pandas as pd

from dtw import dtw
from numpy.linalg import norm
from numpy import array

import os

def str2num(df, ind):
    if df[ind][0][0] == '[' :
        ser_x = df[ind].apply(lambda x : float(x.split(',')[0][1 : -1]))
        ser_y = df[ind].apply(lambda x : float(x.split(',')[1][1 : -2]))
    else :
        ser_x = df[ind].apply(lambda x : float(x.split(',')[0][0 : -1]))
        ser_y = df[ind].apply(lambda x : float(x.split(',')[1][0 : -2]))
    
    pair = pd.concat([ser_x, ser_y], axis=1)
    pair.columns = ['x' + ind, 'y' + ind]
    pair['date'] = pair.index
    return pair


# 读取17个关节点
def readJoin(df):
    # inds = df.columns[:17]
    inds = df.columns[:16]
    
    joins = []
    for ind in inds:
        joins.append(str2num(df, ind))
    return joins


# 根据前后判断是否为高峰
def isPeak(arr, pos):
    length = arr.shape[0]
    # print("pos = ", pos)
    back = pos - 1
    while back >= 0:
        if arr[back] < arr[pos]:
            return False
        elif arr[back] > arr[pos]:
            break
        back -= 1
            
    forw = pos + 1
    while forw < length:
        if arr[forw] < arr[pos]:
            return False
        elif arr[forw] > arr[pos]:
            break
        forw += 1
    
    # print("arr[", pos, "] = " , arr[pos])
    return True

# 根据指定关节计算跳跃个数
def countPeakByJoin(df, pos):
    arr = df[pos].iloc[:,1] 
    # print(arr)
    
    peak_arr = []
    left_df = arr
    
    length = left_df.shape[0]
    for i in range(1, length - 1):
        if left_df[i] == left_df[i - 1]:
            continue
            
        if (isPeak(left_df, i)):
            if len(peak_arr) < 1:
                peak_arr.append(i)
            elif (i - peak_arr[-1] >= 9):
                peak_arr.append(i)
    
    return peak_arr

def seqPlot(filepath, pos):
    df = pd.read_csv(filepath)
    df['date'] = df.index
    p2 = str2num(df, df.columns[pos])
    ggplot(aes(x="date", y="y"+str(pos)), data=p2) + geom_point() + geom_line(color = 'blue') + scale_x_date(labels = date_format("%Y-%m-%d"))


def testFile(path):
    filePath = path
  
    fileNames = os.listdir(filePath)
    print(fileNames)

    # ========== 2. 筛选序列文件 ==========
    picNames = []
    for file in fileNames:
        # print(file)
        if len(file.split('.')) < 2:
            continue

        ftype = file.split('.')[1]
        if ftype == 'csv':
            picNames.append(file)

    print("待识别序列包括：", picNames)
    
    for aim in picNames:
        df = pd.read_csv(path + aim)
        df['date'] = df.index
        joins = readJoin(df)
        
        parr = countPeakByJoin(joins, 14)
        print(aim, "共跳了", len(parr), "次")



aim = "err/f19d627296.csv" 
# aim = "test/220406/67times8.csv" 

df = pd.read_csv(aim)
df['date'] = df.index
joins = readJoin(df)
parr = countPeakByJoin(joins, 15)
print("共跳了", len(parr), "次")



# testFile("test/220406/")

# from ggplot import *
# filepath = "test/220409/144times17.csv"
# pos = 15

# df = pd.read_csv(filepath)
# df['date'] = df.index
# p2 = str2num(df, df.columns[pos])
# ggplot(aes(x="date", y="y"+str(pos)), data=p2) + geom_point() + geom_line(color = 'blue') + scale_x_date(labels = date_format("%Y-%m-%d"))




