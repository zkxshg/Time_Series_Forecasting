#!/usr/bin/env python
# coding: utf-8

# In[1]:


import utils.TSC_utils as uil
import utils.TSC_rope as rope

import numpy as np
import pandas as pd

from dtw import dtw
from numpy.linalg import norm
from numpy import array


# In[99]:


def str2num(df, ind):
    ser_x = df[ind].apply(lambda x : float(x.split(',')[0][0 : -1]))
    ser_y = df[ind].apply(lambda x : float(x.split(',')[1][0 : -2]))
    pair = pd.concat([ser_x, ser_y], axis=1)
    pair.columns = ['x' + ind, 'y' + ind]
    pair['date'] = pair.index
    return pair


# In[100]:


# 读取17个关节点
def readJoin(df):
    inds = df.columns[:17]
    joins = []
    for ind in inds:
        joins.append(str2num(df, ind))
    return joins


# In[171]:


# 根据前后判断是否为高峰
def isPeak(arr, pos):
    length = arr.shape[0]
    
    back = pos - 1
    while back >= 0:
        if arr[back] < arr[pos]:
            return False
        elif arr[back] > arr[pos]:
            break
        back -= 1
            
    forw = pos + 1
    while forw <= length:
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
        if (isPeak(left_df, i)):
            if len(peak_arr) < 1:
                peak_arr.append(i)
            elif (i - peak_arr[-1] >= 9):
                peak_arr.append(i)
    
    return peak_arr


# In[195]:


aim = "test/324/125-1045.csv" 
df = pd.read_csv(aim)
df['date'] = df.index

joins = readJoin(df)
parr = countPeakByJoin(joins, 16)


# In[196]:


print("共跳了", len(parr), "次")


# In[ ]:




