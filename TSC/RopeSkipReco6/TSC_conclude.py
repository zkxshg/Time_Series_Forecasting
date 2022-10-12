import utils.TSC_utils as uil
import utils.TSC_rope as rope

import numpy as np
import pandas as pd

from dtw import dtw
from numpy.linalg import norm
from numpy import array

# 跳绳
def count_rope(refer, ref_frame, aim, aim_frame):
    cou_rope = rope.TSC(refer, ref_frame, aim, aim_frame)
     # 读入待识别文件
    # df = pd.read_csv(aim, encoding='mbcs')
    df = pd.read_csv(aim)
    # 抽样为30帧
    # df = cou_rope.sample_30(df, aim_frame)
    # 读取关节点
    joins = cou_rope.readJoin(df)
    # 计算跳跃次数
    minJump = len(cou_rope.countPeakByJoin(joins, 1))
    # 计算跳跃次数
    for i in range(2, 17):
        parr = cou_rope.countPeakByJoin(joins, i)
        # print("Join-",i,": 共跳了", len(parr), "次")
        minJump = min(minJump, len(parr))
    print("共跳了", len(parr), "次")
    
def err_rope(refer, ref_frame, aim, aim_frame):
    cou_rope = rope.TSC(refer, ref_frame, aim, aim_frame)
    cou_rope.count_good()

def count_peak(aim, aim_frame, j0, j1):
    # 读入待识别文件
    df = pd.read_csv(aim)
    # 抽样为30帧
    df = uil.sample_30(df, aim_frame)
    # 读取关节点
    joins = uil.readJoin(df)
    # 计算跳跃次数
    parr  = uil.countPeakByJoin(joins, j0)
    parr2 = uil.countPeakByJoin(joins, j1)

    # 提取跳跃运动特征
    feature = uil.getSeq(df, parr)

    # 输出识别结果
    result = int((len(parr) + len(parr2)) / 2)
    return result

def count_err(rpath, ref_frame, aim, aim_frame, j0, j1):
    # 读入待识别文件
    df = pd.read_csv(aim)
    # 抽样为30帧
    df = uil.sample_30(df, aim_frame)
    # 读取关节点
    joins = uil.readJoin(df)
    
    # 计算跳跃次数
    parr = uil.countPeakByJoin(joins, j0)
    
    # 提取跳跃运动特征
    feature = uil.getSeq(df, parr)
    
    # 地日参考序列
    st = uil.readRef('standard1.csv', rpath, ref_frame, j0)
    e1 = uil.readRef('err1.csv', rpath, ref_frame, j0)

    # 计算错误个数
    a1 = uil.ifErr1(feature, st, e1, j0, j1)

    # 计算正确个数
    good = len(parr) - int(a1)
    # 输出识别结果
    print("共有", a1, "处错误")

# 仰卧起坐
def count_Situp(aim, aim_frame):
    print("共做了", count_peak(aim, aim_frame * 20, 5, 6), "次仰卧起坐")

def err_Situp(rpath, ref_frame, aim, aim_frame):
    count_err(rpath, ref_frame, aim, aim_frame * 20, 5, 6)

# 高抬腿
def count_HLU(aim, aim_frame):
    num = count_peak(aim, aim_frame, 13, 14)
    print("共做了",num , "次高抬腿")
    return num
    
def err_HLU(rpath, ref_frame, aim, aim_frame):
    count_err(rpath, ref_frame, aim, aim_frame, 5, 6)

# 引体向上
def count_Pullup(aim, aim_frame):
    num = count_peak(aim, aim_frame * 8, 5, 6)
    print("共做了",num , "次引体向上")
    return num
    
def err_Pullup(rpath, ref_frame, aim, aim_frame):
    count_err(rpath, ref_frame, aim, aim_frame * 8, 5, 6)

# 斜身引体向上
def count_SlantPullup(aim, aim_frame):
    num = count_peak(aim, aim_frame * 8, 5, 6)
    print("共做了",num , "次斜身引体向上")
    return num
    
def err_SlantPullup(rpath, ref_frame, aim, aim_frame):
    count_err(rpath, ref_frame, aim, aim_frame * 8, 5, 6)

# 立定跳远
def count_StandJump(aim, aim_frame):
    num = count_peak(aim, aim_frame, 7, 8)
    print("共做了",num , "次立定跳远")
    return num
    
def err_StandJump(rpath, ref_frame, aim, aim_frame):
    count_err(rpath, ref_frame, aim, aim_frame, 7, 8)

# 抛铅球
def count_Toss(aim, aim_frame):
    num = count_peak(aim, aim_frame * 10, 9, 10)
    print("共抛了",num , "次铅球")
    return num
    
def err_Toss(rpath, ref_frame, aim, aim_frame):
    count_err(rpath, ref_frame, aim, aim_frame, 9, 10)
    
# 平板支撑
def err_Plank(rpath, ref_frame, aim, aim_frame):
    j0 = 11
    j1 = 12
    
    # 读入待识别文件
    df = pd.read_csv(aim)
    # 抽样为30帧
    df = uil.sample_30(df, aim_frame)
    # 读取关节点
    joins = uil.readJoin(df)
    
    # 地日参考序列
    sepath = rpath + 'standard1.csv'
    sedf = pd.read_csv(sepath)
    st = uil.sample_30(sedf, ref_frame)
    
    epath = rpath + 'err1.csv'
    edf = pd.read_csv(epath)
    e1 = uil.sample_30(edf, ref_frame)
    
    # 计算距离
    dist_true = 0
    for i in range(j0, j1 + 1):
        p1 = uil.str2num(df, df.columns[i])
        x1 = uil.normalize(p1.iloc[:,0]) 
        y1 = uil.normalize(p1.iloc[:,1]) 
        
        min_dist = 99999999
        tem_dist = 0
        p2 = uil.str2num(st, st.columns[i])
        x2 = uil.normalize(p2.iloc[:,0]) 
        y2 = uil.normalize(p2.iloc[:,1]) 
            
        tem_dist += uil.DTW(np.array(x1).reshape(-1, 1), np.array(x2).reshape(-1, 1))
        tem_dist += uil.DTW(np.array(y1).reshape(-1, 1), np.array(y2).reshape(-1, 1))
            
        if tem_dist < min_dist:
            min_dist = tem_dist
        
        dist_true = max(min_dist, dist_true)
    
    # dist_false = uil.calErr1(df, st,j0,j1)
    dist_false = 0
    for i in range(j0, j1 + 1):
        p1 = uil.str2num(df, df.columns[i])
        x1 = uil.normalize(p1.iloc[:,0]) 
        y1 = uil.normalize(p1.iloc[:,1]) 
        
        min_dist = 99999999
        tem_dist = 0
        p2 = uil.str2num(e1, e1.columns[i])
        x2 = uil.normalize(p2.iloc[:,0]) 
        y2 = uil.normalize(p2.iloc[:,1]) 
            
        tem_dist += uil.DTW(np.array(x1).reshape(-1, 1), np.array(x2).reshape(-1, 1))
        tem_dist += uil.DTW(np.array(y1).reshape(-1, 1), np.array(y2).reshape(-1, 1))
            
        if tem_dist < min_dist:
            min_dist = tem_dist
        
        dist_false = max(min_dist, dist_false)
    
    if dist_true >= dist_false:
        print("平板支撑姿势有误")
    else:
        print("平板支撑姿势正确")