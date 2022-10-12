import numpy as np
import pandas as pd

from dtw import dtw
from numpy.linalg import norm
from numpy import array


class TSC:
    def __init__(self, refer, ref_frame, aim, aim_frame):
        self.aim_frame = aim_frame * 8
        self.ref_frame = ref_frame
        self.rpath = refer
        self.aim = aim
        self.err_set = []

    # 计算时间序列距离
    def DTW(self, x, y):
        dist, cost, acc, path = dtw(x, y, dist=lambda x, y: norm(x - y, ord=1))
        return dist

    # dataframe标准化
    def normalize(self, df):
        df = (df - df.mean()) / (df.std())
        return df

    # 字段转为 float
    def str2num(self, df, ind):
        ser_x = df[ind].apply(lambda x: float(x.split(',')[0][1: -1]))
        ser_y = df[ind].apply(lambda x: float(x.split(',')[1][1: -2]))
        pair = pd.concat([ser_x, ser_y], axis=1)
        pair.columns = ['x' + ind, 'y' + ind]
        pair['date'] = pair.index
        return pair

    # 读取17个关节点
    def readJoin(self, df):
        inds = df.columns[:17]
        joins = []

        for ind in inds:
            joins.append(self.str2num(df, ind))

        return joins

    # 统一抽样为：30帧/秒
    def sample_30(self, df, ori_frame):
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
        df_30.index = df.index[0: df_30.shape[0]]

        return df_30

    # 根据前后判断是否为高峰
    def isPeak(self, arr, pos):
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

    # 提取特征序列: 17 * 2 * fre
    def getSeq_17(self, df, arr):
        fea = []

        for ind in df:
            pair = []

            peak_str = []
            peak_str.append(ind.iloc[0: arr[1], 0].tolist())
            num = len(arr)
            for i in range(1, num - 1):
                peak_str.append(ind.iloc[arr[i - 1]: arr[i + 1] + 1, 0].tolist())
            pair.append(peak_str)

            y_str = []
            y_str.append(ind.iloc[0: arr[1], 1].tolist())
            num = len(arr)
            for i in range(1, num - 1):
                y_str.append(ind.iloc[arr[i - 1]: arr[i + 1] + 1, 1].tolist())
            pair.append(peak_str)

            fea.append(pair)

        return fea

    # 提取特征序列
    def getSeq(self, df, arr):
        peak_str = []
        peak_str.append(df.iloc[0: arr[1]])

        num = len(arr)
        for i in range(1, num - 1):
            peak_str.append(df.iloc[arr[i - 1]: arr[i + 1] + 1])
        return peak_str

    # 计算跳跃个数
    def countPeak(self, df):
        arr = df[13].iloc[:, 0]

        peak_arr = []
        left_df = arr

        length = left_df.shape[0]
        for i in range(1, length - 1):
            if self.isPeak(left_df, i):
                if len(peak_arr) < 1:
                    peak_arr.append(i)
                elif i - peak_arr[-1] >= 9:
                    peak_arr.append(i)

        return peak_arr
    
    # 计算跳跃个数
    def countPeakByJoin(self, df, pos):
        arr = df[pos].iloc[:,0] 
    
        peak_arr = []
        left_df = arr
    
        length = left_df.shape[0]
        for i in range(1, length - 1):
            if (self.isPeak(left_df, i)):
                if len(peak_arr) < 1:
                    peak_arr.append(i)
                elif (i - peak_arr[-1] >= 9):
                    peak_arr.append(i)
    
        return peak_arr


    # 读入并抽样参考序列
    def readRef(self, path, rpath):
        epath = rpath + path
        edf = pd.read_csv(epath, encoding='mbcs')
        edf = self.sample_30(edf, 61)
        joins = self.readJoin(edf)
        parr = self.countPeak(joins)
        # feature = getSeq_17(joins, parr)
        feature = self.getSeq(edf, parr)[1:-2]
        return feature

    # 计算序列相似度
    def calDist(self, target, refer):
        dist = 0

        for i in range(0, 17):
            p1 = self.str2num(target, target.columns[i])
            x1 = self.normalize(p1.iloc[:, 0])
            y1 = self.normalize(p1.iloc[:, 1])

            p2 = self.str2num(refer, refer.columns[i])
            x2 = self.normalize(p2.iloc[:, 0])
            y2 = self.normalize(p2.iloc[:, 1])

            dist += self.TW(np.array(x1).reshape(-1, 1), np.array(x2).reshape(-1, 1))
            dist += self.DTW(np.array(y1).reshape(-1, 1), np.array(y2).reshape(-1, 1))

        return dist


    def count_peak(self):
         # 读入待识别文件
        df = pd.read_csv(self.aim)
        # 抽样为30帧
        df = self.sample_30(df, self.aim_frame)
        # 读取关节点
        joins = self.readJoin(df)
        # 计算跳跃次数
        parr = self.countPeakByJoin(joins, 5)  # 右肩
        parr2 = self.countPeakByJoin(joins, 6)  # 左肩
        # 提取跳跃运动特征
        feature = self.getSeq(df, parr)
    
        result = int((len(parr) + len(parr2)) / 2)

        # 输出识别结果
        print("共做了", result , "次引体向上。")
    