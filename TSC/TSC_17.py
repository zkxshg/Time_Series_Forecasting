import numpy as np
import pandas as pd

from dtw import dtw
from numpy.linalg import norm
from numpy import array


class TSC:
    def __init__(self, refer, ref_frame, aim, aim_frame):
        self.aim_frame = aim_frame
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

    # 错误1：左右高低手
    # 基于左手、右手运动序列进行判断
    def calErr1(self, target, refer):
        dist = 0

        for i in range(9, 11):
            p1 = self.str2num(target, target.columns[i])
            x1 = self.normalize(p1.iloc[:, 0])
            y1 = self.normalize(p1.iloc[:, 1])

            min_dist = 99999999
            for err in refer:
                tem_dist = 0
                p2 = self.str2num(err, err.columns[i])
                x2 = self.normalize(p2.iloc[:, 0])
                y2 = self.normalize(p2.iloc[:, 1])

                tem_dist += self.DTW(np.array(x1).reshape(-1, 1), np.array(x2).reshape(-1, 1))
                tem_dist += self.DTW(np.array(y1).reshape(-1, 1), np.array(y2).reshape(-1, 1))

                if tem_dist < min_dist:
                    min_dist = tem_dist

            dist = max(min_dist, dist)

        return dist

    def ifErr1(self, feature, st, e1):
        cou = 0
        c = 0

        r = self.calErr1(feature[0], e1) / self.calErr1(feature[0], st)
        r += self.calErr1(feature[1], e1) / self.calErr1(feature[1], st)
        r += self.calErr1(feature[2], e1) / self.calErr1(feature[2], st)
        r += self.calErr1(feature[3], e1) / self.calErr1(feature[3], st)
        r /= 4
        r *= 0.9
        r = min(r, 0.9)

        for seq in feature:
            c += 1
            st_dist = self.calErr1(seq, st)
            err_dist = self.calErr1(seq, e1)
            # print("st_dist = ", st_dist, ", err_dist = ", err_dist, "in c = ", c)
            if (not c in self.err_set) and err_dist < st_dist * r:
                cou += 1
                self.err_set.append(c)
        return cou

    # 错误2：大小臂摇绳
    # 基于左手肘、右手肘运动序列进行判断
    def calErr2(self, target, refer):
        dist = 0

        for i in range(7, 9):
            p1 = self.str2num(target, target.columns[i])
            x1 = self.normalize(p1.iloc[:, 0])
            y1 = self.normalize(p1.iloc[:, 1])

            min_dist = 99999999
            for err in refer:
                tem_dist = 0
                p2 = self.str2num(err, err.columns[i])
                x2 = self.normalize(p2.iloc[:, 0])
                y2 = self.normalize(p2.iloc[:, 1])

                tem_dist += self.DTW(np.array(x1).reshape(-1, 1), np.array(x2).reshape(-1, 1))
                tem_dist += self.DTW(np.array(y1).reshape(-1, 1), np.array(y2).reshape(-1, 1))

                if tem_dist < min_dist:
                    min_dist = tem_dist

            dist += min_dist

        return dist

    def ifErr2(self, feature, st, e1):
        cou = 0
        c = 0

        r = 0

        r = self.calErr2(feature[0], e1) / self.calErr1(feature[0], st)
        r += self.calErr2(feature[1], e1) / self.calErr1(feature[1], st)
        r += self.calErr2(feature[2], e1) / self.calErr1(feature[2], st)
        r += self.calErr2(feature[3], e1) / self.calErr1(feature[3], st)
        r /= 4
        r *= 0.8
        r = min(r, 0.8)

        for seq in feature:
            c += 1
            st_dist = self.calErr2(seq, st)
            err_dist = self.calErr2(seq, e1)
            # print("st_dist = ", st_dist, ", err_dist = ", err_dist, "in c = ", c)
            if (c not in self.err_set) and err_dist < st_dist * r:
                cou += 1
                self.err_set.append(c)
        return cou

    # 错误3：过度屈膝
    # 基于脚踝、膝盖运动序列进行判断
    def calErr3(self, target, refer):
        dist = 0

        for i in range(13, 17):
            p1 = self.str2num(target, target.columns[i])
            x1 = self.normalize(p1.iloc[:, 0])
            y1 = self.normalize(p1.iloc[:, 1])

            min_dist = 99999999
            for err in refer:
                tem_dist = 0
                p2 = self.str2num(err, err.columns[i])
                x2 = self.normalize(p2.iloc[:, 0])
                y2 = self.normalize(p2.iloc[:, 1])

                tem_dist += self.DTW(np.array(x1).reshape(-1, 1), np.array(x2).reshape(-1, 1))
                tem_dist += self.DTW(np.array(y1).reshape(-1, 1), np.array(y2).reshape(-1, 1))

                if tem_dist < min_dist:
                    min_dist = tem_dist

            dist += min_dist

        return dist

    def ifErr3(self, feature, st, e1):
        cou = 0
        c = 0

        r = 0

        r = self.calErr3(feature[0], e1) / self.calErr1(feature[0], st)
        r += self.calErr3(feature[1], e1) / self.calErr1(feature[1], st)
        r += self.calErr3(feature[2], e1) / self.calErr1(feature[2], st)
        r += self.calErr3(feature[3], e1) / self.calErr1(feature[3], st)
        r /= 4
        r *= 0.8
        r = min(r, 0.8)

        for seq in feature:
            c += 1
            st_dist = self.calErr3(seq, st)
            err_dist = self.calErr3(seq, e1)
            # print("st_dist = ", st_dist, ", err_dist = ", err_dist, "in c = ", c)
            if (c not in self.err_set) and err_dist < st_dist * r:
                cou += 1
                self.err_set.append(c)
        return cou

    # 错误5：单脚跳
    # 基于脚踝运动序列进行判断
    def calErr5(self, target, refer):
        dist = 0

        for i in range(15, 17):
            p1 = self.str2num(target, target.columns[i])
            x1 = self.normalize(p1.iloc[:, 0])
            y1 = self.normalize(p1.iloc[:, 1])

            min_dist = 99999999
            for err in refer:
                tem_dist = 0
                p2 = self.str2num(err, err.columns[i])
                x2 = self.normalize(p2.iloc[:, 0])
                y2 = self.normalize(p2.iloc[:, 1])

                tem_dist += self.DTW(np.array(x1).reshape(-1, 1), np.array(x2).reshape(-1, 1))
                tem_dist += self.DTW(np.array(y1).reshape(-1, 1), np.array(y2).reshape(-1, 1))

                if tem_dist < min_dist:
                    min_dist = tem_dist

            dist = max(min_dist, dist)

        return dist

    def ifErr5(self, feature, st, e1):
        cou = 0
        c = 0

        r = self.calErr5(feature[0], e1) / self.calErr1(feature[0], st)
        r += self.calErr5(feature[1], e1) / self.calErr1(feature[1], st)
        r += self.calErr5(feature[2], e1) / self.calErr1(feature[2], st)
        r += self.calErr5(feature[3], e1) / self.calErr1(feature[3], st)
        r /= 4
        r *= 0.9
        r = min(r, 0.9)

        for seq in feature:
            c += 1
            st_dist = self.calErr5(seq, st)
            err_dist = self.calErr5(seq, e1)
            # print("st_dist = ", st_dist, ", err_dist = ", err_dist, "in c = ", c)
            if (c not in self.err_set) and err_dist < st_dist * r:
                cou += 1
                self.err_set.append(c)
        return cou

    def count_good(self):
        # 读入待识别文件
        df = pd.read_csv(self.aim, encoding='mbcs')
        # 抽样为30帧
        df = self.sample_30(df, self.aim_frame)
        # 读取关节点
        joins = self.readJoin(df)
        # 计算跳跃次数
        parr = self.countPeak(joins)
        # 提取跳跃运动特征
        feature = self.getSeq(df, parr)

        # 地日参考序列
        st = self.readRef('standard_1.csv', self.rpath)
        e1 = self.readRef('error_1_1.csv', self.rpath)
        e2 = self.readRef('error_2.csv', self.rpath)
        e3 = self.readRef('error_3.csv', self.rpath)
        e5 = self.readRef('error_4.csv', self.rpath)
        # 计算错误个数
        a1 = self.ifErr1(feature, st, e1)
        a2 = self.ifErr2(feature, st, e2)
        a3 = self.ifErr3(feature, st, e3)
        a5 = self.ifErr5(feature, st, e5)
        # 计算正确个数
        good = len(parr) - a1 - a2 - a3 - a5
        # 输出识别结果
        print("共跳了", len(parr), "次，其中 ", good, "次符合标准，另有", a1, "次错误1, ", a2, "次错误2, ",
              a3, "次错误3, ", a5, "次错误5。")
