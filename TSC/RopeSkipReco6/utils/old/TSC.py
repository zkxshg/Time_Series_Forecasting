import numpy as np
import pandas as pd
from dtw import dtw
from numpy.linalg import norm
from numpy import array


class TSC:
    def __init__(self, refer, aim):
        self.refer = refer
        self.aim = aim

    def read_data(self, path):
        data = np.loadtxt(path, dtype=np.float32, delimiter=",")
        return data

    def normalize(self, df):
        df['0_y'] = ( df['0_y']- df['0_y'].mean())/(df['0_y'].std())  
        return df

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

    def DTW(self, x, y):
        dist, cost, acc, path = dtw(x, y, dist=lambda x, y: norm(x - y, ord=1))
        return dist

    def countGood(self):
        data = pd.read_csv(self.aim)

        good_path = self.refer + 'standard.csv'
        false_1_path = self.refer + 'false1.csv'
        false_2_path = self.refer + 'false2.csv'
        false_3_path = self.refer + 'false3.csv'
        false_4_path = self.refer + 'false4.csv'
        false_5_path = self.refer + 'false5.csv'

        good = pd.read_csv(good_path)
        false_1 = pd.read_csv(false_1_path)
        false_2 = pd.read_csv(false_2_path)
        false_3 = pd.read_csv(false_3_path)
        false_4 = pd.read_csv(false_4_path)
        false_5 = pd.read_csv(false_5_path)

        self.normalize(data)
        self.normalize(good)
        self.normalize(false_1)
        self.normalize(false_2)
        self.normalize(false_3)
        self.normalize(false_4)
        self.normalize(false_5)

        peak_arr = []
        left_df = data
        length = left_df.shape[0]
        for i in range(1, length - 1):
            if (self.isPeak(left_df['0_y'], i)):
                peak_arr.append(i)



        peak_str = []
        peak_str.append(left_df['0_y'][0 : peak_arr[1]].tolist())

        num = len(peak_arr)
        for i in range(1, num - 1):
            peak_str.append(left_df['0_y'][peak_arr[i - 1] : peak_arr[i + 1] + 1].tolist())

        good_fre = 0
        f1_fre = 0
        f2_fre = 0
        f3_fre = 0
        f4_fre = 0
        f5_fre = 0

        for pstr in peak_str:
            good_dist = self.DTW(good['0_y'], np.array(peak_str[2]).reshape(-1, 1))
            f1_dist = self.DTW(false_1['0_y'], np.array(peak_str[2]).reshape(-1, 1))
            f2_dist = self.DTW(false_2['0_y'], np.array(peak_str[2]).reshape(-1, 1))
            f3_dist = self.DTW(false_3['0_y'], np.array(peak_str[2]).reshape(-1, 1))
            f4_dist = self.DTW(false_4['0_y'], np.array(peak_str[2]).reshape(-1, 1))
            f5_dist = self.DTW(false_5['0_y'], np.array(peak_str[2]).reshape(-1, 1))
    
            min_dist = min(good_dist, f1_dist, f2_dist, f3_dist, f4_dist, f5_dist)
    
            if good_dist == min_dist: 
                good_fre += 1
            elif f1_dist == min_dist:
                f1_fre += 1
            elif f2_dist == min_dist:
                f2_fre += 1
            elif f3_dist == min_dist:
                f3_fre += 1
            elif f4_dist == min_dist:
                f4_fre += 1
            else:
                f5_fre += 1

        print("共跳了", len(peak_str), "次，其中 ", good_fre, "次符合标准，另有", f1_fre, "次错误1, ", f2_fre, "次错误2, ", 
            f3_fre, "次错误3, ", f4_fre, "次错误4, ", f5_fre, "次错误5, ")
