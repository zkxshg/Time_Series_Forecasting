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
        path = self.aim
        data = np.loadtxt(path, dtype=np.float32, delimiter=",")

        good_path = self.refer + 'good.csv'
        bad_path = self.refer + 'bad.csv'

        good = np.loadtxt(good_path, dtype=np.float32, delimiter=",")
        bad = np.loadtxt(bad_path, dtype=np.float32, delimiter=",")

        left_df = pd.DataFrame(data)
        left_df.columns = pd.Series(['x', 'y'])
        left_df['date'] = left_df.index

        peak_arr = []
        length = left_df.shape[0]
        for i in range(1, length - 1):
            if self.isPeak(left_df['y'], i):
                peak_arr.append(i)



        peak_str = []
        peak_str.append(left_df['y'][0: peak_arr[1]].tolist())

        num = len(peak_arr)
        for i in range(1, num - 1):
            peak_str.append(left_df['y'][peak_arr[i - 1]: peak_arr[i + 1] + 1].tolist())
        good_fre = 0
        bad_fre = 0

        for pstr in peak_str:
            good_dist = DTW(good[:,1].reshape(-1, 1), np.array(peak_str[2]).reshape(-1, 1))
            bad_dist = DTW(bad[:,1].reshape(-1, 1), np.array(peak_str[2]).reshape(-1, 1))

            if good_dist <= bad_dist:
                good_fre += 1
            else:
                bad_fre += 1

        print("共跳了", len(peak_str), "次，其中 ", good_fre, "次符合标准，另有", bad_fre, "次不合标准")
