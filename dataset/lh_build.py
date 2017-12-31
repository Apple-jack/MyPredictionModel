import numpy as np
import pandas as pd
import os
from datetime import datetime

class DataManager:
    def __init__(self, src):
        self.src = src
        self.firstDates = list()
        self.lastDates = list()
        self.clear()
        self.update()
        self.commonFirstDate = datetime.strptime("2013-04-25", "%Y-%m-%d")
        self.commonLastDate = datetime.strptime("2017-05-15", "%Y-%m-%d")

    def update(self):
        ## update filenames
        self.fns = os.listdir(self.src)
        for i in range(len(self.fns)):
            self.fns[i] = os.path.join(self.src, self.fns[i])
        self.stockNum = len(self.fns)

    def clear(self):
        ## clear empty files
        fns = os.listdir(self.src)
        count = 0
        for fn in fns:
            path = os.path.join(self.src, fn)
            if os.path.getsize(path) < 200 or fn == ".DS_Store":
                os.remove(path)
                count += 1
                print("%s: removed" % path)
        self.update()
        return 0

    def firstDate(self, path):
        data = pd.read_csv(path, encoding="gb2312")
        vars = ["日期"]
        data = np.array(data[vars])
        date = datetime.strptime("".join(data[-1]), "%Y-%m-%d")
        return date

    def lastDate(self, path):
        data = pd.read_csv(path, encoding="gb2312")
        vars = ["日期"]
        data = np.array(data[vars])
        date = datetime.strptime("".join(data[0]), "%Y-%m-%d")
        return date

    def calFirstDates(self):
        ## common first date in this dataset is 2013-04-25
        self.update()
        self.firstDates = list()
        temp = datetime.strptime("2000-1-1", "%Y-%m-%d")
        for fn in self.fns:
            date = self.firstDate(fn)
            self.firstDates.append(date)
            if temp < date:
                temp = date
            print(fn)
        self.commonFirstDate = temp
        print("%d files readed." % len(self.firstDates))
        return self.firstDates

    def calLastDates(self):
        ## common last date in this dataset is 2017-05-15
        self.update()
        self.lastDates = list()
        temp = datetime.today()
        for fn in self.fns:
            date = self.lastDate(fn)
            self.lastDates.append(date)
            if temp > date:
                temp = date
            print(fn)
        self.commonLastDate = temp
        print("%d files readed." % len(self.firstDates))
        return self.lastDates

    def getMatrix(self, startDate, endDate):
        assert startDate >= self.commonFirstDate
        assert endDate <= self.commonLastDate
        vars = ["日期", "收盘价"]
        data = pd.read_csv(self.fns[0], encoding="gb2312")
        data = data[vars]
        data = np.array(data)
        startIndex, endIndex = getDateIndex(data, startDate, endDate)
        data = data[endIndex: startIndex + 1, 1]        ## notice low index refer to newer data, we need to reverse it
        data = data[::-1]       ## reverse data
        data = np.expand_dims(data, 1)
        print(data.shape)
        errors = list()
        for i in range(1, len(self.fns)):
            temp = pd.read_csv(self.fns[i], encoding="gb2312")
            temp = temp[vars]
            temp = np.array(temp)
            startIndex, endIndex = getDateIndex(temp, startDate, endDate)
            temp = temp[endIndex: startIndex + 1, 1]
            temp = temp[::-1]
            temp = np.expand_dims(temp, 1)
            if data.shape[0] != temp.shape[0]:
                errors.append([self.fns[i], temp.shape[0]])
            else:
                data = np.concatenate((data, temp), 1)
                print(data.shape)
            print(i)
        print(errors)
        np.save("errors.npy", errors)
        np.save("matrix.npy", data.transpose())
        return 0

    def test(self):
        names = os.listdir(self.src)
        path = os.path.join(self.src, names[0])
        date1 = self.firstDate(path)
        date2 = datetime.strptime("2012-05-12", "%Y-%m-%d")
        if date1 > date2:
            print("%s > %s" % (date1, date2))
        else:
            print("%s <= %s" % (date1, date2))

def getDateIndex(data, startDate, endDate):
    startIndex = endIndex = 0
    for i in range(data.shape[0]):
        if data[i, 0] == startDate.strftime("%Y-%m-%d"):
            startIndex = i
            break
        if data[i, 0] == endDate.strftime("%Y-%m-%d"):
            endIndex = i
    return startIndex, endIndex

def load_hs300():
    startDate = datetime.strptime("2013-04-25", "%Y-%m-%d")
    endDate = datetime.strptime("2017-05-15", "%Y-%m-%d")
    vars = ["日期", "收盘价"]
    data = pd.read_csv("000300.csv", encoding="gb2312")
    data = data[vars]
    data = np.array(data)
    startIndex, endIndex = getDateIndex(data, startDate, endDate)
    data = data[endIndex: startIndex + 1, 1]  ## notice low index refer to newer data, we need to reverse it
    data = data[::-1]  ## reverse data
    data = np.expand_dims(data, 1)
    print(data.shape)
    np.save("hs300.npy", data.transpose())
    return data.transpose()

def build_data(src, src_hs300, cache_src, target, timesteps, k, related_ts):
    ## using "matrix_no_zero.npy"
    ## src: data src
    ## cache_src: once data is loaded, it would be saved to cache
    ## target: target stocks array
    ## timesteps: time window size of one sample
    ## k: the stock number of each driving series group
    ## related_ts: how many timesteps are used to calculate correlation coefficient

    data = np.load(src)  ## shape=(total_stocks, total_timesteps)
    data_norm, max_data, min_data = data_normalization(data)

    data_hs300 = np.load(src_hs300)
    data_hs300_norm, hs300_max, hs300_min = data_normalization(data_hs300)

    cache_target = "%s/target%d_ts%d_k%d_rts%d.npy" % (cache_src, target, timesteps, k, related_ts)
    cache_targety = "%s/targety%d_ts%d_k%d_rts%d.npy" % (cache_src, target, timesteps, k, related_ts)
    cache_pos = "%s/pos%d_ts%d_k%d_rts%d.npy" % (cache_src, target, timesteps, k, related_ts)
    cache_posy = "%s/posy%d_ts%d_k%d_rts%d.npy" % (cache_src, target, timesteps, k, related_ts)
    cache_neg = "%s/neg%d_ts%d_k%d_rts%d.npy" % (cache_src, target, timesteps, k, related_ts)
    cache_negy = "%s/negy%d_ts%d_k%d_rts%d.npy" % (cache_src, target, timesteps, k, related_ts)
    cache_hs300x = "%s/hs300x%d_ts%d_k%d_rts%d.npy" % (cache_src, target, timesteps, k, related_ts)
    cache_hs300y = "%s/hs300y%d_ts%d_k%d_rts%d.npy" % (cache_src, target, timesteps, k, related_ts)

    if os.path.exists(cache_target) and os.path.exists(cache_targety) and \
            os.path.exists(cache_pos) and os.path.exists(cache_neg) and os.path.exists(cache_posy) and os.path.exists(cache_negy)\
            and os.path.exists(cache_hs300x) and os.path.exists(cache_hs300y):
        print("data exists")
        return np.load(cache_target), np.load(cache_targety), np.load(cache_pos), np.load(cache_posy), np.load(cache_neg), \
               np.load(cache_negy), max_data, min_data, np.load(cache_hs300x), np.load(cache_hs300y)

    print("data not exists")

    X_target_seg = data_norm[target, : -1]        ## shape=(total_timesteps - 1,)
    Y_target_seg = data_norm[target, 1:]          ## shape=(total_timesteps - 1,)
    XY_driving_seg = np.delete(data_norm, target, axis=0)    ## shape=(total_stocks - 1, total_timesteps)
    X_hs300_seg = data_hs300_norm[0, : -1]        ## shape=(total_timesteps - 1,)
    Y_hs300_seg = data_hs300_norm[0, 1:]          ## shape=(total_timesteps - 1,)

    n_samples = X_target_seg.shape[0] - related_ts + 1
    X_target = np.zeros(shape=(n_samples, related_ts))      ## shape=(n_samples, related_ts)
    Y_target = np.zeros(shape=(n_samples, related_ts))
    XY_driving = np.zeros(shape=(n_samples, XY_driving_seg.shape[0], related_ts + 1))     ## shape=(n_samples, total_stocks - 1, related_ts)
    X_pos = np.zeros(shape=(n_samples, k, related_ts))      ## shape=(n_samples, top_k, related_ts)
    X_neg = np.zeros(shape=(n_samples, k, related_ts))      ## shape=(n_samples, top_k, realted_ts)
    Y_pos = np.zeros(shape=(n_samples, k, related_ts))      ## shape=(n_samples, top_k, related_ts)
    Y_neg = np.zeros(shape=(n_samples, k, related_ts))      ## shape=(n_samples, top_k, related_ts)
    X_hs300 = np.zeros(shape=(n_samples, related_ts))
    Y_hs300 = np.zeros(shape=(n_samples, related_ts))

    for i in range(X_target_seg.shape[0] - related_ts + 1):
        ## let time window slices on the total timesteps
        X_target[i] = X_target_seg[i: i + related_ts]       ## shape=(related_ts,)
        Y_target[i] = Y_target_seg[i: i + related_ts]
        XY_driving[i] = XY_driving_seg[:, i: i + related_ts + 1]
        X_pos[i], X_neg[i], Y_pos[i], Y_neg[i] = get_pearson_related_data(X_target[i], XY_driving[i], k)
        X_hs300[i] = X_hs300_seg[i: i + related_ts]
        Y_hs300[i] = Y_hs300_seg[i: i + related_ts]
        ## calculate the difference
        # Y_target[i] = Y_target[i] - X_target[i]
        # Y_hs300[i] = Y_hs300[i] - X_hs300[i]

    X_target = np.reshape(X_target, newshape=(X_target.shape[0], X_target.shape[1], 1))
    Y_target = np.reshape(Y_target, newshape=(Y_target.shape[0], Y_target.shape[1], 1))
    X_hs300 = np.reshape(X_hs300, newshape=(X_hs300.shape[0], X_hs300.shape[1], 1))
    Y_hs300 = np.reshape(Y_hs300, newshape=(Y_hs300.shape[0], Y_hs300.shape[1], 1))

    print("saving cache...")
    np.save(cache_target, X_target[:, -timesteps:])
    np.save(cache_targety, Y_target[:, -timesteps:])
    np.save(cache_pos, X_pos[:, :, -timesteps:])
    np.save(cache_posy, Y_pos[:, :, -timesteps:])
    np.save(cache_neg, X_neg[:, :, -timesteps:])
    np.save(cache_negy, Y_neg[:, :, -timesteps:])
    np.save(cache_hs300x, X_hs300[:, -timesteps:])
    np.save(cache_hs300y, Y_hs300[:, -timesteps:])


    return X_target[:, -timesteps:], Y_target[:, -timesteps:], X_pos[:, :, -timesteps:], Y_pos[:, :, -timesteps:], X_neg[:, :, -timesteps:], \
           Y_neg[:, :, -timesteps:], max_data, min_data, X_hs300[:, -timesteps:], Y_hs300[:, -timesteps:]

def load_data_2(src, src_hs300, timestep, return_sequence):
    data = np.load(src)
    data_hs300 = np.load(src_hs300)
    data = np.concatenate((data, data_hs300), axis=0)
    data_real = data[:, :]
    data = data[:, :]

    # # data normalization
    max_data = np.max(data, axis=1)
    min_data = np.min(data, axis=1)
    max_data = np.reshape(max_data, (max_data.shape[0], 1))
    min_data = np.reshape(min_data, (min_data.shape[0], 1))
    data = (2 * data - (max_data + min_data)) / (max_data - min_data)

    # dataset split
    train_split = round(0.8 * data.shape[1])
    val_split = round(0.9 * data.shape[1])

    x_train_seg = data[:, : train_split]
    y_train_seg = data[:, 1: train_split + 1]
    x_val_seg = data[:, train_split: val_split]
    y_val_seg = data[:, train_split + 1: val_split + 1]
    x_test_seg = data[:, val_split: -1]
    y_test_seg = data[:, val_split + 1:]
    gt_test_seg = data_real[:, val_split + 1:]

    n_train = (x_train_seg.shape[1] - timestep + 1) * x_train_seg.shape[0]
    n_val = (x_val_seg.shape[1] - timestep + 1) * x_val_seg.shape[0]
    n_test = (x_test_seg.shape[1] - timestep + 1) * x_test_seg.shape[0]
    #LSTM
    x_train = np.zeros(shape=(n_train, timestep))
    y_train = np.zeros(shape=(n_train, timestep))
    x_val = np.zeros(shape=(n_val, timestep))
    y_val = np.zeros(shape=(n_val, timestep))
    x_test = np.zeros(shape=(n_test, timestep))
    y_test = np.zeros(shape=(n_test, timestep))
    gt_test = np.zeros(shape=(n_test, timestep))

    i = 0
    for j in range(x_train_seg.shape[0]):
        for k in range(x_train_seg.shape[1] - timestep + 1):
            x_train[i, :] = x_train_seg[j, k: k + timestep]
            y_train[i, :] = y_train_seg[j, k: k + timestep]
            i += 1
    assert i == n_train

    i = 0
    for j in range(x_val_seg.shape[0]):
        for k in range(x_val_seg.shape[1] - timestep + 1):
            x_val[i, :] = x_val_seg[j, k: k + timestep]
            y_val[i, :] = y_val_seg[j, k: k + timestep]
            i += 1
    assert i == n_val

    i = 0
    for j in range(x_test_seg.shape[0]):
        for k in range(x_test_seg.shape[1] - timestep + 1):
            x_test[i, :] = x_test_seg[j, k: k + timestep]
            y_test[i, :] = y_test_seg[j, k: k + timestep]
            gt_test[i, :] = gt_test_seg[j, k: k + timestep]
            i += 1
    assert i == n_test

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
    y_val = np.reshape(y_val, (y_val.shape[0], y_val.shape[1], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))
    gt_test = np.reshape(gt_test, (gt_test.shape[0], gt_test.shape[1], 1))

    if return_sequence:
        return [x_train, y_train, x_val, y_val, x_test, y_test, gt_test, max_data, min_data]
    else:
        return [x_train, y_train[:, -1], x_val, y_val[:, -1], x_test, y_test[:, -1], gt_test[:, -1], max_data, min_data]

def data_normalization(data):
    ## rescale every row in data to [-1, 1]
    max_data = np.max(data, axis=1)
    min_data = np.min(data, axis=1)
    max_data = np.reshape(max_data, (max_data.shape[0], 1))
    min_data = np.reshape(min_data, (min_data.shape[0], 1))
    data_norm = (2 * data - (max_data + min_data)) / (max_data - min_data)

    return data_norm, max_data, min_data

def data_restore(data_norm, max_data, min_data):
    data = (data_norm * (max_data - min_data) + (max_data + min_data)) / 2

    return data

def get_pearson_related_data(x_target, xy_driving, top_k):
    ## returns X_pos and X_neg of X_target, X_target is a sample with size (timesteps, )
    ## x_target: target series sample
    ## x_driving: all other driving series in the same time window of x_target (N, timesteps)
    ## top_k: the top_k driving series with the largest and smallest pcc in the driving series will be selected
    ## as X_pos and X_neg
    x_driving = xy_driving[:, : -1]
    y_driving = xy_driving[:, 1:]
    pcc = np.zeros(shape=x_driving.shape[0])
    for i in range(x_driving.shape[0]):
        pcc[i] = cal_pearson(x_target, x_driving[i])
    pcc_index = pcc.argsort()
    pos_index = pcc_index[-top_k:]
    pos_index = pos_index[::-1]     ## note that we want lower index to store larger pcc series, reverse is needed here
    neg_index = pcc_index[: top_k]  ## lower index already stores small pcc series

    X_pos = x_driving[pos_index[0]: pos_index[0] + 1, :]
    X_neg = x_driving[neg_index[0]: neg_index[0] + 1, :]
    Y_pos = y_driving[pos_index[0]: pos_index[0] + 1, :]
    Y_neg = y_driving[neg_index[0]: neg_index[0] + 1, :]

    for i in range(1, top_k):
        X_pos = np.concatenate((X_pos, x_driving[pos_index[i]: pos_index[i] + 1, :]), axis=0)
        X_neg = np.concatenate((X_neg, x_driving[neg_index[i]: neg_index[i] + 1, :]), axis=0)
        Y_pos = np.concatenate((Y_pos, y_driving[pos_index[i]: pos_index[i] + 1, :]), axis=0)
        Y_neg = np.concatenate((Y_neg, y_driving[neg_index[i]: neg_index[i] + 1, :]), axis=0)

    Y_pos = Y_pos - X_pos
    Y_neg = Y_neg - X_neg

    return X_pos, X_neg, Y_pos, Y_neg

def cal_pearson(x, y):
    ## basic pearson calculate function, x and y are both vectors
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    n = len(x)
    sumTop = 0.0
    sumBottom = 0.0
    x_pow = 0.0
    y_pow = 0.0
    for i in range(n):
        sumTop += (x[i] - x_mean) * (y[i] - y_mean)
    for i in range(n):
        x_pow += np.power(x[i] - x_mean, 2)
    for i in range(n):
        y_pow += np.power(y[i] - y_mean, 2)
    sumBottom = np.sqrt(x_pow * y_pow)
    p = sumTop / sumBottom
    return p

if __name__ == "__main__":
    directory = "history_data"
    # dm = DataManager(directory)
    # dm.getMatrix(dm.commonFirstDate, dm.commonLastDate)
    # data = np.load("matrix.npy")
    # errors = np.load("errors.npy")
    # data_no_zero = np.load("matrix_no_zero.npy")
    # print(load_hs300())
    # print(data.shape)
    # print(len(errors))
    # count = 0
    # total = 0
    #################
    ## this section receives matrix.npy in variable "data" and outputs matrix_no_zero.npy
    # i = 0
    # N = data.shape[0]
    # while total < N:
    #     if 0 in data[i]:
    #         data = np.delete(data, i, axis=0)
    #         count += 1
    #         i -= 1
    #     total += 1
    #     print(total)
    #     i += 1
    #
    # print("%d / %d" % (count, total))
    #
    # print(data.shape)
    # np.save("matrix_no_zero.npy", data)
    ################