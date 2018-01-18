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
        vars = ["日期", "前收盘价(元)"]
        data = pd.read_csv(self.fns[0], encoding="gb2312")
        data = data[vars]
        data = np.array(data)
        ## use new dataset with corrent date order...
        data = data[::-1]
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
            ## use new dataset with corrent date order...
            temp = temp[::-1]
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

    for i in range(n_samples):
        ## let time window slices on the total timesteps
        ## first skip samples which contain zeor value
        if 0 in X_target_seg[i: i + related_ts] or 0 in Y_target_seg[i: i + related_ts]:
            continue
        X_target[i] = X_target_seg[i: i + related_ts]       ## shape=(related_ts,)
        Y_target[i] = Y_target_seg[i: i + related_ts]

        XY_driving[i] = XY_driving_seg[:, i: i + related_ts + 1]
        ## ***** allow XY_driving to have zero values, solve this problem in get_pearson_related_data()
        X_pos[i], X_neg[i], Y_pos[i], Y_neg[i] = get_pearson_related_data(X_target[i], XY_driving[i], k, timesteps, related_ts)
        X_hs300[i] = X_hs300_seg[i: i + related_ts]
        Y_hs300[i] = Y_hs300_seg[i: i + related_ts]
        ## calculate the difference
        # Y_target[i] = Y_target[i] - X_target[i]
        # Y_hs300[i] = Y_hs300[i] - X_hs300[i]

    X_target = np.reshape(X_target, newshape=(X_target.shape[0], X_target.shape[1], 1))
    Y_target = np.reshape(Y_target, newshape=(Y_target.shape[0], Y_target.shape[1], 1))
    X_hs300 = np.reshape(X_hs300, newshape=(X_hs300.shape[0], X_hs300.shape[1], 1))
    Y_hs300 = np.reshape(Y_hs300, newshape=(Y_hs300.shape[0], Y_hs300.shape[1], 1))

    ## ***** notice there will be samples which contain zeros. we don't delete these samples here because
    ## zero samples also contains date information. otherwise the data index and date among different stocks
    ## won't be aligned

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

def delete_zero_samples(X_target, Y_target, X_pos, X_neg, X_hs300):
    ## ***** notice since X_target are initialized with zeros, zero samples should be discarded
    zero_index = list()
    n_samples = X_target.shape[0]
    for i in range(n_samples):
        if 0 in X_target[i]:
            zero_index.append(i)

    _X_target = np.delete(X_target, zero_index, axis=0)
    _Y_target = np.delete(Y_target, zero_index, axis=0)
    _X_pos = np.delete(X_pos, zero_index, axis=0)
    # _Y_pos = np.delete(Y_pos, zero_index, axis=0)
    _X_neg = np.delete(X_neg, zero_index, axis=0)
    # _Y_neg = np.delete(Y_neg, zero_index, axis=0)
    _X_hs300 = np.delete(X_hs300, zero_index, axis=0)
    # _Y_hs300 = np.delete(Y_hs300, zero_index, axis=0)

    return _X_target, _Y_target, _X_pos, _X_neg, _X_hs300

def min_except_zero(data):
    ## calculates min data of each line except 0, and returns a mask which flags whether a value in data equals 0.
    _data = data
    max = np.max(data, axis=1)
    mask = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if(data[i, j] != 0):
                mask[i, j] = 1
            else:
                _data[i, j] = max[i] - 0.5
    min = np.min(_data, axis=1)
    return min, mask

def data_normalization(data):
    ## rescale every row in data to [-1, 1]
    ## notice that some of the stock data contains 0, when calculating min value zeros should be ignored.
    ## since stock value is always larger than 0, so np.max can be used directly.
    max_data = np.max(data, axis=1)
    # min_data = np.min(data, axis=1)
    min_data, mask = min_except_zero(data)
    max_data = np.reshape(max_data, (max_data.shape[0], 1))
    min_data = np.reshape(min_data, (min_data.shape[0], 1))
    data_norm = (2 * data - (max_data + min_data)) / (max_data - min_data)
    data_norm *= mask
    return data_norm, max_data, min_data

def data_restore(data_norm, max_data, min_data):
    data = (data_norm * (max_data - min_data) + (max_data + min_data)) / 2

    return data

def get_pearson_related_data(x_target, xy_driving, top_k, timesteps, related_ts):
    ## returns X_pos and X_neg of X_target, X_target is a sample with size (timesteps, )
    ## x_target: target series sample
    ## x_driving: all other driving series in the same time window of x_target (N, timesteps)
    ## top_k: the top_k driving series with the largest and smallest pcc in the driving series will be selected
    ## as X_pos and X_neg
    ## ***** notice that xy_driving could have zero values, series with zero should be excluded
    x_driving = xy_driving[:, : -1]
    y_driving = xy_driving[:, 1:]
    pcc = np.zeros(shape=x_driving.shape[0])
    for i in range(x_driving.shape[0]):
        ## haha, zero values still wouldn't be solved in this function, it would be solved in cal_pearson()
        pcc[i] = cal_pearson(x_target[:related_ts - timesteps], x_driving[i, :related_ts - timesteps])
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

    return X_pos, X_neg, Y_pos, Y_neg

def cal_pearson(x, y):
    ## basic pearson calculate function, x and y are both vectors
    ## ***** solve zero value problem here. if either x or y contains zero, this function returns 0, which means
    ## this 2 series are totally irrelevant thus the candidate series won't be selected as related series of
    ## target series.
    if 0 in x or 0 in y:
        return 0
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
    if np.isnan(p):
        p = 0
    return p

def cache_gen(src="matrix.npy", src_hs300="hs300.npy", data_cache="../cache"):
    total = 0
    for i in range(261):
        X_target, Y_target, X_pos, Y_pos, X_neg, Y_neg, max_data, min_data, X_hs300, Y_hs300 = build_data(src,
                   src_hs300,
                   data_cache,
                   target=i,
                   timesteps=15,
                   k=10,
                   related_ts=15)
        print("finish: %d" % i)
        print("sample num: %d" % X_target.shape[0])
        total += X_target.shape[0]
    print("total sample num: %d" % total)
    return total

def cal_tag(x, y):
    tag = np.zeros(shape=(x.shape[0], 1))
    flag = y[:, 0] - x[:, -1, 0]
    for i in range(tag.shape[0]):
        if flag[i] > 0:
            tag[i] = 1
        else:
            tag[i] = 0
    return tag

def get_train_val_test(timesteps=10, k=10, related_ts=30, val_days=200, test_days=100, type='proposed_model',
              src="matrix.npy", src_hs300="hs300.npy", data_cache="../cache", targets=range(261)):
    train = list()
    val = list()
    test = list()
    stock_num = len(targets)
    if type == 'proposed_model' or type == 'LSTM':
        X_target = 0
        Y_target = 0
        X_pos = 0
        X_neg = 0
        X_hs300 = 0

        str_data_test = "%s/LSTM%d_test_ts%d_k%d_rts%d_%d" % (data_cache, stock_num, timesteps, k, related_ts, test_days)
        str_data_val = "%s/LSTM%d_val_ts%d_k%d_rts%d_%d" % (data_cache, stock_num, timesteps, k, related_ts, val_days)
        str_data_train = "%s/LSTM%d_train_ts%d_k%d_rts%d" % (data_cache, stock_num, timesteps, k, related_ts)
        str_xt = "_xt.npy"
        str_yt = "_yt.npy"
        str_xp = "_xp.npy"
        str_xn = "_xn.npy"
        str_xhs = "_xhs.npy"

        if stock_num > 5 and os.path.exists("%s%s" % (str_data_test, str_xt)) and os.path.exists("%s%s" % (str_data_val, str_xt))\
                and os.path.exists("%s%s" % (str_data_train, str_xt)):
            print("train/val/test data exists")
            train.append(np.load("%s%s" % (str_data_train, str_xt)))
            train.append(np.load("%s%s" % (str_data_train, str_yt)))
            train.append(cal_tag(train[0], train[1]))
            val.append(np.load("%s%s" % (str_data_val, str_xt)))
            val.append(np.load("%s%s" % (str_data_val, str_yt)))
            val.append(cal_tag(val[0], val[1]))
            test.append(np.load("%s%s" % (str_data_test, str_xt)))
            test.append(np.load("%s%s" % (str_data_test, str_yt)))
            test.append(cal_tag(test[0], test[1]))
            if type == "proposed_model":
                train.append(np.load("%s%s" % (str_data_train, str_xp)))
                train.append(np.load("%s%s" % (str_data_train, str_xn)))
                train.append(np.load("%s%s" % (str_data_train, str_xhs)))
                val.append(np.load("%s%s" % (str_data_val, str_xp)))
                val.append(np.load("%s%s" % (str_data_val, str_xn)))
                val.append(np.load("%s%s" % (str_data_val, str_xhs)))
                test.append(np.load("%s%s" % (str_data_test, str_xp)))
                test.append(np.load("%s%s" % (str_data_test, str_xn)))
                test.append(np.load("%s%s" % (str_data_test, str_xhs)))
            return train, val, test
        else:
            print("no train/val/test data found")
        ## get test set
        for i in targets:
            X_t, Y_t, X_p, Y_p, X_n, Y_n, max_data, min_data, X_hs, Y_hs = build_data(src,
                                                                                      src_hs300,
                                                                                      data_cache,
                                                                                      target=i,
                                                                                      timesteps=timesteps,
                                                                                      k=k,
                                                                                      related_ts=related_ts)
            n = X_t.shape[0]
            X_t = X_t[n - test_days:, :, :]
            Y_t = Y_t[n - test_days:, -1:, 0]
            ## we want to add 2 dims to record min and max data of Y_t before normalization in order to
            ## restore the raw value of Y_t. so remember to use only the first dim for training
            max = np.ones_like(Y_t) * max_data[i]
            min = np.ones_like(Y_t) * min_data[i]
            Y_t = np.concatenate((Y_t, max, min), axis=1)

            X_p = np.transpose(X_p, (0, 2, 1))[n - test_days:, :, :]
            X_n = np.transpose(X_n, (0, 2, 1))[n - test_days:, :, :]
            X_hs = X_hs[n - test_days:, :, :]

            if i == targets[0]:
                X_target = X_t
                Y_target = Y_t
                X_pos = X_p
                X_neg = X_n
                X_hs300 = X_hs
            else:
                X_target = np.concatenate((X_target, X_t), axis=0)
                Y_target = np.concatenate((Y_target, Y_t), axis=0)
                X_pos = np.concatenate((X_pos, X_p), axis=0)
                X_neg = np.concatenate((X_neg, X_n), axis=0)
                X_hs300 = np.concatenate((X_hs300, X_hs), axis=0)
        X_target, Y_target, X_pos, X_neg, X_hs300 = delete_zero_samples(X_target, Y_target, X_pos, X_neg, X_hs300)

        np.save("%s%s" % (str_data_test, str_xt), X_target)
        np.save("%s%s" % (str_data_test, str_yt), Y_target)
        np.save("%s%s" % (str_data_test, str_xp), X_pos)
        np.save("%s%s" % (str_data_test, str_xn), X_neg)
        np.save("%s%s" % (str_data_test, str_xhs), X_hs300)

        test.append(X_target)
        test.append(Y_target)
        test.append(cal_tag(X_target, Y_target))
        if type == 'proposed_model':
            test.append(X_pos)
            test.append(X_neg)
            test.append(X_hs300)

        ## get validate set
        for i in targets:
            X_t, Y_t, X_p, Y_p, X_n, Y_n, max_data, min_data, X_hs, Y_hs = build_data(src,
                                                                                      src_hs300,
                                                                                      data_cache,
                                                                                      target=i,
                                                                                      timesteps=timesteps,
                                                                                      k=k,
                                                                                      related_ts=related_ts)
            n = X_t.shape[0]
            X_t = X_t[n - test_days - val_days: n - test_days, :, :]
            Y_t = Y_t[n - test_days - val_days: n - test_days, -1:, 0]
            ## we want to add 2 dims to record min and max data of Y_t before normalization in order to
            ## restore the raw value of Y_t. so remember to use only the first dim for training
            max = np.ones_like(Y_t) * max_data[i]
            min = np.ones_like(Y_t) * min_data[i]
            Y_t = np.concatenate((Y_t, max, min), axis=1)

            X_p = np.transpose(X_p, (0, 2, 1))[n - test_days - val_days: n - test_days, :, :]
            X_n = np.transpose(X_n, (0, 2, 1))[n - test_days - val_days: n - test_days, :, :]
            X_hs = X_hs[n - test_days - val_days: n - test_days, :, :]

            if i == targets[0]:
                X_target = X_t
                Y_target = Y_t
                X_pos = X_p
                X_neg = X_n
                X_hs300 = X_hs
            else:
                X_target = np.concatenate((X_target, X_t), axis=0)
                Y_target = np.concatenate((Y_target, Y_t), axis=0)
                X_pos = np.concatenate((X_pos, X_p), axis=0)
                X_neg = np.concatenate((X_neg, X_n), axis=0)
                X_hs300 = np.concatenate((X_hs300, X_hs), axis=0)
        X_target, Y_target, X_pos, X_neg, X_hs300 = delete_zero_samples(X_target, Y_target, X_pos, X_neg, X_hs300)

        np.save("%s%s" % (str_data_val, str_xt), X_target)
        np.save("%s%s" % (str_data_val, str_yt), Y_target)
        np.save("%s%s" % (str_data_val, str_xp), X_pos)
        np.save("%s%s" % (str_data_val, str_xn), X_neg)
        np.save("%s%s" % (str_data_val, str_xhs), X_hs300)

        val.append(X_target)
        val.append(Y_target)
        val.append(cal_tag(X_target, Y_target))
        if type == 'proposed_model':
            val.append(X_pos)
            val.append(X_neg)
            val.append(X_hs300)

        ## get train set
        for i in targets:
            X_t, Y_t, X_p, Y_p, X_n, Y_n, max_data, min_data, X_hs, Y_hs = build_data(src,
                                                                                      src_hs300,
                                                                                      data_cache,
                                                                                      target=i,
                                                                                      timesteps=timesteps,
                                                                                      k=k,
                                                                                      related_ts=related_ts)
            n = X_t.shape[0]
            X_t = X_t[:n - test_days - val_days, :, :]
            Y_t = Y_t[:n - test_days - val_days, -1:, 0]
            ## we want to add 2 dims to record min and max data of Y_t before normalization in order to
            ## restore the raw value of Y_t. so remember to use only the first dim for training
            max = np.ones_like(Y_t) * max_data[i]
            min = np.ones_like(Y_t) * min_data[i]
            Y_t = np.concatenate((Y_t, max, min), axis=1)

            X_p = np.transpose(X_p, (0, 2, 1))[:n - test_days - val_days, :, :]
            X_n = np.transpose(X_n, (0, 2, 1))[:n - test_days - val_days, :, :]
            X_hs = X_hs[:n - test_days - val_days, :, :]

            if i == targets[0]:
                X_target = X_t
                Y_target = Y_t
                X_pos = X_p
                X_neg = X_n
                X_hs300 = X_hs
            else:
                X_target = np.concatenate((X_target, X_t), axis=0)
                Y_target = np.concatenate((Y_target, Y_t), axis=0)
                X_pos = np.concatenate((X_pos, X_p), axis=0)
                X_neg = np.concatenate((X_neg, X_n), axis=0)
                X_hs300 = np.concatenate((X_hs300, X_hs), axis=0)
        X_target, Y_target, X_pos, X_neg, X_hs300 = delete_zero_samples(X_target, Y_target, X_pos, X_neg, X_hs300)

        np.save("%s%s" % (str_data_train, str_xt), X_target)
        np.save("%s%s" % (str_data_train, str_yt), Y_target)
        np.save("%s%s" % (str_data_train, str_xp), X_pos)
        np.save("%s%s" % (str_data_train, str_xn), X_neg)
        np.save("%s%s" % (str_data_train, str_xhs), X_hs300)

        train.append(X_target)
        train.append(Y_target)
        train.append(cal_tag(X_target, Y_target))
        if type == 'proposed_model':
            train.append(X_pos)
            train.append(X_neg)
            train.append(X_hs300)

        return train, val, test
    else:
        return 0



if __name__ == "__main__":
    # np.set_printoptions(threshold=np.inf)
    # directory = "history_data"
    # dm = DataManager(directory)
    # dm.getMatrix(dm.commonFirstDate, dm.commonLastDate)
    # data = np.load("matrix.npy")      ## we have 261 stocks
    # print(data.shape)
    print()