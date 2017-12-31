import numpy as np

def build_data(target, timesteps, k, related_ts):
    ## using "matrix_no_zero.npy"
    ## target: target stocks array
    ## timesteps: time window size of one sample
    ## k: the stock number of each driving series group
    ## related_ts: how many timesteps are used to calculate correlation coefficient

    data = np.array([[1,2,3,4,5],
                 [1,3,5,7,9],
                 [-1,-3,-5,-7,-9],
                 [1,2,5,7,3],
                 [2,-2,7,-5,1],
                 [9,6,6,3,1],
                 [-1,2,6,9,8],
                 [-5,-7,-3,1,7]])

    for i_x in target:
        X_target_seg = data[i_x]        ## shape=(total_timesteps,)
        X_driving_seg = np.delete(data, i_x, axis=0)    ## shape=(total_stocks, total_timesteps)

        n_samples = X_target_seg.shape[0] - related_ts + 1
        X_target = np.zeros(shape=(n_samples, related_ts))      ## shape=(n_samples, related_ts)
        X_driving = np.zeros(shape=(n_samples, X_driving_seg.shape[0], related_ts))     ## shape=(n_samples, total_stocks, related_ts)
        X_pos = np.zeros(shape=(n_samples, k, related_ts))      ## shape=(n_samples, top_k, related_ts)
        X_neg = np.zeros(shape=(n_samples, k, related_ts))      ## shape=(n_samples, top_k, realted_ts)
        for i in range(X_target_seg.shape[0] - related_ts + 1):
            ## let time window slices on the total timesteps
            X_target[i] = X_target_seg[i: i + related_ts]       ## shape=(related_ts,)
            X_driving[i] = X_driving_seg[:, i: i + related_ts]
            X_pos[i], X_neg[i] = get_pearson_related_data(X_target[i], X_driving[i], k)

    return X_target[:, -timesteps:], X_pos[:, :, -timesteps:], X_neg[:, :, -timesteps:]

def get_pearson_related_data(x_target, x_driving, top_k):
    ## returns X_pos and X_neg of X_target, X_target is a sample with size (timesteps, )
    ## x_target: target series sample
    ## x_driving: all other driving series in the same time window of x_target (N, timesteps)
    ## top_k: the top_k driving series with the largest and smallest pcc in the driving series will be selected
    ## as X_pos and X_neg
    pcc = np.zeros(shape=x_driving.shape[0])
    for i in range(x_driving.shape[0]):
        pcc[i] = cal_pearson(x_target, x_driving[i])
    pcc_index = pcc.argsort()
    pos_index = pcc_index[-top_k:]
    pos_index = pos_index[::-1]
    neg_index = pcc_index[: top_k]

    X_pos = x_driving[pos_index[0]: pos_index[0] + 1, :]
    X_neg = x_driving[neg_index[0]: neg_index[0] + 1, :]

    for i in range(1, top_k):
        X_pos = np.concatenate((X_pos, x_driving[pos_index[i]: pos_index[i] + 1, :]), axis=0)
        X_neg = np.concatenate((X_neg, x_driving[neg_index[i]: neg_index[i] + 1, :]), axis=0)

    return X_pos, X_neg

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

def data_normalization(data):
    max_data = np.max(data, axis=1)
    min_data = np.min(data, axis=1)
    max_data = np.reshape(max_data, (max_data.shape[0], 1))
    min_data = np.reshape(min_data, (min_data.shape[0], 1))
    data = (2 * data - (max_data + min_data)) / (max_data - min_data)
    return data, max_data, min_data


def data_restore(data_norm, max_data, min_data):
    data = (data_norm * (max_data - min_data) + (max_data + min_data)) / 2

    return data

if __name__ == "__main__":
    x_target = np.array([1,2,3,4,5])
    x_driving = np.array([[1,3,5,7,9],
                 [-1,-3,-5,-7,-9],
                 [1,2,5,7,3],
                 [2,-2,7,-5,1],
                 [9,6,6,3,1],
                 [-1,2,6,9,8],
                 [-5,-7,-3,1,7]])

    # for i in range(x_driving.shape[0]):
    #     print(cal_pearson(x_target, x_driving[i]))
    X_pos, X_neg = get_pearson_related_data(x_target, x_driving, 2)
    print(X_pos)
    print(X_neg)

    print("__________")
    t, p, n = build_data([0], 3, 2, 5)
    print(p)
    print(n)
    print(t)

    print("-----------------")
    x_n, max, min = data_normalization(x_driving)
    print(x_n)
    x_r = data_restore(x_n, max, min)
    print(x_r)
