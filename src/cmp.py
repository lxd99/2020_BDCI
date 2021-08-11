import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import numpy as np
import matplotlib.pyplot as plt

tx = pd.DataFrame([[1, 2], [3, 4]])


def duck():
    data = pd.read_csv('../data/train.csv')
    data['mean'] = data[[f'{j}_{i}' for i in range(1, 6) for j in ['CPU_USAGE', 'cpu']]].std()
    fdata = pd.DataFrame(pd.np.empty((0, len(data.columns))), columns=data.columns)
    for id in tqdm(data['QUEUE_ID'].unique().tolist()):
        mdf = data[data['QUEUE_ID'] == id]
        mdf = mdf.sort_values(by=['mean']).reset_index(drop=True)
        mdf = mdf.iloc[:int(0.9 * mdf.shape[0])]
        fdata = fdata.append(mdf)
    # fdata.drop('mean')
    fdata = fdata.drop('mean', axis=1)
    # print(fdata.info())
    fdata.to_csv('../data/drop_tran.csv')


def DTWDistance(s1, s2):
    DTW = {}

    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = (s1[i] - s2[j]) ** 2
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

    return math.sqrt(DTW[len(s1) - 1, len(s2) - 1])

fx,fy = 'baseline.csv','baseline_lda.csv'
lx, ly, lz = pd.read_csv(fx)[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(1, 6)]].values, \
             pd.read_csv(fy)[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(1, 6)]].values, \
             pd.read_csv('../data/comb_test.csv')[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(1, 6)]].values

x, y, z = pd.read_csv(fx)[[f'CPU_USAGE_{i}' for i in range(1, 6)]].values, \
          pd.read_csv(fy)[[f'CPU_USAGE_{i}' for i in range(1, 6)]].values, \
          pd.read_csv('../data/comb_test.csv')[[f'CPU_USAGE_{i}' for i in range(1, 6)]].values
x, y = np.append(z, x, axis=1), np.append(z, y, axis=1)
lx, ly = np.append(lz, lx, axis=1), np.append(lz, ly, axis=1)

# x = x[[f'CPU_USAGE_{i}'for i in range(1,6)]].values
# y =
x = x.astype(np.float)
y = y.astype(np.float)

print(x.dtype)
cmp = []
for i in range(x.shape[0]):
    cmp.append((DTWDistance(lx[i, -5:], ly[i, -5:]), i))
cmp.sort(reverse=True)

fig = plt.figure()
cnt = 1
pos = 5
need_out = [cmp[i][1]+1 for i in range(16*pos,16*(pos+1))]
for err, i in cmp[16*pos:16*(pos+1)]:
    plt.subplot(4, 4, cnt)
    plt.plot(range(10), x[i], 'r', label='merge')
    plt.plot(range(10), y[i], 'b', label='merege2')
    plt.title(f'error={int(err)}')
    # plt.legend()
    cnt += 1

plt.show()
print(need_out)