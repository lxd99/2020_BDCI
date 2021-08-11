import warnings

import gc

import numpy as np
import math
import pandas as pd
pd.set_option('max_columns', 100)
pd.set_option('max_rows', 100)
from tqdm.notebook import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_squared_error

import lightgbm as lgb


def run_lgb_qid(df_train, df_test, target, qid):  # 对每一个队列训练模型进行预测
    feature_names = list(
        filter(lambda x: x not in ['QUEUE_ID', 'CU', 'QUEUE_TYPE'] + [f'cpu_{i}' for i in range(1, 6)]+ [f'launching_{i}' for i in range(1, 6)],
               df_train.columns))

    # 提取 QUEUE_ID 对应的数据集
    df_train = df_train[df_train.QUEUE_ID == qid]
    df_test = df_test[df_test.QUEUE_ID == qid]

    print(f"QUEUE_ID:{qid}, target:{target}, train:{len(df_train)}, test:{len(df_test)}")

    model = lgb.LGBMRegressor(num_leaves=20,
                              max_depth=4,
                              learning_rate=0.08,
                              n_estimators=10000,  # 树的数量
                              feature_fraction=0.8,  # 每次新建一棵树时，随机使用多少的特征。
                              reg_alpha=0.5,
                              reg_lambda=0.9,
                              random_state=267)
    oof = []
    prediction = df_test[['ID', 'QUEUE_ID']]
    prediction[target] = 0

    kfold = KFold(n_splits=5, random_state=267)
    for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(df_train, df_train[target])):
        X_train = df_train.iloc[trn_idx][feature_names]
        Y_train = df_train.iloc[trn_idx][target]
        X_val = df_train.iloc[val_idx][feature_names]
        Y_val = df_train.iloc[val_idx][target]

        lgb_model = model.fit(X_train,
                              Y_train,
                              eval_names=['train', 'valid'],
                              eval_set=[(X_train, Y_train), (X_val, Y_val)],
                              verbose=0,
                              eval_metric='mse',
                              early_stopping_rounds=300)

        pred_val = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration_)
        df_oof = df_train.iloc[val_idx][[target, 'QUEUE_ID']].copy()
        df_oof['pred'] = pred_val
        oof.append(df_oof)

        pred_test = lgb_model.predict(df_test[feature_names], num_iteration=lgb_model.best_iteration_)
        prediction[target] += pred_test / kfold.n_splits

        del lgb_model, pred_val, pred_test, X_train, Y_train, X_val, Y_val
        gc.collect()

    df_oof = pd.concat(oof)
    score = mean_squared_error(df_oof[target], df_oof['pred'])
    print('MSE:', score)
    #print(prediction.head(10))
    return prediction, score

warnings.simplefilter('ignore')
train = pd.read_csv('data/train.csv')

test = pd.read_csv('evaluation_public.csv')
test = test.sort_values(by=['ID', 'DOTTING_TIME']).reset_index(drop=True)

# 这些 columns 在 test 只有单一值, 所以直接去掉

del test['STATUS']
del test['PLATFORM']
del test['RESOURCE_TYPE']


# 时间排序好后也没什么用了

del test['DOTTING_TIME']


del test['CANCELLED_JOB_NUMS']
del test['FAILED_JOB_NUMS']

df_train = pd.DataFrame()
df_train = train[['QUEUE_ID',
                   'CPU_USAGE_1', 'MEM_USAGE_1','LAUNCHING_JOB_NUMS_1','RUNNING_JOB_NUMS_1','SUCCEED_JOB_NUMS_1','DISK_USAGE_1',
                   'CPU_USAGE_2', 'MEM_USAGE_2','LAUNCHING_JOB_NUMS_2','RUNNING_JOB_NUMS_2','SUCCEED_JOB_NUMS_2','DISK_USAGE_2',
                   'CPU_USAGE_3', 'MEM_USAGE_3','LAUNCHING_JOB_NUMS_3','RUNNING_JOB_NUMS_3','SUCCEED_JOB_NUMS_3','DISK_USAGE_3',
                   'CPU_USAGE_4', 'MEM_USAGE_4','LAUNCHING_JOB_NUMS_4','RUNNING_JOB_NUMS_4','SUCCEED_JOB_NUMS_4','DISK_USAGE_4',
                   'CPU_USAGE_5', 'MEM_USAGE_5','LAUNCHING_JOB_NUMS_5','RUNNING_JOB_NUMS_5','SUCCEED_JOB_NUMS_5','DISK_USAGE_5',
                    'cpu_1', 'cpu_2', 'cpu_3', 'cpu_4', 'cpu_5','launching_1','launching_2','launching_3','launching_4','launching_5'
                ]]

df_test = pd.DataFrame()

for id_ in tqdm(test.QUEUE_ID.unique()):
    df_tmp = test[test.QUEUE_ID == id_]
    features = list()
    values = df_tmp.values
    for i, _ in enumerate(values):
        if i % 5 == 0:
            li_v = list()
            li_v.append(values[i][0])
            li_v.append(values[i][1])
            for j in range(5):
                li_v.extend(values[i+j][4:].tolist())
            features.append(li_v)
    df_feat = pd.DataFrame(features)
    df_feat.columns = ['ID', 'QUEUE_ID',
                       'CPU_USAGE_1', 'MEM_USAGE_1','LAUNCHING_JOB_NUMS_1','RUNNING_JOB_NUMS_1','SUCCEED_JOB_NUMS_1','DISK_USAGE_1',
                       'CPU_USAGE_2', 'MEM_USAGE_2','LAUNCHING_JOB_NUMS_2','RUNNING_JOB_NUMS_2','SUCCEED_JOB_NUMS_2','DISK_USAGE_2',
                       'CPU_USAGE_3', 'MEM_USAGE_3','LAUNCHING_JOB_NUMS_3','RUNNING_JOB_NUMS_3','SUCCEED_JOB_NUMS_3','DISK_USAGE_3',
                       'CPU_USAGE_4', 'MEM_USAGE_4','LAUNCHING_JOB_NUMS_4','RUNNING_JOB_NUMS_4','SUCCEED_JOB_NUMS_4','DISK_USAGE_4',
                       'CPU_USAGE_5', 'MEM_USAGE_5','LAUNCHING_JOB_NUMS_5','RUNNING_JOB_NUMS_5','SUCCEED_JOB_NUMS_5','DISK_USAGE_5',
                      ]
    df = df_feat.copy()
    #print(f'QUEUE_ID: {id_}, lines: {df.shape[0]}')
    df_test = df_test.append(df)
    #print(df_test.head(10))
## 衰减求和特征
pw09 = np.power(0.9, np.arange(5))
pw08 = np.power(0.8, np.arange(5))
pw07 = np.power(0.7, np.arange(5))
pw06 = np.power(0.6, np.arange(5))
pw05 = np.power(0.5, np.arange(5))
for col in ['CPU_USAGE', 'MEM_USAGE','DISK_USAGE']:
    for i, pw in enumerate([pw09, pw08, pw07, pw06, pw05]):
        colname = col + '_pw_' + str(9 - i)
        df_train[colname] = df_train[col+'_5'] * pw[0] + \
                            df_train[col + '_4'] * pw[1] + \
                            df_train[col + '_3'] * pw[2] + \
                            df_train[col + '_2'] * pw[3] + \
                            df_train[col + '_1'] * pw[4]

        df_test[colname] = df_test[col+'_5'] * pw[0] + \
                           df_test[col + '_4'] * pw[1] + \
                           df_test[col + '_3'] * pw[2] + \
                           df_test[col + '_2'] * pw[3] + \
                           df_test[col + '_1'] * pw[4]

# 行内统计特征
df_train['last_3_cpu_mean'] = df_train[[f'CPU_USAGE_{i}' for i in range(3,6)]].mean(axis=1)
df_train['last_3_cpu_std'] = df_train[[f'CPU_USAGE_{i}' for i in range(3,6)]].std(axis=1)
df_train['last_3_cpu_max'] = df_train[[f'CPU_USAGE_{i}' for i in range(3,6)]].max(axis=1)
df_train['last_3_cpu_min'] = df_train[[f'CPU_USAGE_{i}' for i in range(3,6)]].min(axis=1)
df_train['last_3_cpu_median'] = df_train[[f'CPU_USAGE_{i}' for i in range(3,6)]].median(axis=1)
df_train['last_4_cpu_mean'] = df_train[[f'CPU_USAGE_{i}' for i in range(2,6)]].mean(axis=1)
df_train['last_4_cpu_std'] = df_train[[f'CPU_USAGE_{i}' for i in range(2,6)]].std(axis=1)
df_train['last_4_cpu_max'] = df_train[[f'CPU_USAGE_{i}' for i in range(2,6)]].max(axis=1)
df_train['last_4_cpu_min'] = df_train[[f'CPU_USAGE_{i}' for i in range(2,6)]].min(axis=1)
df_train['last_2_cpu_median'] = df_train[[f'CPU_USAGE_{i}' for i in range(4,6)]].median(axis=1)
df_train['last_2_cpu_mean'] = df_train[[f'CPU_USAGE_{i}' for i in range(4,6)]].mean(axis=1)
df_train['last_2_cpu_std'] = df_train[[f'CPU_USAGE_{i}' for i in range(4,6)]].std(axis=1)
df_train['last_2_cpu_max'] = df_train[[f'CPU_USAGE_{i}' for i in range(4,6)]].max(axis=1)
df_train['last_2_cpu_min'] = df_train[[f'CPU_USAGE_{i}' for i in range(4,6)]].min(axis=1)
df_train['last_2_cpu_median'] = df_train[[f'CPU_USAGE_{i}' for i in range(4,6)]].median(axis=1)
df_train['cpu_mean'] = df_train[[f'CPU_USAGE_{i}' for i in range(1,6)]].mean(axis=1)
df_train['cpu_std'] = df_train[[f'CPU_USAGE_{i}' for i in range(1,6)]].std(axis=1)
df_train['cpu_diff'] = df_train['CPU_USAGE_5'] - df_train['CPU_USAGE_1']
df_train['cpu_max'] = df_train[[f'CPU_USAGE_{i}' for i in range(1,6)]].max(axis=1)
df_train['cpu_min'] = df_train[[f'CPU_USAGE_{i}' for i in range(1,6)]].min(axis=1)
df_train['cpu_median'] = df_train[[f'CPU_USAGE_{i}' for i in range(1,6)]].median(axis=1)

df_train['last_3_launch_mean'] = df_train[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(3,6)]].mean(axis=1)
df_train['last_3_launch_std'] = df_train[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(3,6)]].std(axis=1)
df_train['last_3_launch_max'] = df_train[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(3,6)]].max(axis=1)
df_train['last_3_launch_min'] = df_train[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(3,6)]].min(axis=1)
df_train['last_3_launch_median'] = df_train[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(3,6)]].median(axis=1)
df_train['last_4_launch_mean'] = df_train[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(2,6)]].mean(axis=1)
df_train['last_4_launch_std'] = df_train[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(2,6)]].std(axis=1)
df_train['last_4_launch_max'] = df_train[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(2,6)]].max(axis=1)
df_train['last_4_launch_min'] = df_train[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(2,6)]].min(axis=1)
df_train['last_4_launch_median'] = df_train[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(2,6)]].median(axis=1)
df_train['last_2_launch_mean'] = df_train[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(4,6)]].mean(axis=1)
df_train['last_2_launch_std'] = df_train[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(4,6)]].std(axis=1)
df_train['last_2_launch_max'] = df_train[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(4,6)]].max(axis=1)
df_train['last_2_launch_min'] = df_train[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(4,6)]].min(axis=1)
df_train['last_2_launch_median'] = df_train[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(4,6)]].median(axis=1)
df_train['launch_mean'] = df_train[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(1,6)]].mean(axis=1)
df_train['launch_std'] = df_train[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(1,6)]].std(axis=1)
df_train['launch_diff'] = df_train['LAUNCHING_JOB_NUMS_5'] - df_train['LAUNCHING_JOB_NUMS_1']
df_train['launch_max'] = df_train[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(1,6)]].max(axis=1)
df_train['launch_min'] = df_train[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(1,6)]].min(axis=1)
df_train['launch_median'] = df_train[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(1,6)]].median(axis=1)

df_train['last_3_running_mean'] = df_train[[f'RUNNING_JOB_NUMS_{i}' for i in range(3,6)]].mean(axis=1)
df_train['last_3_running_std'] = df_train[[f'RUNNING_JOB_NUMS_{i}' for i in range(3,6)]].std(axis=1)
df_train['last_3_running_max'] = df_train[[f'RUNNING_JOB_NUMS_{i}' for i in range(3,6)]].max(axis=1)
df_train['last_3_running_min'] = df_train[[f'RUNNING_JOB_NUMS_{i}' for i in range(3,6)]].min(axis=1)
df_train['last_3_running_median'] = df_train[[f'RUNNING_JOB_NUMS_{i}' for i in range(3,6)]].median(axis=1)
df_train['last_4_running_mean'] = df_train[[f'RUNNING_JOB_NUMS_{i}' for i in range(2,6)]].mean(axis=1)
df_train['last_4_running_std'] = df_train[[f'RUNNING_JOB_NUMS_{i}' for i in range(2,6)]].std(axis=1)
df_train['last_4_running_max'] = df_train[[f'RUNNING_JOB_NUMS_{i}' for i in range(2,6)]].max(axis=1)
df_train['last_4_running_min'] = df_train[[f'RUNNING_JOB_NUMS_{i}' for i in range(2,6)]].min(axis=1)
df_train['last_4_running_median'] = df_train[[f'RUNNING_JOB_NUMS_{i}' for i in range(2,6)]].median(axis=1)
df_train['last_2_running_mean'] = df_train[[f'RUNNING_JOB_NUMS_{i}' for i in range(4,6)]].mean(axis=1)
df_train['last_2_running_std'] = df_train[[f'RUNNING_JOB_NUMS_{i}' for i in range(4,6)]].std(axis=1)
df_train['last_2_running_max'] = df_train[[f'RUNNING_JOB_NUMS_{i}' for i in range(4,6)]].max(axis=1)
df_train['last_2_running_min'] = df_train[[f'RUNNING_JOB_NUMS_{i}' for i in range(4,6)]].min(axis=1)
df_train['last_2_running_median'] = df_train[[f'RUNNING_JOB_NUMS_{i}' for i in range(4,6)]].median(axis=1)
df_train['running_mean'] = df_train[[f'RUNNING_JOB_NUMS_{i}' for i in range(1,6)]].mean(axis=1)
df_train['running_std'] = df_train[[f'RUNNING_JOB_NUMS_{i}' for i in range(1,6)]].std(axis=1)
df_train['running_diff'] = df_train['RUNNING_JOB_NUMS_5'] - df_train['RUNNING_JOB_NUMS_1']
df_train['running_max'] = df_train[[f'RUNNING_JOB_NUMS_{i}' for i in range(1,6)]].max(axis=1)
df_train['running_min'] = df_train[[f'RUNNING_JOB_NUMS_{i}' for i in range(1,6)]].min(axis=1)
df_train['running_median'] = df_train[[f'RUNNING_JOB_NUMS_{i}' for i in range(1,6)]].median(axis=1)

df_train['last_3_succeed_mean'] = df_train[[f'SUCCEED_JOB_NUMS_{i}' for i in range(3,6)]].mean(axis=1)
df_train['last_3_succeed_std'] = df_train[[f'SUCCEED_JOB_NUMS_{i}' for i in range(3,6)]].std(axis=1)
df_train['last_3_succeed_max'] = df_train[[f'SUCCEED_JOB_NUMS_{i}' for i in range(3,6)]].max(axis=1)
df_train['last_3_succeed_min'] = df_train[[f'SUCCEED_JOB_NUMS_{i}' for i in range(3,6)]].min(axis=1)
df_train['last_3_succeed_median'] = df_train[[f'SUCCEED_JOB_NUMS_{i}' for i in range(3,6)]].median(axis=1)
df_train['last_4_succeed_mean'] = df_train[[f'SUCCEED_JOB_NUMS_{i}' for i in range(2,6)]].mean(axis=1)
df_train['last_4_succeed_std'] = df_train[[f'SUCCEED_JOB_NUMS_{i}' for i in range(2,6)]].std(axis=1)
df_train['last_4_succeed_max'] = df_train[[f'SUCCEED_JOB_NUMS_{i}' for i in range(2,6)]].max(axis=1)
df_train['last_4_succeed_min'] = df_train[[f'SUCCEED_JOB_NUMS_{i}' for i in range(2,6)]].min(axis=1)
df_train['last_4_succeed_median'] = df_train[[f'SUCCEED_JOB_NUMS_{i}' for i in range(2,6)]].median(axis=1)
df_train['last_2_succeed_mean'] = df_train[[f'SUCCEED_JOB_NUMS_{i}' for i in range(4,6)]].mean(axis=1)
df_train['last_2_succeed_std'] = df_train[[f'SUCCEED_JOB_NUMS_{i}' for i in range(4,6)]].std(axis=1)
df_train['last_2_succeed_max'] = df_train[[f'SUCCEED_JOB_NUMS_{i}' for i in range(4,6)]].max(axis=1)
df_train['last_2_succeed_min'] = df_train[[f'SUCCEED_JOB_NUMS_{i}' for i in range(4,6)]].min(axis=1)
df_train['last_2_succeed_median'] = df_train[[f'SUCCEED_JOB_NUMS_{i}' for i in range(4,6)]].median(axis=1)
df_train['succeed_mean'] = df_train[[f'SUCCEED_JOB_NUMS_{i}' for i in range(1,6)]].mean(axis=1)
df_train['succeed_std'] = df_train[[f'SUCCEED_JOB_NUMS_{i}' for i in range(1,6)]].std(axis=1)
df_train['succeed_diff'] = df_train['SUCCEED_JOB_NUMS_5'] - df_train['SUCCEED_JOB_NUMS_1']
df_train['succeed_max'] = df_train[[f'SUCCEED_JOB_NUMS_{i}' for i in range(1,6)]].max(axis=1)
df_train['succeed_min'] = df_train[[f'SUCCEED_JOB_NUMS_{i}' for i in range(1,6)]].min(axis=1)
df_train['succeed_median'] = df_train[[f'SUCCEED_JOB_NUMS_{i}' for i in range(1,6)]].median(axis=1)

df_train['last_3_mem_mean'] = df_train[[f'MEM_USAGE_{i}' for i in range(3,6)]].mean(axis=1)
df_train['last_3_mem_std'] = df_train[[f'MEM_USAGE_{i}' for i in range(3,6)]].std(axis=1)
df_train['last_3_mem_max'] = df_train[[f'MEM_USAGE_{i}' for i in range(3,6)]].max(axis=1)
df_train['last_3_mem_min'] = df_train[[f'MEM_USAGE_{i}' for i in range(3,6)]].min(axis=1)
df_train['last_3_mem_median'] = df_train[[f'MEM_USAGE_{i}' for i in range(3,6)]].median(axis=1)
df_train['last_4_mem_mean'] = df_train[[f'MEM_USAGE_{i}' for i in range(2,6)]].mean(axis=1)
df_train['last_4_mem_std'] = df_train[[f'MEM_USAGE_{i}' for i in range(2,6)]].std(axis=1)
df_train['last_4_mem_max'] = df_train[[f'MEM_USAGE_{i}' for i in range(2,6)]].max(axis=1)
df_train['last_4_mem_min'] = df_train[[f'MEM_USAGE_{i}' for i in range(2,6)]].min(axis=1)
df_train['last_4_mem_median'] = df_train[[f'MEM_USAGE_{i}' for i in range(2,6)]].median(axis=1)
df_train['last_2_mem_mean'] = df_train[[f'MEM_USAGE_{i}' for i in range(4,6)]].mean(axis=1)
df_train['last_2_mem_std'] = df_train[[f'MEM_USAGE_{i}' for i in range(4,6)]].std(axis=1)
df_train['last_2_mem_max'] = df_train[[f'MEM_USAGE_{i}' for i in range(4,6)]].max(axis=1)
df_train['last_2_mem_min'] = df_train[[f'MEM_USAGE_{i}' for i in range(4,6)]].min(axis=1)
df_train['last_2_mem_median'] = df_train[[f'MEM_USAGE_{i}' for i in range(4,6)]].median(axis=1)
df_train['mem_mean'] = df_train[[f'MEM_USAGE_{i}' for i in range(1,6)]].mean(axis=1)
df_train['mem_std'] = df_train[[f'MEM_USAGE_{i}' for i in range(1,6)]].std(axis=1)
df_train['mem_diff'] = df_train['MEM_USAGE_5'] - df_train['MEM_USAGE_1']
df_train['mem_max'] = df_train[[f'MEM_USAGE_{i}' for i in range(1,6)]].max(axis=1)
df_train['mem_min'] = df_train[[f'MEM_USAGE_{i}' for i in range(1,6)]].min(axis=1)
df_train['mem_median'] = df_train[[f'MEM_USAGE_{i}' for i in range(1,6)]].median(axis=1)

df_train['last_3_disk_mean'] = df_train[[f'DISK_USAGE_{i}' for i in range(3,6)]].mean(axis=1)
df_train['last_3_disk_std'] = df_train[[f'DISK_USAGE_{i}' for i in range(3,6)]].std(axis=1)
df_train['last_3_disk_max'] = df_train[[f'DISK_USAGE_{i}' for i in range(3,6)]].max(axis=1)
df_train['last_3_disk_min'] = df_train[[f'DISK_USAGE_{i}' for i in range(3,6)]].min(axis=1)
df_train['last_3_disk_median'] = df_train[[f'DISK_USAGE_{i}' for i in range(3,6)]].median(axis=1)
df_train['last_4_disk_mean'] = df_train[[f'DISK_USAGE_{i}' for i in range(2,6)]].mean(axis=1)
df_train['last_4_disk_std'] = df_train[[f'DISK_USAGE_{i}' for i in range(2,6)]].std(axis=1)
df_train['last_4_disk_max'] = df_train[[f'DISK_USAGE_{i}' for i in range(2,6)]].max(axis=1)
df_train['last_4_disk_min'] = df_train[[f'DISK_USAGE_{i}' for i in range(2,6)]].min(axis=1)
df_train['last_4_disk_median'] = df_train[[f'DISK_USAGE_{i}' for i in range(2,6)]].median(axis=1)
df_train['last_2_disk_mean'] = df_train[[f'DISK_USAGE_{i}' for i in range(4,6)]].mean(axis=1)
df_train['last_2_disk_std'] = df_train[[f'DISK_USAGE_{i}' for i in range(4,6)]].std(axis=1)
df_train['last_2_disk_max'] = df_train[[f'DISK_USAGE_{i}' for i in range(4,6)]].max(axis=1)
df_train['last_2_disk_min'] = df_train[[f'DISK_USAGE_{i}' for i in range(4,6)]].min(axis=1)
df_train['last_2_disk_median'] = df_train[[f'DISK_USAGE_{i}' for i in range(4,6)]].median(axis=1)
df_train['disk_mean'] = df_train[[f'DISK_USAGE_{i}' for i in range(1,6)]].mean(axis=1)
df_train['disk_std'] = df_train[[f'DISK_USAGE_{i}' for i in range(1,6)]].std(axis=1)
df_train['disk_diff'] = df_train['DISK_USAGE_5'] - df_train['DISK_USAGE_1']
df_train['disk_max'] = df_train[[f'DISK_USAGE_{i}' for i in range(1,6)]].max(axis=1)
df_train['disk_min'] = df_train[[f'DISK_USAGE_{i}' for i in range(1,6)]].min(axis=1)
df_train['disk_median'] = df_train[[f'DISK_USAGE_{i}' for i in range(1,6)]].median(axis=1)


df_test['last_3_cpu_mean'] = df_test[[f'CPU_USAGE_{i}' for i in range(3,6)]].mean(axis=1)
df_test['last_3_cpu_std'] = df_test[[f'CPU_USAGE_{i}' for i in range(3,6)]].std(axis=1)
df_test['last_3_cpu_max'] = df_test[[f'CPU_USAGE_{i}' for i in range(3,6)]].max(axis=1)
df_test['last_3_cpu_min'] = df_test[[f'CPU_USAGE_{i}' for i in range(3,6)]].min(axis=1)
df_test['last_3_cpu_median'] = df_test[[f'CPU_USAGE_{i}' for i in range(3,6)]].median(axis=1)
df_test['last_4_cpu_mean'] = df_test[[f'CPU_USAGE_{i}' for i in range(2,6)]].mean(axis=1)
df_test['last_4_cpu_std'] = df_test[[f'CPU_USAGE_{i}' for i in range(2,6)]].std(axis=1)
df_test['last_4_cpu_max'] = df_test[[f'CPU_USAGE_{i}' for i in range(2,6)]].max(axis=1)
df_test['last_4_cpu_min'] = df_test[[f'CPU_USAGE_{i}' for i in range(2,6)]].min(axis=1)
df_test['last_4_cpu_median'] = df_test[[f'CPU_USAGE_{i}' for i in range(2,6)]].median(axis=1)
df_test['last_2_cpu_mean'] = df_test[[f'CPU_USAGE_{i}' for i in range(4,6)]].mean(axis=1)
df_test['last_2_cpu_std'] = df_test[[f'CPU_USAGE_{i}' for i in range(4,6)]].std(axis=1)
df_test['last_2_cpu_max'] = df_test[[f'CPU_USAGE_{i}' for i in range(4,6)]].max(axis=1)
df_test['last_2_cpu_min'] = df_test[[f'CPU_USAGE_{i}' for i in range(4,6)]].min(axis=1)
df_test['last_2_cpu_median'] = df_test[[f'CPU_USAGE_{i}' for i in range(4,6)]].median(axis=1)
df_test['cpu_mean'] = df_test[[f'CPU_USAGE_{i}' for i in range(1,6)]].mean(axis=1)
df_test['cpu_std'] = df_test[[f'CPU_USAGE_{i}' for i in range(1,6)]].std(axis=1)
df_test['cpu_diff'] = df_test['CPU_USAGE_5'] - df_test['CPU_USAGE_1']
df_test['cpu_max'] = df_test[[f'CPU_USAGE_{i}' for i in range(1,6)]].max(axis=1)
df_test['cpu_min'] = df_test[[f'CPU_USAGE_{i}' for i in range(1,6)]].min(axis=1)
df_test['cpu_median'] = df_test[[f'CPU_USAGE_{i}' for i in range(1,6)]].median(axis=1)

df_test['last_3_launch_mean'] = df_test[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(3,6)]].mean(axis=1)
df_test['last_3_launch_std'] = df_test[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(3,6)]].std(axis=1)
df_test['last_3_launch_max'] = df_test[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(3,6)]].max(axis=1)
df_test['last_3_launch_min'] = df_test[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(3,6)]].min(axis=1)
df_test['last_3_launch_median'] = df_test[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(3,6)]].median(axis=1)
df_test['last_4_launch_mean'] = df_test[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(2,6)]].mean(axis=1)
df_test['last_4_launch_std'] = df_test[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(2,6)]].std(axis=1)
df_test['last_4_launch_max'] = df_test[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(2,6)]].max(axis=1)
df_test['last_4_launch_min'] = df_test[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(2,6)]].min(axis=1)
df_test['last_4_launch_median'] = df_test[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(2,6)]].median(axis=1)
df_test['last_2_launch_mean'] = df_test[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(4,6)]].mean(axis=1)
df_test['last_2_launch_std'] = df_test[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(4,6)]].std(axis=1)
df_test['last_2_launch_max'] = df_test[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(4,6)]].max(axis=1)
df_test['last_2_launch_min'] = df_test[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(4,6)]].min(axis=1)
df_test['last_2_launch_median'] = df_test[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(4,6)]].median(axis=1)
df_test['launch_mean'] = df_test[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(1,6)]].mean(axis=1)
df_test['launch_std'] = df_test[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(1,6)]].std(axis=1)
df_test['launch_diff'] = df_test['LAUNCHING_JOB_NUMS_5'] - df_test['LAUNCHING_JOB_NUMS_1']
df_test['launch_max'] = df_test[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(1,6)]].max(axis=1)
df_test['launch_min'] = df_test[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(1,6)]].min(axis=1)
df_test['launch_median'] = df_test[[f'LAUNCHING_JOB_NUMS_{i}' for i in range(1,6)]].median(axis=1)

df_test['last_3_running_mean'] = df_test[[f'RUNNING_JOB_NUMS_{i}' for i in range(3,6)]].mean(axis=1)
df_test['last_3_running_std'] = df_test[[f'RUNNING_JOB_NUMS_{i}' for i in range(3,6)]].std(axis=1)
df_test['last_3_running_max'] = df_test[[f'RUNNING_JOB_NUMS_{i}' for i in range(3,6)]].max(axis=1)
df_test['last_3_running_min'] = df_test[[f'RUNNING_JOB_NUMS_{i}' for i in range(3,6)]].min(axis=1)
df_test['last_3_running_median'] = df_test[[f'RUNNING_JOB_NUMS_{i}' for i in range(3,6)]].median(axis=1)
df_test['last_4_running_mean'] = df_test[[f'RUNNING_JOB_NUMS_{i}' for i in range(2,6)]].mean(axis=1)
df_test['last_4_running_std'] = df_test[[f'RUNNING_JOB_NUMS_{i}' for i in range(2,6)]].std(axis=1)
df_test['last_4_running_max'] = df_test[[f'RUNNING_JOB_NUMS_{i}' for i in range(2,6)]].max(axis=1)
df_test['last_4_running_min'] = df_test[[f'RUNNING_JOB_NUMS_{i}' for i in range(2,6)]].min(axis=1)
df_test['last_4_running_median'] = df_test[[f'RUNNING_JOB_NUMS_{i}' for i in range(2,6)]].median(axis=1)
df_test['last_2_running_mean'] = df_test[[f'RUNNING_JOB_NUMS_{i}' for i in range(4,6)]].mean(axis=1)
df_test['last_2_running_std'] = df_test[[f'RUNNING_JOB_NUMS_{i}' for i in range(4,6)]].std(axis=1)
df_test['last_2_running_max'] = df_test[[f'RUNNING_JOB_NUMS_{i}' for i in range(4,6)]].max(axis=1)
df_test['last_2_running_min'] = df_test[[f'RUNNING_JOB_NUMS_{i}' for i in range(4,6)]].min(axis=1)
df_test['last_2_running_median'] = df_test[[f'RUNNING_JOB_NUMS_{i}' for i in range(4,6)]].median(axis=1)
df_test['running_mean'] = df_test[[f'RUNNING_JOB_NUMS_{i}' for i in range(1,6)]].mean(axis=1)
df_test['running_std'] = df_test[[f'RUNNING_JOB_NUMS_{i}' for i in range(1,6)]].std(axis=1)
df_test['running_diff'] = df_test['RUNNING_JOB_NUMS_5'] - df_test['RUNNING_JOB_NUMS_1']
df_test['running_max'] = df_test[[f'RUNNING_JOB_NUMS_{i}' for i in range(1,6)]].max(axis=1)
df_test['running_min'] = df_test[[f'RUNNING_JOB_NUMS_{i}' for i in range(1,6)]].min(axis=1)
df_test['running_median'] = df_test[[f'RUNNING_JOB_NUMS_{i}' for i in range(1,6)]].median(axis=1)

df_test['last_3_succeed_mean'] = df_test[[f'SUCCEED_JOB_NUMS_{i}' for i in range(3,6)]].mean(axis=1)
df_test['last_3_succeed_std'] = df_test[[f'SUCCEED_JOB_NUMS_{i}' for i in range(3,6)]].std(axis=1)
df_test['last_3_succeed_max'] = df_test[[f'SUCCEED_JOB_NUMS_{i}' for i in range(3,6)]].max(axis=1)
df_test['last_3_succeed_min'] = df_test[[f'SUCCEED_JOB_NUMS_{i}' for i in range(3,6)]].min(axis=1)
df_test['last_3_succeed_median'] = df_test[[f'SUCCEED_JOB_NUMS_{i}' for i in range(3,6)]].median(axis=1)
df_test['last_4_succeed_mean'] = df_test[[f'SUCCEED_JOB_NUMS_{i}' for i in range(2,6)]].mean(axis=1)
df_test['last_4_succeed_std'] = df_test[[f'SUCCEED_JOB_NUMS_{i}' for i in range(2,6)]].std(axis=1)
df_test['last_4_succeed_max'] = df_test[[f'SUCCEED_JOB_NUMS_{i}' for i in range(2,6)]].max(axis=1)
df_test['last_4_succeed_min'] = df_test[[f'SUCCEED_JOB_NUMS_{i}' for i in range(2,6)]].min(axis=1)
df_test['last_4_succeed_median'] = df_test[[f'SUCCEED_JOB_NUMS_{i}' for i in range(2,6)]].median(axis=1)
df_test['last_2_succeed_mean'] = df_test[[f'SUCCEED_JOB_NUMS_{i}' for i in range(4,6)]].mean(axis=1)
df_test['last_2_succeed_std'] = df_test[[f'SUCCEED_JOB_NUMS_{i}' for i in range(4,6)]].std(axis=1)
df_test['last_2_succeed_max'] = df_test[[f'SUCCEED_JOB_NUMS_{i}' for i in range(4,6)]].max(axis=1)
df_test['last_2_succeed_min'] = df_test[[f'SUCCEED_JOB_NUMS_{i}' for i in range(4,6)]].min(axis=1)
df_test['last_2_succeed_median'] = df_test[[f'SUCCEED_JOB_NUMS_{i}' for i in range(4,6)]].median(axis=1)
df_test['succeed_mean'] = df_test[[f'SUCCEED_JOB_NUMS_{i}' for i in range(1,6)]].mean(axis=1)
df_test['succeed_std'] = df_test[[f'SUCCEED_JOB_NUMS_{i}' for i in range(1,6)]].std(axis=1)
df_test['succeed_diff'] = df_test['SUCCEED_JOB_NUMS_5'] - df_test['SUCCEED_JOB_NUMS_1']
df_test['succeed_max'] = df_test[[f'SUCCEED_JOB_NUMS_{i}' for i in range(1,6)]].max(axis=1)
df_test['succeed_min'] = df_test[[f'SUCCEED_JOB_NUMS_{i}' for i in range(1,6)]].min(axis=1)
df_test['succeed_median'] = df_test[[f'SUCCEED_JOB_NUMS_{i}' for i in range(1,6)]].median(axis=1)

df_test['last_3_mem_mean'] = df_test[[f'MEM_USAGE_{i}' for i in range(3,6)]].mean(axis=1)
df_test['last_3_mem_std'] = df_test[[f'MEM_USAGE_{i}' for i in range(3,6)]].std(axis=1)
df_test['last_3_mem_max'] = df_test[[f'MEM_USAGE_{i}' for i in range(3,6)]].max(axis=1)
df_test['last_3_mem_min'] = df_test[[f'MEM_USAGE_{i}' for i in range(3,6)]].min(axis=1)
df_test['last_3_mem_median'] = df_test[[f'MEM_USAGE_{i}' for i in range(3,6)]].median(axis=1)
df_test['last_4_mem_mean'] = df_test[[f'MEM_USAGE_{i}' for i in range(2,6)]].mean(axis=1)
df_test['last_4_mem_std'] = df_test[[f'MEM_USAGE_{i}' for i in range(2,6)]].std(axis=1)
df_test['last_4_mem_max'] = df_test[[f'MEM_USAGE_{i}' for i in range(2,6)]].max(axis=1)
df_test['last_4_mem_min'] = df_test[[f'MEM_USAGE_{i}' for i in range(2,6)]].min(axis=1)
df_test['last_4_mem_median'] = df_test[[f'MEM_USAGE_{i}' for i in range(2,6)]].median(axis=1)
df_test['last_2_mem_mean'] = df_test[[f'MEM_USAGE_{i}' for i in range(4,6)]].mean(axis=1)
df_test['last_2_mem_std'] = df_test[[f'MEM_USAGE_{i}' for i in range(4,6)]].std(axis=1)
df_test['last_2_mem_max'] = df_test[[f'MEM_USAGE_{i}' for i in range(4,6)]].max(axis=1)
df_test['last_2_mem_min'] = df_test[[f'MEM_USAGE_{i}' for i in range(4,6)]].min(axis=1)
df_test['last_2_mem_median'] = df_test[[f'MEM_USAGE_{i}' for i in range(4,6)]].median(axis=1)
df_test['mem_mean'] = df_test[[f'MEM_USAGE_{i}' for i in range(1,6)]].mean(axis=1)
df_test['mem_std'] = df_test[[f'MEM_USAGE_{i}' for i in range(1,6)]].std(axis=1)
df_test['mem_diff'] = df_test['MEM_USAGE_5'] - df_test['MEM_USAGE_1']
df_test['mem_max'] = df_test[[f'MEM_USAGE_{i}' for i in range(1,6)]].max(axis=1)
df_test['mem_min'] = df_test[[f'MEM_USAGE_{i}' for i in range(1,6)]].min(axis=1)
df_test['mem_median'] = df_test[[f'MEM_USAGE_{i}' for i in range(1,6)]].median(axis=1)

df_test['last_3_disk_mean'] = df_test[[f'DISK_USAGE_{i}' for i in range(3,6)]].mean(axis=1)
df_test['last_3_disk_std'] = df_test[[f'DISK_USAGE_{i}' for i in range(3,6)]].std(axis=1)
df_test['last_3_disk_max'] = df_test[[f'DISK_USAGE_{i}' for i in range(3,6)]].max(axis=1)
df_test['last_3_disk_min'] = df_test[[f'DISK_USAGE_{i}' for i in range(3,6)]].min(axis=1)
df_test['last_3_disk_median'] = df_test[[f'DISK_USAGE_{i}' for i in range(3,6)]].median(axis=1)
df_test['last_4_disk_mean'] = df_test[[f'DISK_USAGE_{i}' for i in range(2,6)]].mean(axis=1)
df_test['last_4_disk_std'] = df_test[[f'DISK_USAGE_{i}' for i in range(2,6)]].std(axis=1)
df_test['last_4_disk_max'] = df_test[[f'DISK_USAGE_{i}' for i in range(2,6)]].max(axis=1)
df_test['last_4_disk_min'] = df_test[[f'DISK_USAGE_{i}' for i in range(2,6)]].min(axis=1)
df_test['last_4_disk_median'] = df_test[[f'DISK_USAGE_{i}' for i in range(2,6)]].median(axis=1)
df_test['last_2_disk_mean'] = df_test[[f'DISK_USAGE_{i}' for i in range(4,6)]].mean(axis=1)
df_test['last_2_disk_std'] = df_test[[f'DISK_USAGE_{i}' for i in range(4,6)]].std(axis=1)
df_test['last_2_disk_max'] = df_test[[f'DISK_USAGE_{i}' for i in range(4,6)]].max(axis=1)
df_test['last_2_disk_min'] = df_test[[f'DISK_USAGE_{i}' for i in range(4,6)]].min(axis=1)
df_test['last_2_disk_median'] = df_test[[f'DISK_USAGE_{i}' for i in range(4,6)]].median(axis=1)
df_test['disk_mean'] = df_test[[f'DISK_USAGE_{i}' for i in range(1,6)]].mean(axis=1)
df_test['disk_std'] = df_test[[f'DISK_USAGE_{i}' for i in range(1,6)]].std(axis=1)
df_test['disk_diff'] = df_test['DISK_USAGE_5'] - df_test['DISK_USAGE_1']
df_test['disk_max'] = df_test[[f'DISK_USAGE_{i}' for i in range(1,6)]].max(axis=1)
df_test['disk_min'] = df_test[[f'DISK_USAGE_{i}' for i in range(1,6)]].min(axis=1)
df_test['disk_median'] = df_test[[f'DISK_USAGE_{i}' for i in range(1,6)]].median(axis=1)

predictions = list()
scores = list()

for qid in tqdm(test.QUEUE_ID.unique()):
    df = pd.DataFrame()
    for t in [f'cpu_{i}' for i in range(1,6)]:
        prediction, score = run_lgb_qid(df_train, df_test, target=t, qid=qid)
        if t == 'cpu_1':
            df = prediction.copy()
        else:
            df = pd.merge(df, prediction, on=['ID', 'QUEUE_ID'], how='left')
        scores.append(score)
    # print(df.head(10))
    predictions.append(df)
print('mean MSE score: ', np.mean(scores))
sub = pd.concat(predictions)
sub = sub.sort_values(by='ID').reset_index(drop=True)
sub.drop(['QUEUE_ID'], axis=1, inplace=True)
sub.columns = ['ID'] + ['CPU_USAGE_1']+ ['CPU_USAGE_2']+ \
              ['CPU_USAGE_3']+\
              ['CPU_USAGE_4']+ ['CPU_USAGE_5']

predictions = list()
scores = list()

for qid in tqdm(test.QUEUE_ID.unique()):
    df = pd.DataFrame()
    for t in [f'launching_{i}' for i in range(1,6)]:
        prediction, score = run_lgb_qid(df_train, df_test, target=t, qid=qid)
        if t == 'launching_1':
            df = prediction.copy()
        else:
            df = pd.merge(df, prediction, on=['ID', 'QUEUE_ID'], how='left')
        scores.append(score)
    # print(df.head(10))
    predictions.append(df)
print('mean MSE score: ', np.mean(scores))
sub2 = pd.concat(predictions)
sub2 = sub2.sort_values(by='ID').reset_index(drop=True)
sub2.drop(['QUEUE_ID'], axis=1, inplace=True)
sub2.drop(['ID'], axis=1, inplace=True)
sub2.columns = ['LAUNCHING_JOB_NUMS_1']+['LAUNCHING_JOB_NUMS_2']+ \
              ['LAUNCHING_JOB_NUMS_3']+['LAUNCHING_JOB_NUMS_4']+['LAUNCHING_JOB_NUMS_5']
# 全置 0 都比训练出来的结果好
for col in [f'LAUNCHING_JOB_NUMS_{i}' for i in range(1, 6)]:
    sub[col] = sub2[col]
# print(sub.head(10))
sub = sub[['ID',
           'CPU_USAGE_1', 'LAUNCHING_JOB_NUMS_1',
           'CPU_USAGE_2', 'LAUNCHING_JOB_NUMS_2',
           'CPU_USAGE_3', 'LAUNCHING_JOB_NUMS_3',
           'CPU_USAGE_4', 'LAUNCHING_JOB_NUMS_4',
           'CPU_USAGE_5', 'LAUNCHING_JOB_NUMS_5']]

print(sub.shape)

# 注意: 提交要求预测结果需为非负整数, 包括 ID 也需要是整数

sub['ID'] = sub['ID'].astype(int)

for col in [i for i in sub.columns if i != 'ID']:
    sub[col] = sub[col].apply(np.floor)
    sub[col] = sub[col].apply(lambda x: 0 if x < 0 else x)
    sub[col] = sub[col].astype(int)
sub.to_csv('baseline2_4_2.csv', index=False)