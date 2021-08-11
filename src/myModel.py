import warnings

warnings.simplefilter('ignore')
import gc

import numpy as np
import pandas as pd

pd.set_option('max_columns', 100)
pd.set_option('max_rows', 100)
from tqdm.notebook import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
import lightgbm as lgb
import utils

drops = ['STATUS', 'PLATFORM', 'RESOURCE_TYPE', 'QUEUE_TYPE']
df_train, df_test = pd.read_csv('../data/train.csv'), pd.read_csv('../data/test.csv')


df_train['cpu_mean'] = df_train[[f'CPU_USAGE_{i}' for i in range(1, 6)]].mean(axis=1)
df_train['cpu_std'] = df_train[[f'CPU_USAGE_{i}' for i in range(1, 6)]].std(axis=1)
for i in range(1, 5):
    df_train[f'cpu_diff_{i}'] = (df_train[f'CPU_USAGE_{i + 1}'] - df_train[f'CPU_USAGE_{i}']) * 1.1 ** i
    df_train[f'mem_diff_{i}'] = df_train[f'MEM_USAGE_{i + 1}'] - df_train[f'MEM_USAGE_{i}'] * 1.1 ** i
df_train['cpu_max'] = df_train[[f'CPU_USAGE_{i}' for i in range(1, 6)]].max(axis=1)
df_train['mem_mean'] = df_train[[f'MEM_USAGE_{i}' for i in range(1, 6)]].mean(axis=1)
df_train['mem_std'] = df_train[[f'MEM_USAGE_{i}' for i in range(1, 6)]].std(axis=1)
df_train['mem_max'] = df_train[[f'MEM_USAGE_{i}' for i in range(1, 6)]].max(axis=1)

df_test['cpu_mean'] = df_test[[f'CPU_USAGE_{i}' for i in range(1, 6)]].mean(axis=1)
df_test['cpu_std'] = df_test[[f'CPU_USAGE_{i}' for i in range(1, 6)]].std(axis=1)
for i in range(1, 5):
    df_test[f'cpu_diff_{i}'] = df_test[f'CPU_USAGE_{i + 1}'] - df_test[f'CPU_USAGE_{i}'] * 1.1 ** i
    df_test[f'mem_diff_{i}'] = df_test[f'MEM_USAGE_{i + 1}'] - df_test[f'MEM_USAGE_{i}'] * 1.1 ** i
df_test['cpu_max'] = df_test[[f'CPU_USAGE_{i}' for i in range(1, 6)]].max(axis=1)
df_test['mem_mean'] = df_test[[f'MEM_USAGE_{i}' for i in range(1, 6)]].mean(axis=1)
df_test['mem_std'] = df_test[[f'MEM_USAGE_{i}' for i in range(1, 6)]].std(axis=1)
df_test['mem_max'] = df_test[[f'MEM_USAGE_{i}' for i in range(1, 6)]].max(axis=1)

mas_id = []
min_id = []
mid_id = []


def run_lgb_qid(df_train, df_test, target, qid):
    """
    multiregression: join ways change && loss_fuction_define
    """
    feature_names = list(
        filter(lambda x: x not in ['QUEUE_ID', 'CU', 'QUEUE_TYPE'] + [f'cpu_{i}' for i in range(1, 6)]
                         + [f'launching_{i}' for i in range(1, 6)],
               df_train.columns))

    # 提取 QUEUE_ID 对应的数据集
    df_train = df_train[df_train.QUEUE_ID == qid]
    df_test = df_test[df_test.QUEUE_ID == qid]

    print(f"QUEUE_ID:{qid}, target:{target}, train:{len(df_train)}, test:{len(df_test)}")

    model = lgb.LGBMRegressor(num_leaves=20,
                              max_depth=4,
                              learning_rate=0.08,
                              n_estimators=15000,
                              subsample=0.9,
                              feature_fraction=0.8,
                              reg_alpha=0.6,
                              reg_lambda=1.2,
                              random_state=42)
    oof = []
    prediction = df_test[['ID', 'QUEUE_ID']]
    prediction[target] = 0
    if df_train.shape[0] < 5 or df_test.shape[0] < 5:
        return prediction, np.array([0])
    kfold = KFold(n_splits=5, random_state=42)
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
                              early_stopping_rounds=20)

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

    return prediction, score


predictions = list()
scores = list()

for qid in tqdm(df_test.QUEUE_ID.unique()):
    df = pd.DataFrame()
    for t in [f'cpu_{i}' for i in range(1, 6)]:
        prediction, score = run_lgb_qid(df_train, df_test, target=t, qid=qid)
        if t == 'cpu_1':
            df = prediction.copy()
        else:
            df = pd.merge(df, prediction, on=['ID', 'QUEUE_ID'], how='left')
        scores.append(score)
    if np.mean(scores[-5:]) > 100:
        mas_id.append(qid)
    elif np.mean(scores[-5:]) < 1:
        min_id.append(qid)
    else:
        mid_id.append(qid)
    predictions.append(df)
print('mean MSE score: ', np.mean(scores))

sub = pd.concat(predictions)

sub = sub.sort_values(by='ID').reset_index(drop=True)
sub.drop(['QUEUE_ID'], axis=1, inplace=True)
#
# # 全置 0 都比训练出来的结果好
for col in [f'launching_{i}' for i in range(1, 6)]:
    sub[col] = 0

print(min_id)
print(mid_id)
print(mas_id)
utils.deal_and_save(sub)
