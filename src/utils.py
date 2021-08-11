from operator import itemgetter

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

mi = 3 * 1000 * 60
ma = 7 * 1000 * 60

pred_colum = ['CPU_USAGE_6', 'LAUNCHING_JOB_NUMS_6',
              'CPU_USAGE_7', 'LAUNCHING_JOB_NUMS_7',
              'CPU_USAGE_8', 'LAUNCHING_JOB_NUMS_8',
              'CPU_USAGE_9', 'LAUNCHING_JOB_NUMS_9',
              'CPU_USAGE_10', 'LAUNCHING_JOB_NUMS_10']

def myPCA(data):
    X = data.values
    sklearn_pca = PCA(n_components=5)
    mdata = sklearn_pca.fit_transform(X)
    for i in range(1,6):
        data[f'x_{i}'] = mdata[:,i-1]
    return sklearn_pca.fit_transform(X)
def myLDA(train,test):
    LDA(n_components=2)
    pass
def myread(path):
    if path[-3:] == "csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    return df

def digitalization(fields,train_df,test_df):
    """将非数值型域转换为数值型."""
    # 组合训练集和测试集，只用来构建编码器，不用来训练模型
    df = pd.concat([train_df[fields], test_df[fields]], ignore_index=True)

    for f in fields:
        # 构建编码器
        le = LabelEncoder()
        le.fit(df[f])

        # 设置新值
        train_df[f] = le.transform(train_df[f])
        test_df[f] = le.transform(test_df[f])
        print('%s:' % f, le.classes_)

def readData(path,need):
    """
    缺少均值,缺少降维
    """
    df = myread(path)
    feature = ['CU','CPU_USAGE','MEM_USAGE',"LAUNCHING_JOB_NUMS",
               "RUNNING_JOB_NUMS","SUCCEED_JOB_NUMS","CANCELLED_JOB_NUMS",
               "FAILED_JOB_NUMS","DISK_USAGE"]
    feature = need + [f'{j}_{i}' for i in range (1,6) for j in feature]
    data = df[feature]
    myPCA(data)
    return data
def deal_and_save(prediction):
    prediction['ID'] = prediction['ID'].astype(int)
    prediction = prediction[['ID',
                             'cpu_1', 'launching_1',
                             'cpu_2', 'launching_2',
                             'cpu_3', 'launching_3',
                             'cpu_4', 'launching_4',
                             'cpu_5', 'launching_5']]

    prediction.columns = ['ID',
                          'CPU_USAGE_1', 'LAUNCHING_JOB_NUMS_1',
                          'CPU_USAGE_2', 'LAUNCHING_JOB_NUMS_2',
                          'CPU_USAGE_3', 'LAUNCHING_JOB_NUMS_3',
                          'CPU_USAGE_4', 'LAUNCHING_JOB_NUMS_4',
                          'CPU_USAGE_5', 'LAUNCHING_JOB_NUMS_5']
    for col in [i for i in prediction.columns if i != 'ID']:
        prediction[col] = prediction[col].apply(np.floor)
        prediction[col] = prediction[col].apply(lambda x: 0 if x < 0 else x)
        prediction[col] = prediction[col].astype(int)
    prediction.to_csv('baseline.csv', index=False)

def Wash():
    #丢弃相同时间数据
    train = pd.read_csv('../data/train/train.csv')
    ans = []
    droplist = []
    s = 0
    for name,group in tqdm(train.groupby(by=['QUEUE_ID','DOTTING_TIME'])):
        if group.shape[0] >1:
            group = group.sort_values(by=['CPU_USAGE','MEM_USAGE','LAUNCHING_JOB_NUMS'])
            droplist.extend(group.iloc[:-1].index.tolist())
    train = train.drop(index = droplist)
    train.to_csv('../data/wash_data.csv',index=False)
if __name__ == '__main__':
    #平滑化
    numpad = ['CPU_USAGE','LAUNCHING_JOB_NUMS','RUNNING_JOB_NUMS','SUCCEED_JOB_NUMS',
              'CANCELLED_JOB_NUMS','FAILED_JOB_NUMS','DOTTING_TIME']
    train = pd.read_csv('../data/wash_data.csv')
    ans =[]
    for i in train['QUEUE_ID'].unique():
        tm = train[train['QUEUE_ID'] == i]
        tm = tm.sort_values('DOTTING_TIME')
        print(i)
        len = tm.shape[0]
        for i in tqdm(range(len-1)):
            s,t = tm.iloc[i]['DOTTING_TIME'],tm.iloc[i+1]['DOTTING_TIME']
            x = t-s
            if  x > 10*1000*60 :
                par = tm.iloc[i + 1][numpad] - tm.iloc[i][numpad]
                for j in range(5*60*1000,5*60*1000,x):
                    md = tm.iloc[i]
                    md[numpad] += (par)*j/x
                    ans.append(md)
    train = train.append(pd.DataFrame(ans))
    for i in numpad:
        train[i] = train[i].apply(np.floor)
        train[i] = train[i].apply(lambda x: 0 if x < 0 else x)
        train[i] = train[i].astype(np.int64)
    train.to_csv('../data/end_data.csv')



