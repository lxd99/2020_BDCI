import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

SEQ_LEN = 5
WINDOW_SIZE = 5
dt = 'DOTTING_TIME'

data = pd.read_csv('../data/to_train.csv')

# to_drop = []
# mas = 1000*7*60
# mis = 1000*3*60
# for  i in tqdm(data.iterrows()):
#     mid = i[1]
#     for j in range(1,10):
#         if mid[f'DOTTING_TIME_{j+1}'] -  mid[f'DOTTING_TIME_{j}'] > mas \
#                 or mid[f'DOTTING_TIME_{j + 1}'] - mid[f'DOTTING_TIME_{j }'] < mis:
#             to_drop.append(i[0])
#             break
#
# data = data.drop(index=to_drop)
# data.to_csv('../data/to_train.csv')
for i in range(1,11):
    data[f'DOTTING_TIME_{i}'] = pd.to_datetime(
        data[f'DOTTING_TIME_{i}'], unit='ms')
data.to_csv('../data/test_train.csv', index=False)
