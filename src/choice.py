import numpy as np
import pandas as pd
def deal():
    lstm = pd.read_csv('../data/cmp/baseline_lstm.csv')
    base = pd.read_csv('../data/cmp/baseline_sum.csv')
    std = pd.read_csv('../data/test.csv')
    for i in [lstm, base, std]:
        i['lim'] = i[[f'CPU_USAGE_{i}' for i in range(1, 6)]].max(axis=1) - \
                   i[[f'CPU_USAGE_{i}' for i in range(1, 6)]].min(axis=1)

    ans = pd.DataFrame(pd.np.empty((0,lstm.shape[1])))
    print(ans.shape)
    for i in range(std.shape[0]):
        if (std.iloc[i]['lim'] - base['lim'].iloc[i]) * (lstm['lim'].iloc[i] - base['lim'].iloc[i]) > 0 and \
                (std.iloc[i]['CPU_USAGE_5'] - base.iloc[i]['CPU_USAGE_1'])**2 > 1.2*(std.iloc[i]['CPU_USAGE_5'] - lstm.iloc[i]['CPU_USAGE_1'])**2:
            mid = (lstm.iloc[i]*411 + base.iloc[i]*589)/1000
            ans = ans.append([mid.tolist()])
        else:
            mid = (lstm.iloc[i] * 189 + base.iloc[i] * 811) / 1000
            ans = ans.append([mid.tolist()])

    print(ans.shape)
    ans.columns = base.columns
    ans = ans.drop('lim',axis=1)
    for i in ans.columns:
        ans[i]=ans[i].astype(int)
    ans.to_csv('../data/cmp/baseline_merge_4.csv', index=False)
if __name__ == '__main__':
    x = pd.read_csv('../data/cmp/baseline_merge_3.csv')
    y = pd.read_csv('../data/cmp/baseline_2822.csv')
    z = (x + y)/2
    for i in  [i for i in z.columns]:
        z[i] = z[i].apply(np.round)
        z[i] = z[i].astype(int)
    z.to_csv('../data/cmp/baseline_merge_4.csv',index=False)
    deal()
