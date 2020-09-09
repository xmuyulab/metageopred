import os
import pandas as pd

data = [pd.read_table('../data/kraken_report_20200811/{}'.format(i),header=None).set_index(5)[[0]].rename(columns={0:i[:-7]}).T.fillna(0) for i in os.listdir('./data/kraken_report_20200811/') if 'report' in i]

tmp = data[0].T
error_sample = []
for i in data[1:]:
    try:
        tmp = pd.merge(i.T,tmp,how='outer',left_index=True,right_index=True)
    except:
        error_sample.append(i)

tmp = tmp.fillna(0).T
describe = tmp.describe().T
tmp = tmp[describe[describe['std']>0].index]

tmp.to_csv('../data/kraken_data/kraken_data.csv')

 