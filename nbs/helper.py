import numpy as np
import pandas as pd

# Universal functions

def convert_df(data):
    df = data.copy()
#     df = df.replace(-1, np.nan)

    for i in df.columns:
        ll = []
        for j in df[i]:
            try:
                ll.append(float(j))
            except:
                ll.append(j)

        df[i] = ll

    return df


# Media Functions
def brand_level(_data, _total_data, val):
    data_ = _data.copy()
    cols = ['country', 'year', 'timeline', 'analysis_period', 'category', 'brand', 'media_type', 'metric_type']
    # grouping the _data pandas dataframe on the cols list specified above for the sum aggregation function
    # data_ = data_.drop(columns=['id', ]).groupby(cols, sort=False).sum().replace(0, np.nan)
    # _total_data[val] = data_.reset_index()[val]

    data_ = data_.groupby(cols, as_index=False).aggregate({val: 'sum'})
    if val in _total_data.columns:
        _total_data = _total_data.drop(columns=[val, ])
    _total_data = _total_data.merge(right=data_, on=cols, how='left')
    return _total_data


# execution functions
def out(lv, val):
    if lv == 'price': return (val, np.nan, np.nan)
    if lv == 'distribution': return (np.nan, val, np.nan)
    if lv == 'trade': return (np.nan, np.nan, val)


def data_manipulation_execution(data, inp_cols, inp_col_name):

#     data = data.replace(-1, np.nan)
    dfs, lever = [], ['price', 'distribution', 'trade']

    for lev, cv in zip(lever, inp_cols):
        op = data[['id', 'current_volume', cv, lev + '_elasticity', 'channel', 'pack_name']].rename(
            columns={cv: inp_col_name, lev + '_elasticity': 'elasticity'}).dropna(how='any')
        op['current_price_per_volume'] = data['current_price_per_volume']
        op['lever'] = lev
        dfs.append(op)

    df = pd.concat(dfs)
    dda1 = df.query("pack_name not in ['Total'] and lever in ['price', 'distribution']")
    dda2 = df.query("pack_name in ['Total'] and lever in ['trade']")
    df = pd.concat([dda1, dda2])
    return df.reset_index(drop=True)


def total_execution(data, cols, try_):
    data = data.copy()
    if try_ == 0: l = 'recommendation'
    if try_ == 1: l = 'scenario'

#     data = data.replace(-1, np.nan)

    ind = data[data['pack_name'] == 'Total'].index
    oth_data = data[data['pack_name'] != 'Total']

    gro = oth_data.groupby('channel', sort=False).sum()

    cols = [i for i in cols if ((l in i) and
                                ('_volume' in i) and
                                ('trade' not in i))]

    val = gro[cols].values
    data.loc[ind, cols] = val
    data = data.fillna(-1)
    return data


def volume_growth_execution(data, return_sum=False, try_=0):
    if try_ == 0:
        l = 'Recommended_val'
    if try_ == 1:
        l = 'sce_val'
    pct_change_lever = (data[l] / data['current_val']) - 1
    pct_change_volume = pct_change_lever * data['elasticity']
    vg = pct_change_volume * data['current_volume']
    if return_sum: return sum(vg)
    return vg