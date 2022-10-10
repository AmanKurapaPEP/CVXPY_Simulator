# Importing Libraries
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint, shgo, dual_annealing, \
    differential_evolution, basinhopping
from functools import partial
from helper import *

def recommendation_media(data, current_volume, growth_ambition_perc):
    data = convert_df(data)

    data['current_effectiveness_per_metric'] = data['current_effectiveness_per_unit'] / data['one_unit_metric_quantity']
    data['input_cost_per_metric'] = data['input_cost_per_unit'] / data['one_unit_metric_quantity']

    # separating the dataframe , will be used for aggregating afterwards
    df_only = data[data['genre_platform'] != 'Total'].reset_index(drop=True)
    df_total = data[data['genre_platform'] == 'Total'].reset_index(drop=True)

    #seperating TV and digital data into two seperate dataframes(total_tv,total_digital)
    total_tv = df_total.query("media_type = = 'TV'")
    total_digital = df_total.query("media_type == 'Digital'")

    growth_ambition_volume = current_volume * growth_ambition_perc
    current_volume_contribution = df_total['current_volume'].sum()
    matrix_3_min_cons = growth_ambition_volume + current_volume_contribution

    print("matrix_3_min_cons", matrix_3_min_cons)
    # This is one of the main feature for the algorithm
    def bounds_col(value, media_type, min_=0, max_=0, total_tv_spends=0, total_digital_spends=0):
        '''
        This function is used for making lower and upper bounds for the minimize function.
        it takes in 7 arguments - 
        '''

        if min_ != 0:
            # if media_type == 'TV': return 0.05 * value  # 20% of current_metric_value
            if media_type == 'TV': return (0.05 * total_tv_spends) / value  # 5% of input_total_tv_spends
            if media_type == 'Digital': return (0.05 * total_digital_spends) / value  # 5% of input_total_digital_spends

        if max_ != 0:
            if media_type == 'TV': return (0.3 * total_tv_spends) / value  # 30% of input_total_tv_spends
            if media_type == 'Digital': return (0.6 * total_digital_spends) / value # 60% of input_total_digital_spends

    def volume_growth(value_to_optimize, data=df_only):
        data['recommendation_metric_value'] = value_to_optimize
        cal_values = list(map(lambda rv, mi, ma: True if mi <= rv <= ma else False, data['recommendation_metric_value'],
                              data['lower_bounds'], data['upper_bounds']))
        if all(cal_values):
            total_val = -(data['current_effectiveness_per_metric'] * data['recommendation_metric_value']).sum()
        else:
            total_val = 1e9
        return total_val
    
    # getting lower and upper bounds
    if total_tv.shape[0]>=1:
        tv_spends = total_tv['input_spends'].values[0]
    else:
        tv_spends = 0
    
    if total_digital.shape[0]>=1:
        digital_spends = total_digital['input_spends'].values[0]
    else:
        digital_spends = 0
        
    df_only['lower_bounds'] = tuple(
        map(lambda x, y: bounds_col(x, y, min_=1, total_tv_spends=tv_spends,
                                    total_digital_spends=digital_spends),
            df_only['input_cost_per_metric'], df_only['media_type']))

    df_only['upper_bounds'] = tuple(
        map(lambda x, y: bounds_col(x, y, max_=1, total_tv_spends=tv_spends,
                                    total_digital_spends=digital_spends),
            df_only['input_cost_per_metric'], df_only['media_type']))

    # making constraint matrix
    # df_only['matrix_row_1_tv'] = df_only['media_type'].apply(lambda x: 1 if x == 'TV' else 0)
    df_only['matrix_row_1_tv'] = df_only['media_type'].apply(lambda x: np.nan if x == 'TV' else 0).fillna(
        df_only['input_cost_per_metric'])
    df_only['matrix_row_2_digital'] = df_only['media_type'].apply(lambda x: np.nan if x == 'Digital' else 0).fillna(
        df_only['input_cost_per_metric'])

    df_only['matrix_row_3_vg'] = df_only['current_effectiveness_per_metric'].copy()

    # start_values are the game.
    # This is one of the key feature for the algorithm
    df_only['start_values'] = 0
    
    # bounds and constraint
    bounds = Bounds(lb=df_only['lower_bounds'], ub=df_only['upper_bounds'], keep_feasible=True)
    
    #Constructing constraints ,When there is only one media_type i.e, Digital
    if (len(data['media_type'].unique()) == 1):
        if (data['media_type'].unique() == 'Digital'):
            # below cons_ for whenever there is only digital data
            cons_ = [LinearConstraint(A=df_only[['matrix_row_2_digital']].T.values.tolist(),
                                      lb=(total_digital['input_spends'].values[0]),
                                      ub=(total_digital['input_spends'].values[0]),
                                      keep_feasible=True)]
    else:
        # below cons_ for generic data
        cons_ = [LinearConstraint(A=df_only[['matrix_row_1_tv', 'matrix_row_2_digital']].T.values.tolist(),
                                  lb=([total_tv['input_spends'].values[0], total_digital['input_spends'].values[0]]),
                                  ub=([total_tv['input_spends'].values[0], total_digital['input_spends'].values[0]]),
                                  keep_feasible=True)]

    ineqcons_ = LinearConstraint(
        A=df_only[['matrix_row_3_vg']].T.values.tolist(),
        lb=([matrix_3_min_cons]),
        ub=([np.inf]),
        keep_feasible=False)

    cons_.append(ineqcons_)

    '''minimize is the scipy optimize func where it takes the start ,bounds and constraint values
      optimize_output is an o/p from the scipy algo'''
    optimize_output = minimize(volume_growth, x0=df_only['start_values'], args=df_only,
                               bounds=bounds, constraints=cons_,
                               hess=lambda x, data: np.zeros((x.shape[0], x.shape[0])))

    # recommended value
    df_total = brand_level(df_only, df_total, 'recommendation_metric_value')

    # recommended spends
    df_only['recommendation_spends'] = df_only['recommendation_metric_value'] * df_only['input_cost_per_metric']
    df_total = brand_level(df_only, df_total, 'recommendation_spends')

    # recommended volume
    df_only['recommendation_volume'] = df_only['recommendation_metric_value'] * df_only[
        'current_effectiveness_per_metric']
    df_total = brand_level(df_only, df_total, 'recommendation_volume')

    # revenue 
    df_only['recommendation_revenue'] = (df_only['recommendation_volume'] * df_only['current_price_per_volume'])
    df_total = brand_level(df_only, df_total, 'recommendation_revenue')

    # roi
    df_only['recommendation_roi'] = df_only['recommendation_revenue'] / df_only['recommendation_spends']

    # brand level roi
    
    df_total['recommendation_roi'] = df_total['recommendation_revenue'] / df_total['recommendation_spends']

    # merging the frames and returning
    recommendation_frame = pd.concat([df_only, df_total], ignore_index=True)
    recommendation_frame[['scenario_spends', 'scenario_metric_value']] = np.nan
    debug_columns = ['lower_bounds', 'upper_bounds', 'matrix_row_1_tv', 'matrix_row_2_digital', 'matrix_row_3_vg', 'start_values']
#     recommendation_frame = recommendation_frame.drop(columns=debug_columns)
    return recommendation_frame.fillna(-1), optimize_output.success

''' Below func will excute if algo didn't converge at the user specified growth_amb_perc. 
    It will find the appropriate/max growth_amb_perc ,ROI and spends where algo converges it better. 
'''
def binary_search(df_constraints_media, current_volume, array, low, high):
    if high >= low:
        mid = low + (high - low) // 2
        recommendation_media_df, media_convergence_flag = recommendation_media(df_constraints_media, current_volume,
                                                                               array[mid])
        if media_convergence_flag:
            return array[mid]
        else:
            return binary_search(df_constraints_media, current_volume, array, low, mid + 1)
    return -1

'''Below snippet will recommended us where to invest out our remaining input spends on  different paramerts'''

def recommendation_execution(df_dist, remainder_volume_growth):
    df_dist = convert_df(df_dist)

    #Remaining input spends will be chop into the below list of variables
    current_val = ['current_price_per_volume', 'current_distribution', 'current_trade']
    df_dist_analysis = data_manipulation_execution(df_dist, current_val, inp_col_name='current_val')

    #calculations for three variables
    def volume_constraint(data, x):
        data['Recommended_val'] = x
        vg = volume_growth_execution(data, return_sum=True)
        return vg

    def cost_of_distribution(data):
        cost = 0.01 * (data['Recommended_val'] - data['current_val'])
        return sum(cost)

    def cost_of_price(data):
        vg = volume_growth_execution(data)
        cost = vg * (data['Recommended_val'] - data['current_val'])
        return sum(cost)

    def cost_of_trade(data):
        vg = volume_growth_execution(data)
        cost = vg * (data['Recommended_val'] - data['current_val']) * data['current_price_per_volume']
        return sum(cost)

    def cost(values_to_optimize, data):
        data['Recommended_val'] = values_to_optimize
        cal_values = list(
            map(lambda rv, mi, ma: True if mi <= rv <= ma else False, data['Recommended_val'], data['lower_bounds'],
                data['upper_bounds']))

        if all(cal_values):

            p = cost_of_price(data[data['lever'] == 'price'])
            d = cost_of_distribution(data[data['lever'] == 'distribution'])
            t = cost_of_trade(data[data['lever'] == 'trade'])

            total_val = p + d + t
            return total_val

        else:
            total_val = 1e9

        return total_val

    # Bounds
    margin = 0.1
    df_dist_analysis['lower_bounds'] = df_dist_analysis['current_val'] * (1 - margin)
    df_dist_analysis['upper_bounds'] = df_dist_analysis['current_val'] * (1 + margin)
    df_dist_analysis['start_values'] = 0

    df_dist_analysis.loc[df_dist_analysis['lever'] == 'price', 'upper_bounds'] = df_dist_analysis['current_val']
    df_dist_analysis.loc[df_dist_analysis['lever'] == 'distribution', 'lower_bounds'] = df_dist_analysis['current_val']

    # Changing trade bounds as per user input
    cc = df_dist[['id', 'channel', 'pack_name', 'current_trade', 'input_trade']].dropna().reset_index(drop=True)
    cc['lower_bounds'] = cc[['current_trade', 'input_trade']].dropna().min(axis=1)
    cc['upper_bounds'] = cc[['current_trade', 'input_trade']].dropna().max(axis=1)

    for i in cc.itertuples():
        df_dist_analysis.loc[(df_dist_analysis['id'] == i[1]) & (df_dist_analysis['channel'] == i[2]) & (df_dist_analysis['lever'] == 'trade'), 'lower_bounds'] = \
            i[-2]
        df_dist_analysis.loc[(df_dist_analysis['id'] == i[1]) & (df_dist_analysis['channel'] == i[2]) & (df_dist_analysis['lever'] == 'trade'), 'upper_bounds'] = \
            i[-1]

    # df_dist_analysis = df_dist_analysis.query("lower_bounds < upper_bounds")
    bounds = Bounds(lb=df_dist_analysis['lower_bounds'], ub=df_dist_analysis['upper_bounds'], keep_feasible=False)

    # constraints
    nl_constraints = NonlinearConstraint(partial(volume_constraint, df_dist_analysis.copy(deep=True)),
                                         lb=remainder_volume_growth,
                                         ub=np.inf,
                                         keep_feasible=False)
    '''minimize is the scipy optimize func where it takes the start ,bounds and constraint values
      result is an o/p from the scipy algo'''

    result = minimize(cost, x0=df_dist_analysis['start_values'], args=df_dist_analysis,
                      bounds=bounds, constraints=[nl_constraints],
                      hess=lambda x, data: np.zeros((x.shape[0], x.shape[0])),
                      method='trust-constr',
                      options={'maxiter': 1000, 'verbose': 1, 'factorization_method': 'SVDFactorization'})

    # getting recommendation cols
    new_names_r = ['recommendation_' + i.replace('current_', '') for i in current_val]
    df_dist_analysis[new_names_r] = tuple(map(lambda lv, val: out(lv, val),
                                              df_dist_analysis['lever'], df_dist_analysis['Recommended_val']))
    # volume growth
    df_dist_analysis['volume_growth'] = volume_growth_execution(df_dist_analysis)
    new_names_v = [i + '_incremental_volume' for i in new_names_r]
    df_dist_analysis[new_names_v] = tuple(map(lambda lv, val: out(lv, val),
                                              df_dist_analysis['lever'], df_dist_analysis['volume_growth']))
    df_dist_analysis.drop(columns=['Recommended_val', ], inplace=True)

    final_exec = df_dist.copy()
    for r, v in zip(new_names_r, new_names_v): final_exec = pd.merge(final_exec, df_dist_analysis[['id', r, v]].dropna(
        subset=[r, ]), on=['id', ], how='left')

    final_exec = total_execution(final_exec, new_names_v, try_=0)
    new_names_s = ['scenario_' + i.replace('current_', '') for i in current_val]
    final_exec[new_names_s] = np.nan
    return final_exec.fillna(-1), result.success