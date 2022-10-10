# Importing Libraries
import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint, shgo, dual_annealing, \
    differential_evolution, basinhopping
from functools import partial
from helper import *

def recommendation_media_cvxpy(data, current_volume, growth_ambition_perc):
    
    media_convergence_flag = False
    
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
    
    ###########################################
    cipm = df_only["current_effectiveness_per_metric"].values
    
    optimized_variable = cp.Variable(df_only.shape[0], "recommendation_metric_value")
    
    lower_bound = df_only['lower_bounds'].values
    upper_bound = df_only["upper_bounds"].values
    
    func = (cipm @ optimized_variable)
    
    obj = cp.Maximize(func)
    
    const1 = [optimized_variable >= lower_bound]
    const2 = [optimized_variable <= upper_bound]
    
    const = const1 + const2
    
    #Constructing constraints ,When there is only one media_type i.e, Digital
    if (len(data['media_type'].unique()) == 1):
        if (data['media_type'].unique() == 'Digital'):
        # below cons_ for whenever there is only digital data                     
            const3 = [(optimized_variable @ df_only["matrix_row_2_digital"].values) == total_digital['input_spends'].values[0]]
            const += const3
    else:
        
        const3 = [(optimized_variable @ df_only["matrix_row_1_tv"].values) == total_tv['input_spends'].values[0]]
        const4 = [(optimized_variable @ df_only["matrix_row_2_digital"].values) == total_digital['input_spends'].values[0]]
        
        const += const3
        const += const4

    const7 = [(optimized_variable @ df_only['matrix_row_3_vg'].values) >= matrix_3_min_cons]

    const += const7

    prob = cp.Problem(obj, constraints=const)
   
    solver_list = [cp.ECOS, cp.SCIPY, cp.SCS, cp.OSQP, cp.GLPK, cp.CBC, cp.CVXOPT, cp.ECOS_BB, cp.GLOP, cp.GLPK_MI]
    for i in solver_list:
        try:
            prob.solve(solver=i)
            print("optimality status with {} :".format(i), prob.status)
            print("optimal value with {} :".format(i), prob.value) 
            if prob.status == "optimal":
                print("Optimal solution found with solver {}".format(i))
                break
        except:
            pass

    if prob.status not in ["optimal"]:
        try:
            for i in solver_list:
                prob.solve(solver = i)
                if prob.status == "optimal_inaccurate":
                    print("Solution found but not the optimal one with solver {}".format(i))
                    break
        except:
            pass
                
        if prob.status not in ["optimal_inaccurate"]:
            print("No Solution Found with any solver")
            df_only["recommendation_metric_value"] = 0
    
    if prob.status in ["optimal", "optimal_inaccurate"]:
        df_only["recommendation_metric_value"] = optimized_variable.value
        media_convergence_flag = True
    ############################################
    
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
    debug_columns = ['lower_bounds', 'upper_bounds', 'matrix_row_1_tv', 'matrix_row_2_digital', 'matrix_row_3_vg']
#     recommendation_frame = recommendation_frame.drop(columns=debug_columns)

    return recommendation_frame.fillna(-1), media_convergence_flag

''' Below func will excute if algo didn't converge at the user specified growth_amb_perc. 
    It will find the appropriate/max growth_amb_perc ,ROI and spends where algo converges it better. 
'''
def binary_search_cvxpy(df_constraints_media, current_volume, array, low, high):
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

def recommendation_execution_cvxpy(df_dist, remainder_volume_growth):
    
    execution_convergence_flag = False
    
    df_dist = convert_df(df_dist)

    #Remaining input spends will be chop into the below list of variables
    current_val = ['current_price_per_volume', 'current_distribution', 'current_trade']
    df_dist_analysis = data_manipulation_execution(df_dist, current_val, inp_col_name='current_val')

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

    ########################################################
#     CVXYPY OPTIMIZER
    cost_price_index = (df_dist_analysis[df_dist_analysis["lever"] == "price"].index)
    cost_distribution_index = (df_dist_analysis[df_dist_analysis["lever"] == "distribution"].index)
    cost_trade_index = (df_dist_analysis[df_dist_analysis["lever"] == "trade"].index)
    
    optimized_variable = cp.Variable(df_dist_analysis.shape[0], "Recommended_val")
    
    a = df_dist_analysis['current_val'].values
    b = df_dist_analysis['current_price_per_volume'].values
    c = df_dist_analysis['elasticity'].values
    d = df_dist_analysis['current_volume'].values
    
    def vg_of_price_cvxpy(optimized_variable):
        vg_price = []
        for i in cost_price_index:
            vg_price.append((optimized_variable[i]-a[i])*c[i]*d[i]*(1/a[i]))
        return sum(vg_price)
    
    def vg_of_distribution_cvxpy(optimized_variable):
        vg_distribution = []
        for i in cost_distribution_index:
            vg_distribution.append((optimized_variable[i]-a[i])*c[i]*d[i]*(1/a[i]))
        return sum(vg_distribution)
    
    def vg_of_trade_cvxpy(optimized_variable):
        vg_trade = []
        for i in cost_trade_index:
            vg_trade.append((optimized_variable[i]-a[i])*c[i]*d[i]*(1/a[i]))
        return sum(vg_trade)
                            
    def vg_cvxpy(optimized_variable):

        vg_of_price = vg_of_price_cvxpy(optimized_variable)
        
        vg_of_distribution = vg_of_distribution_cvxpy(optimized_variable)
            
        vg_of_trade = vg_of_trade_cvxpy(optimized_variable)                                          
        
        return vg_of_price + vg_of_distribution + vg_of_trade

    fun = vg_cvxpy(optimized_variable)
        
    obj = cp.Maximize(fun)
    
    lower_bound = df_dist_analysis['lower_bounds'].values
    upper_bound = df_dist_analysis["upper_bounds"].values 

    const1 = [optimized_variable >= lower_bound]
    const2 = [optimized_variable <= upper_bound]
    
    def vol_constr_cvxpy(optimized_variable):
        vg = []
        
        for i in df_dist_analysis.index: 
            vg.append((optimized_variable[i] - a[i])*c[i]*d[i]*(1/a[i]))
            
        return sum(vg)
    
    const3 = [vol_constr_cvxpy(optimized_variable)>=remainder_volume_growth]
    
    prob = cp.Problem(obj, constraints = const1+const2+const3)
    
    solver_list = [cp.ECOS, cp.SCIPY, cp.SCS, cp.OSQP, cp.GLPK, cp.CBC, cp.CVXOPT, cp.ECOS_BB, cp.GLOP, cp.GLPK_MI, cp.PDLP]
    for i in solver_list:
        prob.solve(solver=i)
        print("optimality status with {} :".format(i), prob.status)
        print("optimal value with {} :".format(i), prob.value) 
        if prob.status == "optimal":
            print("Optimal solution found with solver {}".format(i))
            break

    if prob.status not in ["optimal"]:
        for i in solver_list:
            prob.solve(solver = i)
            if prob.status == "optimal_inaccurate":
                print("Solution found but not the optimal one with solver {}".format(i))
                break 
        if prob.status not in ["optimal_inaccurate"]:
            print("No Solution Found with any solver")
            df_dist_analysis["Recommended_val"] = 0
    
    if prob.status in ["optimal", "optimal_inaccurate"]:
        df_dist_analysis["Recommended_val"] = optimized_variable.value
        execution_convergence_flag = True
    
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

    return final_exec.fillna(-1), execution_convergence_flag