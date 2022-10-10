# Importing Libraries
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
from helper import *


def scenario_media(recommeded_frame_with_scenrio_val):
    data = convert_df(recommeded_frame_with_scenrio_val)

    data['scenario_cost_per_metric'] = data['input_cost_per_metric'].copy()

    # separating the dataframe , will be used for aggregating afterwards
    df_only = data[data['genre_platform'] != 'Total'].reset_index(drop=True)
    df_total = data[data['genre_platform'] == 'Total'].reset_index(drop=True)

    # try vc output
    df_only['scenario_volume_output'] = tuple(map(lambda med, eff, cpm, sp, grp: grp * eff if med == 'TV'
    else (sp / cpm) * eff if med == 'Digital' else 0,
                                                  df_only['media_type'], df_only['current_effectiveness_per_metric'],
                                                  df_only['input_cost_per_metric'],
                                                  df_only['scenario_spends_digital_input'],
                                                  df_only['scenario_metric_value_input']))

    df_total = brand_level(df_only, df_total, 'scenario_volume_output')

    # spends
    df_only['scenario_spends_tv_output'] = df_only['scenario_metric_value_input'] * df_only[
        'input_cost_per_metric']  # tv
    df_only['scenario_metric_value_digital_output'] = df_only['scenario_spends_digital_input'] / df_only[
        'input_cost_per_metric']  # digital

    df_only['temp_spends_output'] = df_only[['scenario_spends_tv_output', 'scenario_spends_digital_input']].fillna(0).sum(axis=1)

    df_total = brand_level(df_only, df_total, 'temp_spends_output')

    # revenue
    df_only['scenario_revenue_output'] = (df_only['scenario_volume_output'] * df_only['current_price_per_volume'])
    df_total = brand_level(df_only, df_total, 'scenario_revenue_output')

    # roi scenario
    df_only['scenario_roi_output'] = df_only['scenario_revenue_output'] / df_only['temp_spends_output']

    # brand level roi
    df_total['scenario_roi_output'] = df_total['scenario_revenue_output'] / df_total['temp_spends_output']

    df_total = brand_level(df_only, df_total, 'scenario_spends_tv_output')
    df_total = brand_level(df_only, df_total, 'scenario_metric_value_digital_output')

    # summing spends and metric values
    df_only['scenario_spends_digital_output'] = df_only['scenario_spends_digital_input'].copy()
    df_only['scenario_metric_tv_output'] = df_only['scenario_metric_value_input'].copy()

    df_total = brand_level(df_only, df_total, 'scenario_spends_digital_output')
    df_total = brand_level(df_only, df_total, 'scenario_metric_tv_output')

    # returning frame
    scenario_frame = pd.concat([df_only, df_total])

    # replacing 0 with NAN
    scenario_frame['scenario_metric_tv_output'].replace(['0', 0], np.nan,inplace=True)

    scenario_frame['scenario_spends_output'] = scenario_frame['scenario_spends_tv_output'].fillna(scenario_frame['scenario_spends_digital_output'])
    scenario_frame['scenario_metric_value_output'] = scenario_frame['scenario_metric_tv_output'].fillna(scenario_frame['scenario_metric_value_digital_output'])

    #Dropping and renaming the necessary columns
    scenario_frame['scenario_spends']=scenario_frame['temp_spends_output']
    scenario_frame['scenario_metric_value']=scenario_frame['scenario_metric_value_output']
    scenario_frame.drop(['scenario_spends_output','scenario_spends_digital_input','scenario_metric_value_output',
                         'scenario_metric_value_input','scenario_spends_tv_output', 'scenario_spends_digital_output',
                          'scenario_metric_value_digital_output','scenario_metric_tv_output','temp_spends_output'], axis=1, inplace=True)
    scenario_frame.rename(columns={'scenario_roi_output': 'scenario_roi', 'scenario_revenue_output': 'scenario_revenue',
                                   'scenario_volume_output': 'scenario_volume'}, inplace=True)

    return scenario_frame.fillna(-1)

def scenario_execution(data):
    scna = 'scenario_price_per_volume'
    rna = 'current_price_per_volume'

    data_1 = data_manipulation_execution(data, [scna, 'scenario_distribution', 'scenario_trade'], 'sce_val')
    data_2 = data_manipulation_execution(data, [rna, 'current_distribution', 'current_trade'], 'current_val')

    data_2['sce_val'] = data_1['sce_val']
    data_2['pct_change_lever'] = (data_2['sce_val'] / data_2['current_val']) - 1
    data_2['pct_change_volume'] = data_2['pct_change_lever'] * data_2['elasticity']
    data_2['vg'] = data_2['pct_change_volume'] * data_2['current_volume']

    # getting recommendation cols
    new_names_s = [scna, 'scenario_distribution', 'scenario_trade']
    data_2[new_names_s] = tuple(map(lambda lv, val: out(lv, val), data_2['lever'], data_2['sce_val']))

    # volume growth
    data_2['volume_growth'] = volume_growth_execution(data_2, try_=1)

    new_names_v = [i + '_incremental_volume' for i in new_names_s]
    data_2[new_names_v] = tuple(map(lambda lv, val: out(lv, val), data_2['lever'], data_2['volume_growth']))
    data_2.drop(columns=['sce_val', ], inplace=True)

    final_exec = data.copy()
    for r, v in zip(new_names_s, new_names_v):
        final_exec = pd.merge(final_exec, data_2[['id', r, v]].dropna(subset=[r, ]).drop(
                                  columns=[r, ]), on=['id', ], how='left')
    final_exec = total_execution(final_exec, new_names_v, try_=1)

    return final_exec.fillna(-1)