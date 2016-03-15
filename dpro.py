import pandas as pd
import numpy as np
import datetime
from numpy import arange, array, ones
from scipy import stats


# function: allocate a time period (dt) to one of periods in a period range (ps)
def __get_period(dt, ps):
    for i in range(0, len(ps) - 1):
        if dt >= ps[i] and dt < ps[i] + 1:
            return ps[i]
    return None


# function: process raw dataframe and return feature values
def process(sk):
    #######################################################
    # add a column DT as datetime
    sk['DT'] = pd.to_datetime(sk['Time'].apply(int), unit='ms')
    # read the start time and end time
    # Timestamp('2016-02-29 01:49:18.523000')
    s = sk['DT'][0]
    # Timestamp('2016-02-29 01:59:29.025000')
    e = sk['DT'][len(sk) - 1]
    #######################################################

    #######################################################
    # create two period ranges(a period range is a list of successive windows),
    # both with window-width of 4 seconds
    # range 1 and range2 will have 50% overlapping

    # period range 1: first build the range based on start and end time,
    # then append additional one to the last
    prg1 = pd.period_range(s, e, freq='4s')
    prg1 = pd.period_range(s, periods=len(prg1) + 1, freq='4s')

    # period range 2: first build the range based on (start time + 2s) and end time,
    # then append additional one to the last
    prg2 = pd.period_range(s + datetime.timedelta(seconds=2), e, freq='4s')
    prg2 = pd.period_range(s + datetime.timedelta(seconds=2), periods=len(prg2) + 1, freq='4s')
    #######################################################

    #######################################################
    # for each row, compare its timestamp to range1 and range2 respectively,
    # record the right window in two column, Period1 and Period2
    p1 = []
    p2 = []
    for dt in pd.DatetimeIndex(sk['DT']).to_period('4s'):
        p1.append(__get_period(dt, prg1))
        p2.append(__get_period(dt, prg2))
    sk['Period1'] = p1
    sk['Period2'] = p2
    #######################################################

    #######################################################
    # get rid of unnecessary colums
    sk = sk.drop('ID', 1)
    sk = sk.drop('Time', 1)
    sk = sk.drop('DT', 1)
    #######################################################

    #######################################################
    # get two grouped dataset,
    # one is derived by grouping original dataset by Period 1, the other by Period 2
    grouped1 = sk.groupby('Period1', as_index=False)
    grouped2 = sk.groupby('Period2', as_index=False)
    #######################################################

    #######################################################
    # create a set of aggregation functions
    # which is used to calculate feature based on each window
    f = {
        'Label': {'postive_rate': lambda x: sum(x) * 1.0 / len(x)},

        'CO2Value': {  # 'sum': np.sum,
            'mean': np.mean,
            'mean_trim': lambda x: 40 if np.mean(x) > 40 else np.mean(x),
            'std': np.std,
            'max': np.max,
            'min': np.min,
            # 'integ_den': lambda x: sum(abs(x))/len(x),
            'slope': lambda x: stats.linregress(arange(0, len(x)), x)[0],
            'mean_diff': lambda x: (
            np.diff([sum(x[i:i + (len(x) + 1) / 2]) for i in xrange(0, len(x), (len(x) + 1) / 2)])[0]) if len(
                x) > 1 else 0,
            'std_diff': lambda x: (
            np.diff([np.std(x[i:i + (len(x) + 1) / 2]) for i in xrange(0, len(x), (len(x) + 1) / 2)])[0]) if len(
                x) > 1 else 0,
            'slope_diff': lambda x: (np.diff(
                [stats.linregress(arange(0, len(x[i:i + (len(x) + 1) / 2])), x[i:i + (len(x) + 1) / 2])[0] for i in
                 xrange(0, len(x), (len(x) + 1) / 2)])[0]) if len(x) > 1 else 0,
            'intr_ac_x_max_max': lambda x: np.max(x) * np.max(sk.ix[x.index].Accel_X),
            'intr_ac_x_max_slope': lambda x: np.max(x) * stats.linregress(arange(0, len(x)), sk.ix[x.index].Accel_X)[
                0]},

        'Accel_X': {  # 'sum': np.sum,
            'mean': np.mean,
            # 'std': np.std,
            'max': np.max,
            'mn': np.min,
            # 'integ_den': lambda x: sum(abs(x))/len(x),
            'slope': lambda x: stats.linregress(arange(0, len(x)), x)[0],
            'corr_ac_x_y': lambda x: np.corrcoef(x, sk.ix[x.index].Accel_Y)[0, 1],
            'corr_ac_x_z': lambda x: np.corrcoef(x, sk.ix[x.index].Accel_Z)[0, 1],
            # 'sum_ac_xyz2': lambda x: np.sum(
            #   x*x + sk.ix[x.index].Accel_Y*sk.ix[x.index].Accel_Y +
            #   sk.ix[x.index].Accel_Y*sk.ix[x.index].Accel_Y),
            'mean_ac_xyz2': lambda x: np.mean(
                x * x + sk.ix[x.index].Accel_Y * sk.ix[x.index].Accel_Y +
                sk.ix[x.index].Accel_Y * sk.ix[x.index].Accel_Y),
            'std_ac_xyz2': lambda x: np.std(
                x * x + sk.ix[x.index].Accel_Y * sk.ix[x.index].Accel_Y +
                sk.ix[x.index].Accel_Y * sk.ix[x.index].Accel_Y),
            'slope_ac_xyz2': lambda x: stats.linregress(arange(0, len(x)), (
                x * x + sk.ix[x.index].Accel_Y * sk.ix[x.index].Accel_Y +
                sk.ix[x.index].Accel_Y * sk.ix[x.index].Accel_Y))[0],
            'mean_diff': lambda x: (
            np.diff([sum(x[i:i + (len(x) + 1) / 2]) for i in xrange(0, len(x), (len(x) + 1) / 2)])[0]) if len(
                x) > 1 else 0,
            'std_diff': lambda x: (
            np.diff([np.std(x[i:i + (len(x) + 1) / 2]) for i in xrange(0, len(x), (len(x) + 1) / 2)])[0]) if len(
                x) > 1 else 0,
            'slope_diff': lambda x: (np.diff(
                [stats.linregress(arange(0, len(x[i:i + (len(x) + 1) / 2])), x[i:i + (len(x) + 1) / 2])[0] for i in
                 xrange(0, len(x), (len(x) + 1) / 2)])[0]) if len(x) > 1 else 0},
        'Accel_Y': {  # 'sum': np.sum,
            'mean': np.mean,
            # 'std': np.std,
            'max': np.max,
            'min': np.min,
            # 'integ_den': lambda x: sum(abs(x))/len(x),
            'slope': lambda x: stats.linregress(arange(0, len(x)), x)[0],
            'corr_ac_y_z': lambda x: np.corrcoef(x, sk.ix[x.index].Accel_Z)[0, 1],
            'mean_diff': lambda x: (
            np.diff([sum(x[i:i + (len(x) + 1) / 2]) for i in xrange(0, len(x), (len(x) + 1) / 2)])[0]) if len(
                x) > 1 else 0,
            'std_diff': lambda x: (
            np.diff([np.std(x[i:i + (len(x) + 1) / 2]) for i in xrange(0, len(x), (len(x) + 1) / 2)])[0]) if len(
                x) > 1 else 0,
            'slope_diff': lambda x: (np.diff(
                [stats.linregress(arange(0, len(x[i:i + (len(x) + 1) / 2])), x[i:i + (len(x) + 1) / 2])[0] for i in
                 xrange(0, len(x), (len(x) + 1) / 2)])[0]) if len(x) > 1 else 0},
        'Accel_Z': {  # 'sum': np.sum,
            'mean': np.mean,
            # 'std': np.std,
            'max': np.max,
            'min': np.min,
            # 'integ_den': lambda x: sum(abs(x))/len(x),
            'slope': lambda x: stats.linregress(arange(0, len(x)), x)[0],
            'mean_diff': lambda x: (
            np.diff([sum(x[i:i + (len(x) + 1) / 2]) for i in xrange(0, len(x), (len(x) + 1) / 2)])[0]) if len(
                x) > 1 else 0,
            'std_diff': lambda x: (
            np.diff([np.std(x[i:i + (len(x) + 1) / 2]) for i in xrange(0, len(x), (len(x) + 1) / 2)])[0]) if len(
                x) > 1 else 0,
            'slope_diff': lambda x: (np.diff(
                [stats.linregress(arange(0, len(x[i:i + (len(x) + 1) / 2])), x[i:i + (len(x) + 1) / 2])[0] for i in
                 xrange(0, len(x), (len(x) + 1) / 2)])[0]) if len(x) > 1 else 0},

        'Gyro_X': {  # 'sum': np.sum,
            'mean': np.mean,
            # 'std': np.std,
            # 'max': np.max,
            # 'min': np.min,
            # 'integ_den': lambda x: sum(abs(x))/len(x),
            'slope': lambda x: stats.linregress(arange(0, len(x)), x)[0],
            'corr_gy_x_y': lambda x: np.corrcoef(x, sk.ix[x.index].Gyro_Y)[0, 1],
            'corr_gy_x_z': lambda x: np.corrcoef(x, sk.ix[x.index].Gyro_Z)[0, 1]},
        'Gyro_Y': {  # 'sum': np.sum,
            'mean': np.mean,
            # 'std': np.std,
            # 'max': np.max,
            # 'min': np.min,
            # 'integ_den': lambda x: sum(abs(x))/len(x),
            'slope': lambda x: stats.linregress(arange(0, len(x)), x)[0],
            'corr_gy_y_z': lambda x: np.corrcoef(x, sk.ix[x.index].Gyro_Z)[0, 1]},
        'Gyro_Z': {  # 'sum': np.sum,
            'mean': np.mean,
            # 'std': np.std,
            # 'max': np.max,
            # 'min': np.min,
            # 'integ_den': lambda x: sum(abs(x))/len(x),
            'slope': lambda x: stats.linregress(arange(0, len(x)), x)[0]},

        'Magn_0': {  # 'sum': np.sum,
            'mean': np.mean,
            # 'std': np.std,
            'max': np.max,
            # 'min': np.min,
            # 'integ_den': lambda x: sum(abs(x))/len(x),
            'slope': lambda x: stats.linregress(arange(0, len(x)), x)[0],
            'corr_mg_0_1': lambda x: np.corrcoef(x, sk.ix[x.index].Magn_1)[0, 1],
            'corr_mg_0_2': lambda x: np.corrcoef(x, sk.ix[x.index].Magn_2)[0, 1]},
        'Magn_1': {  # 'sum': np.sum,
            'mean': np.mean,
            'std': np.std,
            'max': np.max,
            # 'min': np.min,
            # 'integ_den': lambda x: sum(abs(x))/len(x),
            'slope': lambda x: stats.linregress(arange(0, len(x)), x)[0],
            'corr_mg_1_2': lambda x: np.corrcoef(x, sk.ix[x.index].Magn_2)[0, 1]},
        'Magn_2': {  # 'sum': np.sum,
            'mean': np.mean,
            'std': np.std,
            'max': np.max,
            # 'min': np.min,
            # 'integ_den': lambda x: sum(abs(x))/len(x),
            'slope': lambda x: stats.linregress(arange(0, len(x)), x)[0]},
    }

    # apply aggregation functions to each group
    g1 = grouped1.agg(f)
    g2 = grouped2.agg(f)
    #######################################################



    #######################################################
    # change the column names Period1 and Period2 into Period
    g1.columns = g1.columns.set_levels(
        [u'Label', u'Magn_2', u'Magn_0', u'Magn_1', u'Dust', u'Accel_Z', u'Accel_X',
         u'Accel_Y', u'Gyro_Z', u'Gyro_X', u'Gyro_Y', u'Period'], level=0)
    g2.columns = g2.columns.set_levels(
        [u'Label', u'Magn_2', u'Magn_0', u'Magn_1', u'Dust', u'Accel_Z', u'Accel_X',
         u'Accel_Y', u'Gyro_Z', u'Gyro_X', u'Gyro_Y', u'Period'], level=0)
    #######################################################



    #######################################################
    # concatenate two aggreated datasets, and resort by Period
    result = pd.concat([g1, g2]).sort_values(by='Period', ascending=1)
    result = result.set_index(['Period'])
    #######################################################


    #######################################################
    # get leading and lagging data set
    result_lag2 = result.shift(1)
    result_lag2.columns = result_lag2.columns.set_levels(
        [u'Label.lag2', u'Magn_2.lag2', u'Magn_0.lag2', u'Magn_1.lag2',
         u'Dust.lag2', u'Accel_Z.lag2', u'Accel_X.lag2', u'Accel_Y.lag2',
         u'Gyro_Z.lag2', u'Gyro_X.lag2', u'Gyro_Y.lag2', u'Period.lag2'], level=0)

    result_lead2 = result.shift(-1)
    result_lead2.columns = result_lead2.columns.set_levels(
        [u'Label.lead2', u'Magn_2.lead2', u'Magn_0.lead2', u'Magn_1.lead2',
         u'Dust.lead2', u'Accel_Z.lead2', u'Accel_X.lead2', u'Accel_Y.lead2',
         u'Gyro_Z.lead2', u'Gyro_X.lead2', u'Gyro_Y.lead2', u'Period.lead2'], level=0)
    #######################################################


    #######################################################
    # flatten column names
    result.columns = ['%s%s' % (a, '|%s' % b if b else '') for a, b in result.columns]
    result_lag2.columns = ['%s%s' % (a, '|%s' % b if b else '') for a, b in result_lag2.columns]
    result_lead2.columns = ['%s%s' % (a, '|%s' % b if b else '') for a, b in result_lead2.columns]

    # get rid of unnecessary colums
    result_lag2 = result_lag2.drop('Label.lag2|postive_rate', 1)
    result_lead2 = result_lead2.drop('Label.lead2|postive_rate', 1)

    # horizontally concatenate the above result with its lagging and leading datasets
    result_with_shift = pd.concat([result, result_lag2, result_lead2], axis=1)
    #######################################################



    #######################################################
    # remove all rows with null values
    result_with_shift = result_with_shift.dropna()

    # standardize all culumns
    # cols_to_stand = result_with_shift.columns[range(1,len(result_with_shift.columns))]
    # result_with_shift[cols_to_stand] = result_with_shift[cols_to_stand].apply(
    #    lambda x: (x - np.mean(x)) / np.std(x) )
    #######################################################



    #######################################################
    # ONLY for training
    # cut the postive_rate at a threshood at 0.8, above that set to 1, otherwise 0
    # result_with_shift['Label|postive_rate'] = (
    #                                              result_with_shift['Label|postive_rate'] > 0.80)*1
    #######################################################

    return result_with_shift
