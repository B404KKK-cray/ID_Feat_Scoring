"""
Copyright, Rinat Maksutov, 2017.
License: GNU General Public License
"""

import numpy as np
import pandas as pd

"""
Exponential moving average
Source: http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_averages
Params: 
    data: pandas DataFrame
    period: smoothing period
    column: the name of the column with values for calculating EMA in the 'data' DataFrame
    
Returns:
    copy of 'data' DataFrame with 'ema[period]' column added
"""
def ema(data, period=0, column='<CLOSE>'):
    data['ema' + str(period)] = data[column].ewm(ignore_na=False, min_periods=period, com=period, adjust=True).mean()
    
    return data

"""
Moving Average Convergence/Divergence Oscillator (MACD)
Source: http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_average_convergence_divergence_macd
Params: 
    data: pandas DataFrame
    period_long: the longer period EMA (26 days recommended)
    period_short: the shorter period EMA (12 days recommended)
    period_signal: signal line EMA (9 days recommended)
    column: the name of the column with values for calculating MACD in the 'data' DataFrame
    
Returns:
    copy of 'data' DataFrame with 'macd_val' and 'macd_signal_line' columns added
"""
def macd(data, period_long=26, period_short=12, period_signal=9, column='<CLOSE>'):
    remove_cols = []
    if not 'ema' + str(period_long) in data.columns:
        data = ema(data, period_long)
        remove_cols.append('ema' + str(period_long))

    if not 'ema' + str(period_short) in data.columns:
        data = ema(data, period_short)
        remove_cols.append('ema' + str(period_short))

    data['macd_val'] = data['ema' + str(period_short)] - data['ema' + str(period_long)]
    data['macd_signal_line'] = data['macd_val'].ewm(ignore_na=False, min_periods=0, com=period_signal, adjust=True).mean()

    data = data.drop(remove_cols, axis=1)
        
    return data

"""
Accumulation Distribution 
Source: http://stockcharts.com/school/doku.php?st=accumulation+distribution&id=chart_school:technical_indicators:accumulation_distribution_line
Params: 
    data: pandas DataFrame
    trend_periods: the over which to calculate AD
    open_col: the name of the OPEN values column
	high_col: the name of the HIGH values column
	low_col: the name of the LOW values column
	close_col: the name of the CLOSE values column
	vol_col: the name of the VOL values column
    
Returns:
    copy of 'data' DataFrame with 'acc_dist' and 'acc_dist_ema[trend_periods]' columns added
"""
def acc_dist(data, trend_periods=21, open_col='<OPEN>', high_col='<HIGH>', low_col='<LOW>', close_col='<CLOSE>', vol_col='<VOL>'):
    for index, row in data.iterrows():
        if row[high_col] != row[low_col]:
            ac = ((row[close_col] - row[low_col]) - (row[high_col] - row[close_col])) / (row[high_col] - row[low_col]) * row[vol_col]
        else:
            ac = 0
        data.set_value(index, 'acc_dist', ac)
    data['acc_dist_ema' + str(trend_periods)] = data['acc_dist'].ewm(ignore_na=False, min_periods=0, com=trend_periods, adjust=True).mean()
    
    return data

def sht_ui(client_con):
	
	##DECLARE date1 STRING DEFAULT '2025-07-31';

	sql = '''
With base as (
  select 
      *
  from joey-bi-ss-risk-fraud-project.credit.SHTList_2025_08
)
select *
from base
'''
	sht_ui = client_con.query(sql).to_dataframe()
	return sht_ui

def inco_ui(client_con):
	
	##DECLARE date1 STRING DEFAULT '2025-07-31';

	sql = '''
With base as (
  select 
      *
  from joey-bi-ss-risk-fraud-project.credit.inco_SK_2025_08
)
select *
from base
'''
	inco_ui = client_con.query(sql).to_dataframe()
	return inco_ui

def disb_ui(client_con):
	
	##DECLARE date1 STRING DEFAULT '2025-07-31';

	sql = '''
With base as (
  select 
      *
  from joey-bi-ss-risk-fraud-project.credit.disb_SK_2025_08
)
select *
from base
'''
	disb_ui = client_con.query(sql).to_dataframe()
	return disb_ui

def porto_ui(client_con):
	
	##DECLARE date1 STRING DEFAULT '2025-07-31';

	sql = '''
With base as (
  select 
      *
  from joey-bi-ss-risk-fraud-project.credit.porto_SK_2025_08
)
select *
from base
'''
	porto_ui = client_con.query(sql).to_dataframe()
	return porto_ui


def vint_ui(client_con):
	
	##DECLARE date1 STRING DEFAULT '2025-07-31';

	sql = '''
With base as (
  select 
      *
  from joey-bi-ss-risk-fraud-project.credit.vint_SK_2025_08
)
select *
from base
'''
	vint_ui = client_con.query(sql).to_dataframe()
	return vint_ui

def ttd_ui(client_con):
	
	##DECLARE date1 STRING DEFAULT '2025-07-31';

	sql = '''
With base as (
  select 
      *
  from joey-bi-ss-risk-fraud-project.credit.TD_SK_2025_08
)
select *
from base
'''
	ttd_ui = client_con.query(sql).to_dataframe()
	return ttd_ui

"""
On Balance Volume (OBV)
Source: http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:on_balance_volume_obv
Params: 
    data: pandas DataFrame
    trend_periods: the over which to calculate OBV
	close_col: the name of the CLOSE values column
	vol_col: the name of the VOL values column
    
Returns:
    copy of 'data' DataFrame with 'obv' and 'obv_ema[trend_periods]' columns added
"""
def on_balance_volume(data, trend_periods=21, close_col='<CLOSE>', vol_col='<VOL>'):
    for index, row in data.iterrows():
        if index > 0:
            last_obv = data.at[index - 1, 'obv']
            if row[close_col] > data.at[index - 1, close_col]:
                current_obv = last_obv + row[vol_col]
            elif row[close_col] < data.at[index - 1, close_col]:
                current_obv = last_obv - row[vol_col]
            else:
                current_obv = last_obv
        else:
            last_obv = 0
            current_obv = row[vol_col]

        data.set_value(index, 'obv', current_obv)

    data['obv_ema' + str(trend_periods)] = data['obv'].ewm(ignore_na=False, min_periods=0, com=trend_periods, adjust=True).mean()
    
    return data

"""
Price-volume trend (PVT) (sometimes volume-price trend)
Source: https://en.wikipedia.org/wiki/Volume%E2%80%93price_trend
Params: 
    data: pandas DataFrame
    trend_periods: the over which to calculate PVT
	close_col: the name of the CLOSE values column
	vol_col: the name of the VOL values column
    
Returns:
    copy of 'data' DataFrame with 'pvt' and 'pvt_ema[trend_periods]' columns added
"""
def price_volume_trend(data, trend_periods=21, close_col='<CLOSE>', vol_col='<VOL>'):
    for index, row in data.iterrows():
        if index > 0:
            last_val = data.at[index - 1, 'pvt']
            last_close = data.at[index - 1, close_col]
            today_close = row[close_col]
            today_vol = row[vol_col]
            current_val = last_val + (today_vol * (today_close - last_close) / last_close)
        else:
            current_val = row[vol_col]

        data.set_value(index, 'pvt', current_val)

    data['pvt_ema' + str(trend_periods)] = data['pvt'].ewm(ignore_na=False, min_periods=0, com=trend_periods, adjust=True).mean()
        
    return data

"""
Average true range (ATR)
Source: https://en.wikipedia.org/wiki/Average_true_range
Params: 
    data: pandas DataFrame
    trend_periods: the over which to calculate ATR
    open_col: the name of the OPEN values column
	high_col: the name of the HIGH values column
	low_col: the name of the LOW values column
	close_col: the name of the CLOSE values column
	vol_col: the name of the VOL values column
	drop_tr: whether to drop the True Range values column from the resulting DataFrame
    
Returns:
    copy of 'data' DataFrame with 'atr' (and 'true_range' if 'drop_tr' == True) column(s) added
"""
def average_true_range(data, trend_periods=14, open_col='<OPEN>', high_col='<HIGH>', low_col='<LOW>', close_col='<CLOSE>', drop_tr = True):
    for index, row in data.iterrows():
        prices = [row[high_col], row[low_col], row[close_col], row[open_col]]
        if index > 0:
            val1 = np.amax(prices) - np.amin(prices)
            val2 = abs(np.amax(prices) - data.at[index - 1, close_col])
            val3 = abs(np.amin(prices) - data.at[index - 1, close_col])
            true_range = np.amax([val1, val2, val3])

        else:
            true_range = np.amax(prices) - np.amin(prices)

        data.set_value(index, 'true_range', true_range)
    data['atr'] = data['true_range'].ewm(ignore_na=False, min_periods=0, com=trend_periods, adjust=True).mean()
    if drop_tr:
        data = data.drop(['true_range'], axis=1)
        
    return data

"""
Bollinger Bands
Source: https://en.wikipedia.org/wiki/Bollinger_Bands
Params: 
    data: pandas DataFrame
    trend_periods: the over which to calculate BB
	close_col: the name of the CLOSE values column
    
Returns:
    copy of 'data' DataFrame with 'bol_bands_middle', 'bol_bands_upper' and 'bol_bands_lower' columns added
"""
def bollinger_bands(data, trend_periods=20, close_col='<CLOSE>'):

    data['bol_bands_middle'] = data[close_col].ewm(ignore_na=False, min_periods=0, com=trend_periods, adjust=True).mean()
    for index, row in data.iterrows():

        s = data[close_col].iloc[index - trend_periods: index]
        sums = 0
        middle_band = data.at[index, 'bol_bands_middle']
        for e in s:
            sums += np.square(e - middle_band)

        std = np.sqrt(sums / trend_periods)
        d = 2
        upper_band = middle_band + (d * std)
        lower_band = middle_band - (d * std)

        data.set_value(index, 'bol_bands_upper', upper_band)
        data.set_value(index, 'bol_bands_lower', lower_band)

    return data

"""
Chaikin Oscillator
Source: http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_oscillator
Params: 
    data: pandas DataFrame
	periods_short: period for the shorter EMA (3 days recommended)
	periods_long: period for the longer EMA (10 days recommended)
	high_col: the name of the HIGH values column
	low_col: the name of the LOW values column
	close_col: the name of the CLOSE values column
	vol_col: the name of the VOL values column
    
Returns:
    copy of 'data' DataFrame with 'ch_osc' column added
"""
def chaikin_oscillator(data, periods_short=3, periods_long=10, high_col='<HIGH>',
                       low_col='<LOW>', close_col='<CLOSE>', vol_col='<VOL>'):
    ac = pd.Series([])
    val_last = 0
	
    for index, row in data.iterrows():
        if row[high_col] != row[low_col]:
            val = val_last + ((row[close_col] - row[low_col]) - (row[high_col] - row[close_col])) / (row[high_col] - row[low_col]) * row[vol_col]
        else:
            val = val_last
            ac.set_value(index, val)
            
    val_last = val

    ema_long = ac.ewm(ignore_na=False, min_periods=0, com=periods_long, adjust=True).mean()
    ema_short = ac.ewm(ignore_na=False, min_periods=0, com=periods_short, adjust=True).mean()
    data['ch_osc'] = ema_short - ema_long

    return data

"""
Typical Price
Source: https://en.wikipedia.org/wiki/Typical_price
Params: 
    data: pandas DataFrame
	high_col: the name of the HIGH values column
	low_col: the name of the LOW values column
	close_col: the name of the CLOSE values column
    
Returns:
    copy of 'data' DataFrame with 'typical_price' column added
"""
def typical_price(data, high_col = '<HIGH>', low_col = '<LOW>', close_col = '<CLOSE>'):
    
    data['typical_price'] = (data[high_col] + data[low_col] + data[close_col]) / 3

    return data

"""
Ease of Movement
Source: http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ease_of_movement_emv
Params: 
    data: pandas DataFrame
	period: period for calculating EMV
	high_col: the name of the HIGH values column
	low_col: the name of the LOW values column
	vol_col: the name of the VOL values column
    
Returns:
    copy of 'data' DataFrame with 'emv' and 'emv_ema_[period]' columns added
"""
def ease_of_movement(data, period=14, high_col='<HIGH>', low_col='<LOW>', vol_col='<VOL>'):
    for index, row in data.iterrows():
        if index > 0:
            midpoint_move = (row[high_col] + row[low_col]) / 2 - (data.at[index - 1, high_col] + data.at[index - 1, low_col]) / 2
        else:
            midpoint_move = 0
        
        diff = row[high_col] - row[low_col]
		
        if diff == 0:
			#this is to avoid division by zero below
            diff = 0.000000001			
            
        vol = row[vol_col]
        if vol == 0:
            vol = 1
        box_ratio = (vol / 100000000) / (diff)
        emv = midpoint_move / box_ratio
        
        data.set_value(index, 'emv', emv)
        
    data['emv_ema_'+str(period)] = data['emv'].ewm(ignore_na=False, min_periods=0, com=period, adjust=True).mean()
        
    return data

"""
Mass Index
Source: http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:mass_index
Params: 
    data: pandas DataFrame
	period: period for calculating MI (9 days recommended)
	high_col: the name of the HIGH values column
	low_col: the name of the LOW values column
    
Returns:
    copy of 'data' DataFrame with 'mass_index' column added
"""
def mass_index(data, period=25, ema_period=9, high_col='<HIGH>', low_col='<LOW>'):
    high_low = data[high_col] - data[low_col] + 0.000001	#this is to avoid division by zero below
    ema = high_low.ewm(ignore_na=False, min_periods=0, com=ema_period, adjust=True).mean()
    ema_ema = ema.ewm(ignore_na=False, min_periods=0, com=ema_period, adjust=True).mean()
    div = ema / ema_ema

    for index, row in data.iterrows():
        if index >= period:
            val = div[index-25:index].sum()
        else:
            val = 0
        data.set_value(index, 'mass_index', val)
         
    return data

"""
Average directional movement index
Source: https://en.wikipedia.org/wiki/Average_directional_movement_index
Params: 
    data: pandas DataFrame
	periods: period for calculating ADX (14 days recommended)
	high_col: the name of the HIGH values column
	low_col: the name of the LOW values column
    
Returns:
    copy of 'data' DataFrame with 'adx', 'dxi', 'di_plus', 'di_minus' columns added
"""
def directional_movement_index(data, periods=14, high_col='<HIGH>', low_col='<LOW>'):
    remove_tr_col = False
    if not 'true_range' in data.columns:
        data = average_true_range(data, drop_tr = False)
        remove_tr_col = True

    data['m_plus'] = 0.
    data['m_minus'] = 0.
    
    for i,row in data.iterrows():
        if i>0:
            data.set_value(i, 'm_plus', row[high_col] - data.at[i-1, high_col])
            data.set_value(i, 'm_minus', row[low_col] - data.at[i-1, low_col])
    
    data['dm_plus'] = 0.
    data['dm_minus'] = 0.
    
    for i,row in data.iterrows():
        if row['m_plus'] > row['m_minus'] and row['m_plus'] > 0:
            data.set_value(i, 'dm_plus', row['m_plus'])
            
        if row['m_minus'] > row['m_plus'] and row['m_minus'] > 0:
            data.set_value(i, 'dm_minus', row['m_minus'])
    
    data['di_plus'] = (data['dm_plus'] / data['true_range']).ewm(ignore_na=False, min_periods=0, com=periods, adjust=True).mean()
    data['di_minus'] = (data['dm_minus'] / data['true_range']).ewm(ignore_na=False, min_periods=0, com=periods, adjust=True).mean()
    
    data['dxi'] = np.abs(data['di_plus'] - data['di_minus']) / (data['di_plus'] + data['di_minus'])
    data.set_value(0, 'dxi',1.)
    data['adx'] = data['dxi'].ewm(ignore_na=False, min_periods=0, com=periods, adjust=True).mean()
    data = data.drop(['m_plus', 'm_minus', 'dm_plus', 'dm_minus'], axis=1)
    if remove_tr_col:
        data = data.drop(['true_range'], axis=1)
         
    return data

"""
Money Flow Index (MFI)
Source: http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:money_flow_index_mfi
Params: 
    data: pandas DataFrame
	periods: period for calculating MFI (14 days recommended)
	vol_col: the name of the VOL values column
    
Returns:
    copy of 'data' DataFrame with 'money_flow_index' column added
"""
def money_flow_index(data, periods=14, vol_col='<VOL>'):
    remove_tp_col = False
    if not 'typical_price' in data.columns:
        data = typical_price(data)
        remove_tp_col = True
    
    data['money_flow'] = data['typical_price'] * data[vol_col]
    data['money_ratio'] = 0.
    data['money_flow_index'] = 0.
    data['money_flow_positive'] = 0.
    data['money_flow_negative'] = 0.
    
    for index,row in data.iterrows():
        if index > 0:
            if row['typical_price'] < data.at[index-1, 'typical_price']:
                data.set_value(index, 'money_flow_positive', row['money_flow'])
            else:
                data.set_value(index, 'money_flow_negative', row['money_flow'])
    
        if index >= periods:
            period_slice = data['money_flow'][index-periods:index]
            positive_sum = data['money_flow_positive'][index-periods:index].sum()
            negative_sum = data['money_flow_negative'][index-periods:index].sum()

            if negative_sum == 0.:
				#this is to avoid division by zero below
                negative_sum = 0.00001
            m_r = positive_sum / negative_sum

            mfi = 1-(1 / (1 + m_r))

            data.set_value(index, 'money_ratio', m_r)
            data.set_value(index, 'money_flow_index', mfi)
          
    data = data.drop(['money_flow', 'money_ratio', 'money_flow_positive', 'money_flow_negative'], axis=1)
    
    if remove_tp_col:
        data = data.drop(['typical_price'], axis=1)

    return data

"""
Negative Volume Index (NVI)
Source: http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:negative_volume_inde
Params: 
    data: pandas DataFrame
	periods: period for calculating NVI (255 days recommended)
	close_col: the name of the CLOSE values column
	vol_col: the name of the VOL values column
    
Returns:
    copy of 'data' DataFrame with 'nvi' and 'nvi_ema' columns added
"""
def negative_volume_index(data, periods=255, close_col='<CLOSE>', vol_col='<VOL>'):
    data['nvi'] = 0.
    
    for index,row in data.iterrows():
        if index > 0:
            prev_nvi = data.at[index-1, 'nvi']
            prev_close = data.at[index-1, close_col]
            if row[vol_col] < data.at[index-1, vol_col]:
                nvi = prev_nvi + (row[close_col] - prev_close / prev_close * prev_nvi)
            else: 
                nvi = prev_nvi
        else:
            nvi = 1000
        data.set_value(index, 'nvi', nvi)
    data['nvi_ema'] = data['nvi'].ewm(ignore_na=False, min_periods=0, com=periods, adjust=True).mean()
    
    return data

"""
Positive Volume Index (PVI)
Source: https://www.equities.com/news/the-secret-to-the-positive-volume-index
Params: 
    data: pandas DataFrame
	periods: period for calculating PVI (255 days recommended)
	close_col: the name of the CLOSE values column
	vol_col: the name of the VOL values column
    
Returns:
    copy of 'data' DataFrame with 'pvi' and 'pvi_ema' columns added
"""
def positive_volume_index(data, periods=255, close_col='<CLOSE>', vol_col='<VOL>'):
    data['pvi'] = 0.
    
    for index,row in data.iterrows():
        if index > 0:
            prev_pvi = data.at[index-1, 'pvi']
            prev_close = data.at[index-1, close_col]
            if row[vol_col] > data.at[index-1, vol_col]:
                pvi = prev_pvi + (row[close_col] - prev_close / prev_close * prev_pvi)
            else: 
                pvi = prev_pvi
        else:
            pvi = 1000
        data.set_value(index, 'pvi', pvi)
    data['pvi_ema'] = data['pvi'].ewm(ignore_na=False, min_periods=0, com=periods, adjust=True).mean()

    return data

"""
Momentum
Source: https://en.wikipedia.org/wiki/Momentum_(technical_analysis)
Params: 
    data: pandas DataFrame
	periods: period for calculating momentum
	close_col: the name of the CLOSE values column
    
Returns:
    copy of 'data' DataFrame with 'momentum' column added
"""
def momentum(data, periods=14, close_col='<CLOSE>'):
    data['momentum'] = 0.
    
    for index,row in data.iterrows():
        if index >= periods:
            prev_close = data.at[index-periods, close_col]
            val_perc = (row[close_col] - prev_close)/prev_close

            data.set_value(index, 'momentum', val_perc)

    return data

"""
Relative Strenght Index
Source: https://en.wikipedia.org/wiki/Relative_strength_index
Params: 
    data: pandas DataFrame
	periods: period for calculating momentum
	close_col: the name of the CLOSE values column
    
Returns:
    copy of 'data' DataFrame with 'rsi' column added
"""
def rsi(data, periods=14, close_col='<CLOSE>'):
    data['rsi_u'] = 0.
    data['rsi_d'] = 0.
    data['rsi'] = 0.
    
    for index,row in data.iterrows():
        if index >= periods:
            
            prev_close = data.at[index-periods, close_col]
            if prev_close < row[close_col]:
                data.set_value(index, 'rsi_u', row[close_col] - prev_close)
            elif prev_close > row[close_col]:
                data.set_value(index, 'rsi_d', prev_close - row[close_col])
            
    data['rsi'] = data['rsi_u'].ewm(ignore_na=False, min_periods=0, com=periods, adjust=True).mean() / (data['rsi_u'].ewm(ignore_na=False, min_periods=0, com=periods, adjust=True).mean() + data['rsi_d'].ewm(ignore_na=False, min_periods=0, com=periods, adjust=True).mean())
    
    data = data.drop(['rsi_u', 'rsi_d'], axis=1)
        
    return data

"""
Chaikin Volatility (CV)
Source: https://www.marketvolume.com/technicalanalysis/chaikinvolatility.asp
Params: 
    data: pandas DataFrame
	ema_periods: period for smoothing Highest High and Lowest Low difference
	change_periods: the period for calculating the difference between Highest High and Lowest Low
	high_col: the name of the HIGH values column
	low_col: the name of the LOW values column
	close_col: the name of the CLOSE values column
    
Returns:
    copy of 'data' DataFrame with 'chaikin_volatility' column added
"""
def chaikin_volatility(data, ema_periods=10, change_periods=10, high_col='<HIGH>', low_col='<LOW>', close_col='<CLOSE>'):
    data['ch_vol_hl'] = data[high_col] - data[low_col]
    data['ch_vol_ema'] = data['ch_vol_hl'].ewm(ignore_na=False, min_periods=0, com=ema_periods, adjust=True).mean()
    data['chaikin_volatility'] = 0.
    
    for index,row in data.iterrows():
        if index >= change_periods:
            
            prev_value = data.at[index-change_periods, 'ch_vol_ema']
            if prev_value == 0:
                #this is to avoid division by zero below
                prev_value = 0.0001
            data.set_value(index, 'chaikin_volatility', ((row['ch_vol_ema'] - prev_value)/prev_value))
            
    data = data.drop(['ch_vol_hl', 'ch_vol_ema'], axis=1)
        
    return data

"""
William's Accumulation/Distribution
Source: https://www.metastock.com/customer/resources/taaz/?p=125
Params: 
    data: pandas DataFrame
	high_col: the name of the HIGH values column
	low_col: the name of the LOW values column
	close_col: the name of the CLOSE values column
    
Returns:
    copy of 'data' DataFrame with 'williams_ad' column added
"""
def williams_ad(data, high_col='<HIGH>', low_col='<LOW>', close_col='<CLOSE>'):
    data['williams_ad'] = 0.
    
    for index,row in data.iterrows():
        if index > 0:
            prev_value = data.at[index-1, 'williams_ad']
            prev_close = data.at[index-1, close_col]
            if row[close_col] > prev_close:
                ad = row[close_col] - min(prev_close, row[low_col])
            elif row[close_col] < prev_close:
                ad = row[close_col] - max(prev_close, row[high_col])
            else:
                ad = 0.
                                                                                                        
            data.set_value(index, 'williams_ad', (ad+prev_value))
        
    return data

"""
William's % R
Source: https://www.metastock.com/customer/resources/taaz/?p=126
Params: 
    data: pandas DataFrame
	periods: the period over which to calculate the indicator value
	high_col: the name of the HIGH values column
	low_col: the name of the LOW values column
	close_col: the name of the CLOSE values column
    
Returns:
    copy of 'data' DataFrame with 'williams_r' column added
"""
def williams_r(data, periods=14, high_col='<HIGH>', low_col='<LOW>', close_col='<CLOSE>'):
    data['williams_r'] = 0.
    
    for index,row in data.iterrows():
        if index > periods:
            data.set_value(index, 'williams_r', ((max(data[high_col][index-periods:index]) - row[close_col]) / 
                                                 (max(data[high_col][index-periods:index]) - min(data[low_col][index-periods:index]))))
        
    return data

"""
TRIX
Source: https://www.metastock.com/customer/resources/taaz/?p=114
Params: 
    data: pandas DataFrame
	periods: the period over which to calculate the indicator value
	signal_periods: the period for signal moving average
	close_col: the name of the CLOSE values column
    
Returns:
    copy of 'data' DataFrame with 'trix' and 'trix_signal' columns added
"""
def trix(data, periods=14, signal_periods=9, close_col='<CLOSE>'):
    data['trix'] = data[close_col].ewm(ignore_na=False, min_periods=0, com=periods, adjust=True).mean()
    data['trix'] = data['trix'].ewm(ignore_na=False, min_periods=0, com=periods, adjust=True).mean()
    data['trix'] = data['trix'].ewm(ignore_na=False, min_periods=0, com=periods, adjust=True).mean()
    data['trix_signal'] = data['trix'].ewm(ignore_na=False, min_periods=0, com=signal_periods, adjust=True).mean()
        
    return data

"""
Ultimate Oscillator
Source: http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ultimate_oscillator
Params: 
    data: pandas DataFrame
	period_1: the period of the first average (7 days recommended)
	period_2: the period of the second average (14 days recommended)
	period_3: the period of the third average (28 days recommended)
	high_col: the name of the HIGH values column
	low_col: the name of the LOW values column
	close_col: the name of the CLOSE values column
    
Returns:
    copy of 'data' DataFrame with 'ultimate_oscillator' column added
"""
def ultimate_oscillator(data, period_1=7,period_2=14, period_3=28, high_col='<HIGH>', low_col='<LOW>', close_col='<CLOSE>'):
    data['ultimate_oscillator'] = 0.
    data['uo_bp'] = 0.
    data['uo_tr'] = 0.
    data['uo_avg_1'] = 0.
    data['uo_avg_2'] = 0.
    data['uo_avg_3'] = 0.

    for index,row in data.iterrows():
        if index > 0:
                           
            bp = row[close_col] - min(row[low_col], data.at[index-1, close_col])
            tr = max(row[high_col], data.at[index-1, close_col]) - min(row[low_col], data.at[index-1, close_col])
            
            data.set_value(index, 'uo_bp', bp)
            data.set_value(index, 'uo_tr', tr)
            if index >= period_1:
                uo_avg_1 = sum(data['uo_bp'][index-period_1:index]) / sum(data['uo_tr'][index-period_1:index])
                data.set_value(index, 'uo_avg_1', uo_avg_1)
            if index >= period_2:
                uo_avg_2 = sum(data['uo_bp'][index-period_2:index]) / sum(data['uo_tr'][index-period_2:index])
                data.set_value(index, 'uo_avg_2', uo_avg_2)
            if index >= period_3:
                uo_avg_3 = sum(data['uo_bp'][index-period_3:index]) / sum(data['uo_tr'][index-period_3:index])
                data.set_value(index, 'uo_avg_3', uo_avg_3)
                uo = (4 * uo_avg_1 + 2 * uo_avg_2 + uo_avg_3) / 7
                data.set_value(index, 'ultimate_oscillator', uo)

    data = data.drop(['uo_bp', 'uo_tr', 'uo_avg_1', 'uo_avg_2', 'uo_avg_3'], axis=1)
        
    return data   



"""
Copyright, 404 KKK , 2045.
License: ночной рыцарь лицензия
"""
from google.cloud import bigquery
from google.oauth2 import service_account

def connect():
	credentials = service_account.Credentials.from_service_account_file(r'C:\Users\kevin.puika\OneDrive - Bank Jasa Jakarta\Desktop\KRP-CKPN\joey-bi-ss-risk-fraud-project-36e37a4eccd7 1.json')
	project_id = "joey-bi-ss-risk-fraud-project"
	client = bigquery.Client(credentials= credentials,project=project_id)

	return client


def shortlisted(client_con):
	sql = '''with shortlisted as (
  select mis_date,
         t24_customer_id
  from joey-bi-prod-project.staging.stg_loan__sake_credit_limit_shortlisted
  where mis_date = last_day(mis_date)
  group by mis_date, t24_customer_id
),
funded_detail as (
  select t24_customer_id,
        min(ever_funded) as ever_funded
  from ( select t24_customer_id,
                ever_funded
        from joey-bi-ss-risk-fraud-project.credit.funded_info_2025_06
        
        UNION ALL

        select t24_customer_id,
                ever_funded
        from joey-bi-ss-risk-fraud-project.credit.funded_info_2025_07
  )
  group by t24_customer_id
),
combine as (
  select a.*,
        case when b.ever_funded is null then 0
		else ever_funded end as ever_funded
  from shortlisted a
  left join funded_detail b on a.t24_customer_id = cast(b.t24_customer_id as string)
),
combine_ntb as(
  select substr(string(mis_date),1,10) as mis_date,
        cast(ever_funded as string) as flagging,
        t24_customer_id
  from combine

  UNION ALL
  
  select substr(string(onboard_date),1,10) as mis_date,
        "NTB" as flagging,
        t24_customer_id
  from joey-bi-prod-project.staging.ntb_prefilter_saku_kredit
  
)
select substr(mis_date,1,7) as mis_date,
        flagging,
        count(distinct t24_customer_id) as count_client
from combine_ntb
group by mis_date,
        flagging
'''
	shortlisted = client_con.query(sql).to_dataframe()
	return shortlisted




def Incoming_apps(client_con):
	
	##DECLARE date1 STRING DEFAULT '2025-07-31';

	sql = '''with base_limit as (
  select mis_date,
        t24_customer_id,
        limit_id,
        substr(string(created_at),1,10) as create_date,
        substr(string(created_at),1,7) as create_mth,
        reason,
        case 
          when reason in ('["API ERROR"]','["The account opening process has not been completed"]','["CB Large Exposure"]','["Exceed Hard Cap (>5% from Hard Cap Buffer)"]') then reason
          else substr(reason,3,6) end as first_reason,
        status,
        product_code,
        case
          when limit_amount < 0 then 0
          else limit_amount end as limit_amount,
        deviation_level
  from joey-bi-prod-project.staging.stg_loan__personal_loan_sys_limit_application
  where string(mis_date) = date1
),
flagging_fund_ntb as (
  select cast(t24_customer_id as string) as t24_customer_id,
        min(ever_funded) as flagging
  from ( select t24_customer_id,
                ever_funded
        from joey-bi-ss-risk-fraud-project.credit.funded_info_2025_06
        
        UNION ALL

        select t24_customer_id,
                ever_funded
        from joey-bi-ss-risk-fraud-project.credit.funded_info_2025_08
  )
  group by t24_customer_id

  union ALL
  
  select t24_customer_id,
        max(2) as flagging --NTB
  from joey-bi-prod-project.staging.ntb_prefilter_saku_kredit
  group by t24_customer_id
),
flagging_grp as (
  select t24_customer_id,
      case
        when max(flagging) = 2 then 'NTB'
        when max(flagging) = 1 then '1'
        when max(flagging) = 0 then '0'
        else '' end as flagging
  from flagging_fund_ntb
  group by t24_customer_id
),
transact_flag as (
  select  limit_id,
          mis_date,
          1 as transact_flag
  from joey-bi-prod-project.staging.stg_collection__limit_loan
  where string(mis_date) = date1
  group by limit_id , mis_date
),
rej_reason_dict as (
  select *
  from joey-bi-ss-risk-fraud-project.credit.Saku_kredit_reject_reason_desc_2025_06_30
),
onboarding_req as (
  select mis_date,
          t24_customer_id,
          province,
          REGEXP_EXTRACT(email, r'@(.+)') AS email_domain,
          channel,
          gender,
          customer_range_age,
          phone_number,
          case
            when substr(phone_number,1,5) in ('62812','62821','62822',
                              '62823','62813','62852','62851','62811',
                              '62853') then '1. Telkomsel'
            when substr(phone_number,1,5) in ('62856','62857','62815',
                              '62858','62814','62855','62816') then '2. Indosat'
            when substr(phone_number,1,5) in ('62817','62818','62877',
                              '62819','62878','62859') then '3. XL'
            when substr(phone_number,1,5) in ('62881','62882','62888',
                              '62889','62887') then '4. Smartfren'
            when substr(phone_number,1,5) in ('62838','62831') then '5. Axis'
            when substr(phone_number,1,5) in ('62899','62898','62897',
                              '62896','62895') then '6. Tri'
            else '9. Others' 
            end as No_Provider
          
  from joey-bi-prod-project.staging.stg_onboarding__onboarding_request__hive_merge
  where string(mis_date) = date1
),
cred_summary as (
  select mis_date,
          application_id,
          credit_bureau_score as clik_bur_score,
          credit_bureau_risk_level as clik_risk_lvl,
          bureau_type as clik_bureau_info,
          income_verified as izi_income_verification,
          home_address_verified as aai_home_verification,
          office_address_verified as aai_office_verification,
          alternative_advance_ai_score as bps_score,
          final_risk_level
  from joey-bi-prod-project.staging.stg_loan__loan_credit_summary
  where string(mis_date) = date1
),
combine as (
  select a.*,
        case 
          when b.flagging is null then "0"
		      else b.flagging end as flagging,
        case 
          when c.transact_flag is null then 0
          else c.transact_flag end as transact_flag,
        d.DescCode,
        e.province,
          e.email_domain,
          e.channel,
          e.gender,
          e.customer_range_age,
          e.phone_number,
          e.No_Provider,
          f.clik_bur_score,
          f.clik_risk_lvl,
          f.clik_bureau_info,
          f.izi_income_verification,
          f.aai_home_verification,
          f.aai_office_verification,
          f.bps_score,
          f.final_risk_level
  from base_limit a
  left join flagging_grp b on a.t24_customer_id = cast(b.t24_customer_id as string)
  left join transact_flag c 
          on a.limit_id = c.limit_id
          and a.mis_date = c.mis_date
  left join rej_reason_dict d
          on a.reason = d.Reason
  left join onboarding_req e 
          on a.t24_customer_id = cast(e.t24_customer_id as string) 
          and a.mis_date = e.mis_date
  left join cred_summary f on a.limit_id = f.application_id 
          and a.mis_date = f.mis_date
),
comb_band as (
  select *,
        case
          when status in ('agreementSigned','approved') then '1. Approved/Signed'
          when status in ('rejected') then '2. Rejected'
          when status in ('incomplete') then '3. Incomplete'
          else '9. Others' end as status_grp,
        case
          when reason is null then reason
          when first_reason in ('SCA001','SCB002','SLC001','SCO002','SCA002',
          'SCA013','SCA015','["API ERROR"]','SOT007','SCA005') then first_reason
          else '9. Others' end as first_reason_grp,
        case
          when limit_amount =         0 then '01. 0'
          when limit_amount =    500000 then '02. Base Limit: 500K'
          when limit_amount =   1000000 then '03. Base Limit: 1 Mio'
          when limit_amount =   2000000 then '04. Base Limit: 2 Mio'
          when limit_amount <=  5000000 then '05. <= 5 Mio'
          when limit_amount <= 10000000 then '06. (5Mio - 10Mio]'
          when limit_amount <= 15000000 then '07. (10Mio - 15Mio]'
          when limit_amount <= 20000000 then '08. (15Mio - 20Mio]'
          when limit_amount <= 28000000 then '09. (20Mio - 28Mio]'
          when limit_amount >  28000000 then '10. > 28 Mio'
          else '9. Else' end as limit_band_grp,
        case
          when status in ('agreementSigned') then 1 
            else 0 end as activated_limit_flag,
        case
          when email_domain in ('gmail.com') then '1. Gmail'
          when email_domain in ('yahoo.com','ymail.com','yahoo.co.id') then '2. Yahoo'
          when email_domain in ('icloud.com') then '3. Apple-icloud'
          when email_domain in ('outlook.com','hotmail.com','windowslive.com') then '4. outlook/hotmail/windows'
          else '9. Others' end as email_domain_grp,
        case
          when clik_bur_score is null then '00. Null'
          when clik_bur_score  =   0 then '00. 0'
          when clik_bur_score <= 149 then '00. 1 - 149'
          when clik_bur_score <= 267 then '01. 150 - 267'
          when clik_bur_score <= 379 then '02. 268 - 379'
          when clik_bur_score <= 519 then '03. 380 - 519'
          when clik_bur_score <= 545 then '04. 520 - 545'
          when clik_bur_score <= 561 then '05. 546 - 561'
          when clik_bur_score  > 561 then '06. above 561' --up to 659 based on docs....
          else '99. else' end as clik_bur_score_grp_2,
        case
          when bps_score is null then '00. null'
          when bps_score  =    0 then '00. 0'
          when bps_score <=  499 then '01. 370 - 499'
          when bps_score <=  530 then '02. 500 - 530'
          when bps_score <=  559 then '03. 531 - 559'
          when bps_score <=  593 then '04. 560 - 593'
          when bps_score <=  627 then '05. 594 - 627'
          when bps_score <=  806 then '06. 628 - 806'
          when bps_score >   806 then '07. above 806'
          else '99. Else' end as bps_score_grp_2,
        case
          when province in ('BANTEN','DKI JAKARTA','JAWA BARAT',
                          'JAWA TENGAH','JAWA TIMUR') then province
          when province in ('ACEH','BENGKULU', 'JAMBI', 'KEP. BANGKA BELITUNG', 
                          'LAMPUNG','RIAU') then 'SUMATERA'
          when province like '%SUMATERA%' then 'SUMATERA'
          when province in ('BALI') then 'BALI'
          when province in ('DAERAH ISTIMEWA YOGYAKARTA') then 'JAWA TENGAH'
          when province like '%KALIMANTAN%' then 'KALIMANTAN'
          when province in ('MALUKU','NUSA TENGGARA BARAT','PAPUA') then 'Indonesia Bagian Timur'
          when province like '%SULAWESI%' then 'Indonesia Bagian Timur'
          else '99. Others' end as prov_grp,
        case
          when reason like '%SCB001%' then 'CB already bad'
          when reason like '%SCB002%' then 'CB DPD>0 in L3M'
          when reason like '%SCB003%' then 'CB Ever 30+ in L12M'
          when reason like '%SCB004%' then 'CB Restructure'
          when reason like '%SCB005%' then 'CB DPD OS'
          when reason like '%SCB006%' then 'CB Ever DPD'
          else 'Others' end as CB_Bur_reason

  from combine
)
, result_incoming as (
select substr(string(mis_date),1,7) as mis_date,
        create_mth,
        DescCode,
        status_grp,
        product_code,
        limit_band_grp,
        sum(limit_amount) as total_limit,
        sum(1) as count_apps,
        count(distinct t24_customer_id) as count_client,
        flagging,
        activated_limit_flag,
        transact_flag,
        deviation_level
from comb_band
where status_grp in ('1. Approved/Signed','2. Rejected')
group by substr(string(mis_date),1,7),
        create_mth,
        DescCode,
        status_grp,
        product_code,
        limit_band_grp,
        flagging,
        activated_limit_flag,
        transact_flag,
        deviation_level
),
upd_rej_reason as (
  select t24_customer_id,
        create_mth,
        reason,
        status
  from comb_band
  where DescCode is null
      and status = 'rejected'
      and reason is not null
)
select *
from 
    result_incoming
    -- upd_rej_reason
'''
	incoming_apps = client_con.query(sql).to_dataframe()
	return incoming_apps


def disbursed(client_con):
	
	##DECLARE date1 STRING DEFAULT '2025-07-31';

	sql = '''
With disb_base as (
  select arrangement_id,
          limit_id,
          limit_loan_amount as disb_amt,
          tenor,
          product_code,
          collectability,
          created_at as disb_at,
          substr(cast(created_at as STRING),0,10) as disb_dt,
          concat(substr(string(created_at),1,4),'-',substr(string(created_at),6,2)) as disb_mth,
          extract(hour from created_at) as disb_hr
  from joey-bi-prod-project.staging.stg_collection__limit_loan disb
  where STRING(disb.mis_date)= date1
),
limit_base as (
  select 
      case 
          when datetime(submitted_at) < '2024-09-03T00:00:00.000000' then 'Internal Launch'
          when datetime(submitted_at) >= '2024-09-03T00:00:00.000000' and datetime(submitted_at) < '2024-11-18T13:30:00.000000' then '3K Public Launch'
          when datetime(submitted_at) >= '2024-11-18T13:30:00.000000' and datetime(submitted_at) < '2024-11-25T00:00:00.000000' then 'Solopreneur'
          when datetime(submitted_at) >= '2024-11-25T00:00:00.000000' and datetime(submitted_at) < '2024-12-18T22:00:00.000000' then '22 Public Launch' 
          when datetime(submitted_at) >= '2024-12-18T22:00:00.000000' and datetime(submitted_at) < '2025-01-20T00:00:00.000000' then '112K Public Launch' 
          when datetime(submitted_at) >= '2025-01-20T00:00:00.000000' then '250K Public Launch' 
          end as phase,
      t24_customer_id,
      limit_id,
      cast(limit_amount as numeric) as limit_amt,
      date(datetime(created_at)) as create_dt,
      date(datetime(submitted_at)) as submit_dt,
      extract(hour from datetime(submitted_at)) as submit_hour,
      date(datetime(agreement_signed_at)) as signed_at,
      deviation_level
  from joey-bi-prod-project.staging.stg_loan__personal_loan_sys_limit_application
  where STRING(mis_date)=date1
),
fund_info as  (
  select *
  from (select *,
           RANK() OVER (
                PARTITION BY t24_customer_id 
                ORDER BY eligible_date DESC) AS data_rank 
        from joey-bi-ss-risk-fraud-project.credit.eligible_fund_info) 
        where data_rank = 1
),
flagging_fund_ntb as (
  select cast(t24_customer_id as string) as t24_customer_id,
        min(ever_funded) as flagging,
        -- max(uninstall) as uninstall
  from ( select t24_customer_id,
                ever_funded,
                -- uninstall
        from joey-bi-ss-risk-fraud-project.credit.funded_info_2025_06
        
        UNION ALL

        select t24_customer_id,
                ever_funded,
                -- cast(uninstall as int)
        from joey-bi-ss-risk-fraud-project.credit.funded_info_2025_08
  )  group by t24_customer_id

  union ALL
  
  select t24_customer_id,
        max(2) as flagging, --NTB
        -- max(0) as uninstall
  from joey-bi-prod-project.staging.ntb_prefilter_saku_kredit
  group by t24_customer_id
),
flagging_grp as (
  select t24_customer_id,
      case
        when max(flagging) = 2 then 'NTB'
        when max(flagging) = 1 then '1'
        when max(flagging) = 0 then '0'
        else '' end as flagging,
      -- max(uninstall) as uninstall
  from flagging_fund_ntb
  group by t24_customer_id
),
repay_data as (
  select a.arrangement_id,
          a.mis_date,
          sum(a.owed_amount) as sum_owed,
          b.limit_id
  from joey-bi-prod-project.staging.stg_collection__repayment_schedule a
  left join joey-bi-prod-project.staging.stg_collection__limit_loan b
        on a.arrangement_id = b.arrangement_id
        and String(b.mis_date) = date1
  where a.mis_date = last_day(a.mis_date)
  group by a.arrangement_id, a.mis_date, b.limit_id
),
repay_data_cust as (
  select limit_id,
        mis_date,
        sum(sum_owed) as os_client
  from repay_data
  group by limit_id, mis_date
),
combine as (
  select a.*,
          b.*,
          c.funded_info,
          case
            when d.flagging is null then "0"
            else d.flagging end as flagging,
          -- d.uninstall,
          e.os_client
  from disb_base a
  left join limit_base b on a.limit_id = cast(b.limit_id as STRING)
  left join fund_info c on b.t24_customer_id=cast(c.t24_customer_id as string)
  left join flagging_grp d on b.t24_customer_id = cast(d.t24_customer_id as string) 
  left join repay_data_cust e on a.limit_id = e.limit_id
        and last_day(cast(a.disb_dt as date)) = e.mis_date
),
comb_band as (
  select *,
        case
          when tenor = 1 then 'tenor 1'
          when tenor <= 3 then 'tenor 2-3'
          when tenor <= 6 then 'tenor 4-6'
          when tenor <= 11 then 'tenor 7-11'
          when tenor = 12 then 'tenor 12'
          else  'else' end as tenor_grp,
        case
          when os_client = 0 then '0. 0%'
          when (os_client/limit_amt) = 0.00 then '1. 0%'
          when (os_client/limit_amt) <= 0.20 then '2. 1 - 20%'
          when (os_client/limit_amt) <= 0.40 then '3. 21 - 40%'
          when (os_client/limit_amt) <= 0.60 then '4. 41 - 60%'
          when (os_client/limit_amt) <= 0.80 then '5. 61 - 80%'
          when (os_client/limit_amt) <= 1.00 then '6. 81 - 100%'
          else '9. Else' end as util_grp
          
  from combine
)
select funded_info,
      phase,
      tenor_grp,
      product_code,
      disb_mth,
      sum(disb_amt) as total_disb,
      sum(1) as count_loan,
      count(distinct t24_customer_id) as count_client,
      flagging,
      0, -- uninstall,
      0, -- placeholder
      deviation_level,
      util_grp
from comb_band
group by funded_info,
      phase,
      tenor_grp,
      product_code,
      disb_mth,
      flagging,
      -- uninstall,
      deviation_level,
      util_grp
'''
	disbursed = client_con.query(sql).to_dataframe()
	return disbursed

def Porto_DPD(client_con):
	
	##DECLARE date1 STRING DEFAULT '2025-07-31';

	sql = '''
with disb_base as (
  select mis_date,
        arrangement_id,
        limit_id,
        customer_info_id,
        limit_loan_amount,
        tenor,
        product_code,
        collectability,
        created_at,
        substr(cast(created_at as STRING),0,7) as disbym
  from joey-bi-prod-project.staging.stg_collection__limit_loan 
  where STRING(mis_date) = date1 
),
limit_base as (
  select *
  from joey-bi-prod-project.staging.stg_loan__personal_loan_sys_limit_application 
  where STRING(mis_date) = date1
),
repay_dpd_base as (
  select concat(substr(string(mis_date),1,4),'-',substr(string(mis_date),6,2)) as dpd_pt_date,
          arrangement_id,
         sum(owed_amount) as os_bal,
         max(dpd) as max_dpd,
         case when sum(owed_amount) =0 then 'closed' else 'active' end as loan_status,
         case
          when max(dpd) is null then '1. CURRENT'
          when max(dpd) = 0 then '1. CURRENT'
          when max(dpd) between 1 and 30 then '2. X-days'
          when max(dpd) between 31 and 60 then '3. DPD30'
          when max(dpd) between 61 and 90 then '4. DPD60'
          when max(dpd) between 91 and 120 then '5. DPD90'
          when max(dpd) between 121 and 150 then '6. DPD120'
          when max(dpd) between 151 and 180 then '7. DPD150'
          when max(dpd) >=181 then '8. DPD180/WO'
          end as bucket,
        sum(
          case when current_tenor=1 and dpd >0 then 1 
              else 0 end) as FPD1_flag,
        sum(
          case when current_tenor=1 and dpd >=7 then 1 
              else 0 end) as FPD7_flag,
        sum(
          case when current_tenor=1 and dpd >=14 then 1 
              else 0 end) as FPD14_flag,
        sum(
          case when current_tenor=1 and dpd >30 then 1 
              else 0 end) as FPD30_flag
  from joey-bi-prod-project.staging.stg_collection__repayment_schedule 
  where mis_date = last_day(mis_date)
  -- STRING(mis_date) = date1
  group by mis_date, arrangement_id 
),
flagging_fund_ntb as (
  select cast(t24_customer_id as string) as t24_customer_id,
        min(ever_funded) as flagging,
  from ( select t24_customer_id,
                ever_funded
        from joey-bi-ss-risk-fraud-project.credit.funded_info_2025_06
        
        UNION ALL

        select t24_customer_id,
                ever_funded
        from joey-bi-ss-risk-fraud-project.credit.funded_info_2025_08
  )  group by t24_customer_id

  union ALL
  
  select t24_customer_id,
        max(2) as flagging, --NTB
  from joey-bi-prod-project.staging.ntb_prefilter_saku_kredit
  group by t24_customer_id
),
flagging_grp as (
  select t24_customer_id,
      case
        when max(flagging) = 2 then 'NTB'
        when max(flagging) = 1 then '1'
        when max(flagging) = 0 then '0'
        else '' end as flagging
  from flagging_fund_ntb
  group by t24_customer_id
),
cred_summary as (
  select mis_date,
          application_id,
          final_risk_level
  from joey-bi-prod-project.staging.stg_loan__loan_credit_summary
  where string(mis_date) = date1
),
combine as(
  select a.*,
        cast(b.t24_customer_id as STRING) as CIF,
        c.dpd_pt_date,
        c.os_bal,
        c.max_dpd,
        c.loan_status,
        c.bucket,
        c.FPD1_flag,
        c.FPD7_flag,
        c.FPD14_flag,
        c.FPD30_flag,
        case 
          when d.flagging is null then "0"
          else d.flagging end as flagging,
        e.final_risk_level,
        case 
          when f.max_dpd is null then 0
          else f.max_dpd end as max_dpd_prev_mth
  from disb_base a
  left join limit_base b on a.limit_id = cast(b.limit_id as STRING)
  left join repay_dpd_base c on a.arrangement_id = c.arrangement_id
  left join flagging_grp d
        on b.t24_customer_id = cast(d.t24_customer_id as string)
  left join cred_summary e on a.limit_id = e.application_id 
  left join repay_dpd_base f on c.arrangement_id = f.arrangement_id
            -- c = 2024-02, f = 2024-01
        and cast(substr(c.dpd_pt_date,1,4) as int)*12 + cast(substr(c.dpd_pt_date,6,2) as int)
            = cast(substr(f.dpd_pt_date,1,4) as int)*12 + cast(substr(f.dpd_pt_date,6,2) as int) +1
),
comb_band as (
  select *,
        case
          when tenor = 1 then 'tenor 1'
          when tenor <= 3 then 'tenor 2-3'
          when tenor <= 6 then 'tenor 4-6'
          when tenor <= 11 then 'tenor 7-11'
          when tenor = 12 then 'tenor 12'
          else  'else' end as tenor_grp,
        case
          when (os_bal/limit_loan_amount) = 0.00 then '1. 0%'
          when (os_bal/limit_loan_amount) <= 0.20 then '2. 1 - 20%'
          when (os_bal/limit_loan_amount) <= 0.40 then '3. 21 - 40%'
          when (os_bal/limit_loan_amount) <= 0.60 then '4. 41 - 60%'
          when (os_bal/limit_loan_amount) <= 0.80 then '5. 61 - 80%'
          when (os_bal/limit_loan_amount) <= 1.00 then '6. 81 - 100%'
          else '9. Else' end as util_grp,
        case
          when FPD1_flag > 0 then os_bal
          else 0 end as FPD1_amt,
        case
          when FPD7_flag > 0 then os_bal
          else 0 end as FPD7_amt,
        case
          when FPD14_flag > 0 then os_bal
          else 0 end as FPD14_amt,
        case
          when FPD30_flag > 0 then os_bal
          else 0 end as FPD30_amt,
        case
          when max_dpd_prev_mth = 0 then 'Yes'
          when max_dpd_prev_mth >=  max_dpd then 'No'
          when (max_dpd - max_dpd_prev_mth) > 30
                and mod(max_dpd,30) = 1 then 'Skip bucket' 
          else 'Yes' end as Forward_looking_FR
  from combine
  where CIF not in ('101174407','101858872','102072000',
            '102084939','102072802','102096422') -- fraud
)
select dpd_pt_date,
        tenor_grp,
        product_code,
        bucket,
        loan_status,
        util_grp,
        sum(os_bal) as total_os,
       sum(limit_loan_amount) as total_limit,
       sum(1) as total_loan,
       count(distinct CIF) as total_cust,
       sum(FPD1_amt) as total_FPD1_amt,
       sum(FPD7_amt) as total_FPD7_amt,
       sum(FPD14_amt) as total_FPD14_amt,
       sum(FPD30_amt) as total_FPD30_amt,
       disbym,
       sum(FPD1_flag) as FPD1_count,
       sum(FPD7_flag) as FPD7_count,
       sum(FPD14_flag) as FPD14_count,
       sum(FPD30_flag) as FPD30_count,
       flagging,
       final_risk_level,
       Forward_looking_FR
from comb_band
group by dpd_pt_date,
        tenor_grp,
        product_code,
        bucket,
        loan_status,
        util_grp,
        disbym,
        flagging,
        final_risk_level,
        Forward_looking_FR
'''
	porto_DPD = client_con.query(sql).to_dataframe()
	return porto_DPD

def vintage_apps(client_con):
	
	##DECLARE date1 STRING DEFAULT '2025-07-31';

	sql = '''
with disb_data as (
  select mis_date as mis_pt_date,
        limit_id,
        limit_loan_amount,
        tenor,
        product_code,
        collectability,
        created_at,
        substr(cast(created_at as STRING),0,7) as disbym,
        date_diff(mis_date,created_at,MONTH) as mob,
        arrangement_id
  from joey-bi-prod-project.staging.stg_collection__limit_loan 
  where mis_date= last_day(mis_date)
),
limit_data as (
  select * 
  from joey-bi-prod-project.staging.stg_loan__personal_loan_sys_limit_application 
  where mis_date= last_day(mis_date)
),
repayment_data as (
  select mis_date,
         arrangement_id,
         max(dpd) max_dpd,
         sum(owed_amount) os_balance,
         case 
          when max(dpd) is null then 0
          when max(dpd) >30 then sum(owed_amount)
          end as dpd_os_30,
         case 
          when max(dpd) is null then 0
          when max(dpd) >0 then sum(owed_amount)
          end as dpd_os_1,
         case 
          when max(dpd) is null then 0
          when max(dpd) >=14 then sum(owed_amount)
          end as dpd_os_14
  from joey-bi-prod-project.staging.stg_collection__repayment_schedule 
  where mis_date= last_day(mis_date)
        and repay_status='REPAYING'
  group by mis_date, arrangement_id
),
flagging_fund_ntb as (
  select cast(t24_customer_id as string) as t24_customer_id,
        min(ever_funded) as flagging,
        max(uninstall) as uninstall
  from ( select t24_customer_id,
                ever_funded,
                uninstall
        from joey-bi-ss-risk-fraud-project.credit.funded_info_2025_06
        
        UNION ALL

        select t24_customer_id,
                ever_funded,
                cast(uninstall as int)
        from joey-bi-ss-risk-fraud-project.credit.funded_info_2025_07
  )  group by t24_customer_id

  union ALL
  
  select t24_customer_id,
        max(2) as flagging, --NTB
        max(0) as uninstall
  from joey-bi-prod-project.staging.ntb_prefilter_saku_kredit
  group by t24_customer_id
),
flagging_grp as (
  select t24_customer_id,
      case
        when max(flagging) = 2 then 'NTB'
        when max(flagging) = 1 then '1'
        when max(flagging) = 0 then '0'
        else '' end as flagging,
      max(uninstall) as uninstall
  from flagging_fund_ntb
  group by t24_customer_id
),
combine as (
  select a.*,
          b.t24_customer_id,
          b.deviation_level,
          c.*,
          case
            when d.flagging is null then "0"
            else d.flagging end as flagging,
          d.uninstall
  from disb_data a
  left join limit_data b 
        on a.limit_id=cast(b.limit_id as STRING)
        and a.mis_pt_date = b.mis_date
  left join repayment_data c 
        on a.arrangement_id = c.arrangement_id
        and a.mis_pt_date = c.mis_date
  left join flagging_grp d
        on b.t24_customer_id = cast(d.t24_customer_id as string)
),
comb_band as (
  select *,
        case
          when tenor = 1 then 'tenor 1'
          when tenor <= 3 then 'tenor 2-3'
          when tenor <= 6 then 'tenor 4-6'
          when tenor <= 11 then 'tenor 7-11'
          when tenor = 12 then 'tenor 12'
          else  'else' end as tenor_grp,
        case
          when t24_customer_id in ('101174407','101858872','102072000',
                                    '102084939','102072802','102096422') then 1
          else 0 end as fraud_flag
  from combine
)
select substr(string(mis_pt_date),1,7) as mis_date,
        tenor_grp,
        product_code,
        disbym,
        mob,
        sum(1) as count_loan,
        count(distinct t24_customer_id) as count_client,
        sum(limit_loan_amount) as total_limit,
        sum(os_balance) as total_os,
        sum(dpd_os_30) as dpd_30_os,
        fraud_flag,
        flagging,
        uninstall,
        0, -- placeholder
        sum(dpd_os_1) as dpd_1_os,
        0, -- placeholder
        sum(dpd_os_14) as dpd_14_os
from comb_band
where mob >= 0
group by substr(string(mis_pt_date),1,7),
        tenor_grp,
        product_code,
        disbym,
        mob,
        fraud_flag,
        flagging,
        uninstall

'''
	vintage_apps = client_con.query(sql).to_dataframe()
	return vintage_apps

def Segment_bad(client_con):
	
	##DECLARE date1 STRING DEFAULT '2025-07-31';

	sql = '''
With disb_base as (
  select arrangement_id,
          limit_id,
          limit_loan_amount as disb_amt,
          tenor,
          product_code,
          substr(cast(created_at as STRING),0,10) as disb_dt,
          concat(substr(string(created_at),1,4),'-',substr(string(created_at),6,2)) as disb_mth,
          extract(hour from created_at) as disb_hr
  from joey-bi-prod-project.staging.stg_collection__limit_loan disb
  where STRING(disb.mis_date)= date1
),
limit_base as (
  select 
      t24_customer_id,
      limit_id,
      deviation_level,
      request_location,
      limit_amount
  from joey-bi-prod-project.staging.stg_loan__personal_loan_sys_limit_application
  where STRING(mis_date)=date1
),
funded_detail as (
  select t24_customer_id,
        max(ever_funded) as ever_funded
  from joey-bi-ss-risk-fraud-project.credit.funded_info_2025_05
  group by t24_customer_id
),
work_info as (
  select *
  from joey-bi-prod-project.staging.stg_loan__sake_limit_work_info
  where STRING(mis_date)=date1
),
onboarding_req as (
  select t24_customer_id,
          province,
          REGEXP_EXTRACT(email, r'@(.+)') AS email_domain,
          channel,
          gender,
          customer_range_age,
          phone_number,
          case
            when substr(phone_number,1,5) in ('62812','62821','62822',
                              '62823','62813','62852','62851','62811',
                              '62853') then '1. Telkomsel'
            when substr(phone_number,1,5) in ('62856','62857','62815',
                              '62858','62814','62855','62816') then '2. Indosat'
            when substr(phone_number,1,5) in ('62817','62818','62877',
                              '62819','62878','62859') then '3. XL'
            when substr(phone_number,1,5) in ('62881','62882','62888',
                              '62889','62887') then '4. Smartfren'
            when substr(phone_number,1,5) in ('62838','62831') then '5. Axis'
            when substr(phone_number,1,5) in ('62899','62898','62897',
                              '62896','62895') then '6. Tri'
            else '9. Others' 
            end as No_Provider
          
  from joey-bi-prod-project.staging.stg_onboarding__onboarding_request__hive_merge
  where STRING(mis_date)=date1
),
cred_summary as (
  select application_id,
          credit_bureau_score as clik_bur_score,
          credit_bureau_risk_level as clik_risk_lvl,
          bureau_type as clik_bureau_info,
          income_verified as izi_income_verification,
          home_address_verified as aai_home_verification,
          office_address_verified as aai_office_verification,
          alternative_advance_ai_score as bps_score,
          final_risk_level
  from joey-bi-prod-project.staging.stg_loan__loan_credit_summary
  where STRING(mis_date)=date1
),
repay_data as (
  select a.arrangement_id,
          a.mis_date,
          sum(a.owed_amount) as sum_owed,
          b.limit_id
  from joey-bi-prod-project.staging.stg_collection__repayment_schedule a
  left join joey-bi-prod-project.staging.stg_collection__limit_loan b
        on a.arrangement_id = b.arrangement_id
        and String(b.mis_date) = date1
  where a.mis_date = last_day(a.mis_date)
  group by a.arrangement_id, a.mis_date, b.limit_id
),
repay_data_cust as (
  select limit_id,
        mis_date,
        sum(sum_owed) as os_client
  from repay_data
  group by limit_id, mis_date
),
dpd_data as (
  select substr(string(mis_date),1,7) as mis_mth,
          arrangement_id,
         max(dpd) max_dpd,
         sum(owed_amount) os_balance,
         case 
          when max(dpd) is null then 0
          when max(dpd) >=14 then sum(owed_amount)
          end as dpd_os_14,
         case 
          when max(dpd) is null then 0
          when max(dpd) >30 then sum(owed_amount)
          end as dpd_os_30
  from joey-bi-prod-project.staging.stg_collection__repayment_schedule 
  where mis_date= last_day(mis_date)
        and repay_status='REPAYING'
        -- and current_tenor = 1
  group by mis_date, arrangement_id
),
combine as (
  select a.*,
          b.*,
          d.ever_funded,
          e.province,
          e.email_domain,
          e.channel,
          e.gender,
          e.customer_range_age,
          e.phone_number,
          e.No_Provider,
          f.clik_bur_score,
          f.clik_risk_lvl,
          f.clik_bureau_info,
          f.izi_income_verification,
          f.aai_home_verification,
          f.aai_office_verification,
          f.bps_score,
          f.final_risk_level,
          g.os_client,
          h.declared_income,
          t.dpd_os_14,
          t.dpd_os_30
  from disb_base a
  left join limit_base b on a.limit_id = cast(b.limit_id as STRING)
  left join funded_detail d on b.t24_customer_id = cast(d.t24_customer_id as string) 
  left join onboarding_req e on b.t24_customer_id = cast(e.t24_customer_id as string) 
  left join cred_summary f on a.limit_id = f.application_id 
  left join repay_data_cust g on a.limit_id = g.limit_id
        and last_day(cast(a.disb_dt as date)) = g.mis_date
  left join work_info h on a.limit_id = h.id
  -- left join dpd_data s on a.arrangement_id = s.arrangement_id
                        -- disb: 2025-01, dpd: 2025-02
                    -- and (cast(substr(a.disb_mth,1,4) as int)*12 + cast(substr(a.disb_mth,6,2) as int))
                    --   = (cast(substr(s.mis_mth,1,4) as int)*12 + cast(substr(s.mis_mth,6,2) as int) -1)
  left join dpd_data t on a.arrangement_id = t.arrangement_id
                        -- disb: 2025-01, dpd: 2025-03
                    and (cast(substr(a.disb_mth,1,4) as int)*12 + cast(substr(a.disb_mth,6,2) as int))
                      = (cast(substr(t.mis_mth,1,4) as int)*12 + cast(substr(t.mis_mth,6,2) as int) -2)
),
comb_band as (
  select *,
        case
          when disb_hr between 5 and 10 then '1. Morning: 5-10'
          when disb_hr between 11 and 15 then '2. Noon: 11-15'
          when disb_hr between 16 and 22 then '3. Evening: 16-22'
          when disb_hr between 22 and 24 then '4. Night: 22-4'
          when disb_hr between 0 and 4 then '4. Night: 22-4'
          else '9. Else' end as disb_hr_grp,
        case
          when tenor = 1 then 'tenor 1'
          when tenor <= 3 then 'tenor 2-3'
          when tenor <= 6 then 'tenor 4-6'
          when tenor <= 11 then 'tenor 7-11'
          when tenor = 12 then 'tenor 12'
          else  'else' end as tenor_grp,
        case
          when email_domain in ('gmail.com') then '1. Gmail'
          when email_domain in ('yahoo.com','ymail.com','yahoo.co.id') then '2. Yahoo'
          when email_domain in ('icloud.com') then '3. Apple-icloud'
          when email_domain in ('outlook.com','hotmail.com','windowslive.com') then '4. outlook/hotmail/windows'
          else '9. Others' end as email_domain_grp,
        case
          when clik_bur_score is null then '00. Null'
          when clik_bur_score  =   0 then '00. 0'
          when clik_bur_score <= 150 then '01. 1 - 150'
          when clik_bur_score <= 345 then '02. 151 - 345'
          when clik_bur_score <= 405 then '03. 346 - 405'
          when clik_bur_score <= 440 then '04. 406 - 440'
          when clik_bur_score <= 449 then '05. 441 - 449'
          when clik_bur_score <= 468 then '06. 450 - 468'
          when clik_bur_score <= 488 then '07. 469 - 488'
          when clik_bur_score <= 501 then '08. 489 - 501'
          when clik_bur_score <= 525 then '09. 502 - 525'
          when clik_bur_score <= 659 then '10. 526 - 659'
          when clik_bur_score  > 659 then '11. above 659'
          else '99. else' end as clik_bur_score_grp,
        case
          when clik_bur_score is null then '00. Null'
          when clik_bur_score  =   0 then '00. 0'
          when clik_bur_score <= 149 then '00. 1 - 149'
          when clik_bur_score <= 267 then '01. 150 - 267'
          when clik_bur_score <= 379 then '02. 268 - 379'
          when clik_bur_score <= 519 then '03. 380 - 519'
          when clik_bur_score <= 545 then '04. 520 - 545'
          when clik_bur_score <= 561 then '05. 546 - 561'
          when clik_bur_score  > 561 then '06. above 561' --up to 659 based on docs....
          else '99. else' end as clik_bur_score_grp_2,
        case
          when bps_score is null then '00. null'
          when bps_score  =    0 then '00. 0'
          when bps_score <=  370 then '01. 1 - 370'
          when bps_score <=  500 then '02. 371 - 500'
          when bps_score <=  518 then '03. 501 - 518'
          when bps_score <=  530 then '04. 519 - 530'
          when bps_score <=  539 then '05. 531 - 539'
          when bps_score <=  546 then '06. 540 - 546'
          when bps_score <=  553 then '07. 547 - 553'
          when bps_score <=  559 then '08. 554 - 559'
          when bps_score <=  564 then '09. 560 - 564'
          when bps_score <=  570 then '10. 565 - 570'
          when bps_score <=  576 then '11. 571 - 576'
          when bps_score <=  581 then '12. 577 - 581'
          when bps_score <=  587 then '13. 582 - 587'
          when bps_score <=  593 then '14. 588 - 593'
          when bps_score <=  600 then '15. 594 - 600'
          when bps_score <=  607 then '16. 601 - 607'
          when bps_score <=  616 then '17. 608 - 616'
          when bps_score <=  627 then '18. 617 - 627'
          when bps_score <=  641 then '19. 628 - 641'
          when bps_score <=  663 then '20. 642 - 663'
          when bps_score <=  806 then '21. 664 - 806'
          when bps_score >   806 then '22. above 806'
          else '99. Else' end as bps_score_grp,
        case
          when bps_score is null then '00. null'          
          when bps_score  =    0 then '00. 0'
          when bps_score <=  499 then '01. 370 - 499'
          when bps_score <=  530 then '02. 500 - 530'
          when bps_score <=  559 then '03. 531 - 559'
          when bps_score <=  593 then '04. 560 - 593'
          when bps_score <=  627 then '05. 594 - 627'
          when bps_score <=  806 then '06. 628 - 806'
          when bps_score >   806 then '07. above 806'
          else '99. Else' end as bps_score_grp_2,
        case
          when province in ('BANTEN','DKI JAKARTA','JAWA BARAT',
                          'JAWA TENGAH','JAWA TIMUR') then province
          when province in ('ACEH','BENGKULU', 'JAMBI', 'KEP. BANGKA BELITUNG', 
                          'LAMPUNG','RIAU') then 'SUMATERA'
          when province like '%SUMATERA%' then 'SUMATERA'
          when province in ('BALI') then 'BALI'
          when province in ('DAERAH ISTIMEWA YOGYAKARTA') then 'JAWA TENGAH'
          when province like '%KALIMANTAN%' then 'KALIMANTAN'
          when province in ('MALUKU','NUSA TENGGARA BARAT','PAPUA') then 'Indonesia Bagian Timur'
          when province like '%SULAWESI%' then 'Indonesia Bagian Timur'
          else '99. Others' end as prov_grp,
        case
          when limit_amount =         0 then '1. 0'
          when limit_amount <=   500000 then '2. 500k or below'
          when limit_amount <=  5000000 then '3. 500k - 5Mio'
          when limit_amount <= 10000000 then '4. 5Mio - 10Mio'
          when limit_amount <= 15000000 then '5. 10Mio - 15Mio'
          when limit_amount <= 20000000 then '6. 15Mio - 20Mio'
          when limit_amount <= 28000000 then '7. 20Mio - 28Mio'
          when limit_amount >  28000000 then '8. 28Mio or above'
          else '9. Else' end as limit_band_grp,
        case
          when os_client = 0 then '0. 0%'
          when (os_client/limit_amount) = 0.00 then '1. 0%'
          when (os_client/limit_amount) <= 0.20 then '2. 1 - 20%'
          when (os_client/limit_amount) <= 0.40 then '3. 21 - 40%'
          when (os_client/limit_amount) <= 0.60 then '4. 41 - 60%'
          when (os_client/limit_amount) <= 0.80 then '5. 61 - 80%'
          when (os_client/limit_amount) <= 1.00 then '6. 81 - 100%'
          else '9. Else' end as util_grp,
        case
          when declared_income is null then '00. null'
          when declared_income = 0         then '00. 0'
          when declared_income <=  1000000 then '01. IDR 1 - 1 Mio'
          when declared_income <=  2000000 then '02. 1 Mio - 2 Mio'
          when declared_income <=  5000000 then '03. 2 Mio - 5 Mio'
          when declared_income <= 10000000 then '04. 5 Mio - 10 Mio'
          when declared_income <= 15000000 then '05. 10 Mio - 15 Mio'
          when declared_income <= 20000000 then '06. 15 Mio - 20 Mio'
          when declared_income <= 25000000 then '07. 20 Mio - 25 Mio'
          when declared_income <= 30000000 then '08. 25 Mio - 30 Mio'
          when declared_income <= 35000000 then '09. 30 Mio - 35 Mio'
          when declared_income <= 40000000 then '10. 35 Mio - 40 Mio'
          when declared_income <= 45000000 then '11. 40 Mio - 45 Mio'
          when declared_income <= 50000000 then '12. 45 Mio - 50 Mio'
          when declared_income > 50000000 then '13. > 50 Mio'
          else '99. Else' end as declared_income_grp
  from combine
)
select 
      disb_mth,
      tenor_grp,
      product_code,
      ever_funded,
      deviation_level,
      disb_hr_grp,
      prov_grp,
      email_domain_grp,
      channel,
      gender,
      customer_range_age,
      No_Provider,
      clik_bur_score_grp_2,
      clik_risk_lvl,
      clik_bureau_info,
      izi_income_verification,
      aai_home_verification,
      aai_office_verification,
      bps_score_grp_2,
      final_risk_level,
      sum(disb_amt) as total_disb,
      sum(1) as count_loan,
      sum(dpd_os_14) as dpd_14_mob_1,
      sum(dpd_os_30) as dpd_30_mob_2,
      limit_band_grp,
      util_grp,
      sum(case when dpd_os_14 > 0 then 1 else 0 end) as count_dpd_14_mob_1,
      sum(case when dpd_os_30 > 0 then 1 else 0 end) as count_dpd_30_mob_2,
      declared_income_grp
from comb_band
where t24_customer_id not in ('101174407','101858872','102072000',
                                    '102084939','102072802','102096422') -- fraud
group by 
      disb_mth,
      tenor_grp,
      product_code,
      ever_funded,
      deviation_level,
      disb_hr_grp,
      prov_grp,
      email_domain_grp,
      channel,
      gender,
      customer_range_age,
      No_Provider,
      clik_bur_score_grp_2,
      clik_risk_lvl,
      clik_bureau_info,
      izi_income_verification,
      aai_home_verification,
      aai_office_verification,
      bps_score_grp_2,
      final_risk_level,
      limit_band_grp,
      util_grp,
      declared_income_grp
'''
	Segment_bad = client_con.query(sql).to_dataframe()
	return Segment_bad


def Porto_util(client_con):
	
	##DECLARE date1 STRING DEFAULT '2025-07-31';

	sql = '''
with limit_base as (
  select t24_customer_id as CIF,
          limit_id,
          limit_amount,
          mis_date
          -- substr(string(last_day(created_at)),1,10) as pt_date
  from joey-bi-prod-project.staging.stg_loan__personal_loan_sys_limit_application 
  where mis_date = last_day(mis_date)
      and status in ('agreementSigned')

),
disb_base as (
  select mis_date,
        arrangement_id,
        limit_id,
        customer_info_id,
        limit_loan_amount
  from joey-bi-prod-project.staging.stg_collection__limit_loan 
  where mis_date = last_day(mis_date) 
),
repay_dpd_base as (
  select mis_date,
          arrangement_id,
         sum(owed_amount) as os_bal
  from joey-bi-prod-project.staging.stg_collection__repayment_schedule 
  where mis_date = last_day(mis_date)
  group by mis_date, arrangement_id 
),
flagging_fund_ntb as (
  select cast(t24_customer_id as string) as t24_customer_id,
        min(ever_funded) as flagging
from ( select t24_customer_id,
                ever_funded
        from joey-bi-ss-risk-fraud-project.credit.funded_info_2025_06
        
        UNION ALL

        select t24_customer_id,
                ever_funded
        from joey-bi-ss-risk-fraud-project.credit.funded_info_2025_07
  ) group by t24_customer_id

  union ALL
  
  select t24_customer_id,
        max(2) as flagging, --NTB
  from joey-bi-prod-project.staging.ntb_prefilter_saku_kredit
  group by t24_customer_id
),
flagging_grp as (
  select t24_customer_id,
      case
        when max(flagging) = 2 then 'NTB'
        when max(flagging) = 1 then '1'
        when max(flagging) = 0 then '0'
        else '' end as flagging,
  from flagging_fund_ntb
  group by t24_customer_id
),
combine as(
  select a.*,
        b.arrangement_id,
        c.os_bal,
        case when d.flagging is null then "0"
      		else d.flagging end as flagging
  from limit_base a
  left join disb_base b 
          on cast(a.limit_id as STRING) = b.limit_id
          and a.mis_date = b.mis_date
  left join repay_dpd_base c 
          on b.arrangement_id = c.arrangement_id
          and b.mis_date = c.mis_date
  left join flagging_grp d on a.CIF = cast(d.t24_customer_id as string)

),
grp_client as (
  select mis_date,
        CIF,
        flagging,
        max(limit_amount) as limit_amt,
        sum(case 
            when os_bal is null then 0
            else os_bal end) as os_bal
  from combine
  group by mis_date,
            CIF,
            flagging
),
comb_band as (
  select *,
        case
          when limit_amt = 0 then '1. 0%'
          when (os_bal/limit_amt) = 0.00 then '1. 0%'
          when (os_bal/limit_amt) <= 0.20 then '2. 1 - 20%'
          when (os_bal/limit_amt) <= 0.40 then '3. 21 - 40%'
          when (os_bal/limit_amt) <= 0.60 then '4. 41 - 60%'
          when (os_bal/limit_amt) <= 0.80 then '5. 61 - 80%'
          when (os_bal/limit_amt) <= 1.00 then '6. 81 - 100%'
          when (os_bal/limit_amt) >  1.00 then '7. above 100%'
          else '9. Else' end as util_grp
  from grp_client
  where CIF not in ('101174407','101858872','102072000',
                      '102084939','102072802','102096422') -- fraud
)
select substr(cast(mis_date as string),1,7) as mis_date,
        util_grp,
       count(distinct CIF) as total_cust,
       flagging,
       sum(limit_amt) as total_limit
from comb_band
-- where util_grp not in ('9. Else')
group by mis_date,
        util_grp,
        flagging

'''
	Porto_util_app = client_con.query(sql).to_dataframe()
	return Porto_util_app

def ThruTheDoor(client_con):
  sql = '''
with base_limit as (
  select mis_date,
        t24_customer_id,
        limit_id,
        substr(string(created_at),1,10) as create_date,
        substr(string(created_at),1,7) as create_mth,
        status,
        case
          when limit_amount < 0 then 0
          else limit_amount end as limit_amount
  from joey-bi-prod-project.staging.stg_loan__personal_loan_sys_limit_application
  where mis_date = current_date-1
),
flagging_fund_ntb as (
  select cast(t24_customer_id as string) as t24_customer_id,
        min(ever_funded) as flagging
  from ( select t24_customer_id,
                ever_funded
        from joey-bi-ss-risk-fraud-project.credit.funded_info_2025_06
        
        UNION ALL

        select t24_customer_id,
                ever_funded
        from joey-bi-ss-risk-fraud-project.credit.funded_info_2025_08
  )
  group by t24_customer_id

  union ALL
  
  select t24_customer_id,
        max(2) as flagging --NTB
  from joey-bi-prod-project.staging.ntb_prefilter_saku_kredit
  group by t24_customer_id
),
flagging_grp as (
  select t24_customer_id,
      case
        when max(flagging) = 2 then 'NTB'
        when max(flagging) = 1 then '1'
        when max(flagging) = 0 then '0'
        else '' end as flagging
  from flagging_fund_ntb
  group by t24_customer_id
),
onboarding_req as (
  select mis_date,
          t24_customer_id,
          province,
          channel,
          gender,
          customer_range_age,
          case
            when substr(phone_number,1,5) in ('62812','62821','62822',
                              '62823','62813','62852','62851','62811',
                              '62853') then '1. Telkomsel'
            when substr(phone_number,1,5) in ('62856','62857','62815',
                              '62858','62814','62855','62816') then '2. Indosat'
            when substr(phone_number,1,5) in ('62817','62818','62877',
                              '62819','62878','62859') then '3. XL'
            when substr(phone_number,1,5) in ('62881','62882','62888',
                              '62889','62887') then '4. Smartfren'
            when substr(phone_number,1,5) in ('62838','62831') then '5. Axis'
            when substr(phone_number,1,5) in ('62899','62898','62897',
                              '62896','62895') then '6. Tri'
            else '9. Others' 
            end as No_Provider
          
  from joey-bi-prod-project.staging.stg_onboarding__onboarding_request__hive_merge
  where mis_date = current_date-1
),
cred_summary as (
  select mis_date,
          application_id,
          credit_bureau_score as clik_bur_score,
          bureau_type as clik_bureau_info,
          alternative_advance_ai_score as bps_score,
          final_risk_level
  from joey-bi-prod-project.staging.stg_loan__loan_credit_summary
  where mis_date = current_date-1
),
combine as (
  select a.*,
        case 
          when b.flagging is null then "0"
		      else b.flagging end as flagging,
        e.province,
          e.channel,
          e.gender,
          e.customer_range_age,
          e.No_Provider,
          f.clik_bur_score,
          f.clik_bureau_info,
          f.bps_score,
          f.final_risk_level
  from base_limit a
  left join flagging_grp b on a.t24_customer_id = cast(b.t24_customer_id as string)
  left join onboarding_req e 
          on a.t24_customer_id = cast(e.t24_customer_id as string) 
  left join cred_summary f on a.limit_id = f.application_id 
),
comb_band as (
  select *,
        case
          when status in ('agreementSigned','approved') then '1. Approved/Signed'
          when status in ('rejected') then '2. Rejected'
          when status in ('incomplete') then '3. Incomplete'
          else '9. Others' end as status_grp,
        case
          when limit_amount =         0 then '01. 0'
          when limit_amount =    500000 then '02. Base Limit: 500K'
          when limit_amount =   1000000 then '03. Base Limit: 1 Mio'
          when limit_amount =   2000000 then '04. Base Limit: 2 Mio'
          when limit_amount <=  5000000 then '05. <= 5 Mio'
          when limit_amount <= 10000000 then '06. (5Mio - 10Mio]'
          when limit_amount <= 15000000 then '07. (10Mio - 15Mio]'
          when limit_amount <= 20000000 then '08. (15Mio - 20Mio]'
          when limit_amount <= 28000000 then '09. (20Mio - 28Mio]'
          when limit_amount >  28000000 then '10. > 28 Mio'
          else '9. Else' end as limit_band_grp,
        case
          when clik_bur_score is null then '00. Null'
          when clik_bur_score  =   0 then '00. 0'
          when clik_bur_score <= 149 then '00. 1 - 149'
          when clik_bur_score <= 267 then '01. 150 - 267'
          when clik_bur_score <= 379 then '02. 268 - 379'
          when clik_bur_score <= 519 then '03. 380 - 519'
          when clik_bur_score <= 545 then '04. 520 - 545'
          when clik_bur_score <= 561 then '05. 546 - 561'
          when clik_bur_score  > 561 then '06. above 561' --up to 659 based on docs....
          else '99. else' end as clik_bur_score_grp_2,
        case
          when bps_score is null then '00. null'
          when bps_score  =    0 then '00. 0'
          when bps_score <=  499 then '01. 370 - 499'
          when bps_score <=  530 then '02. 500 - 530'
          when bps_score <=  559 then '03. 531 - 559'
          when bps_score <=  593 then '04. 560 - 593'
          when bps_score <=  627 then '05. 594 - 627'
          when bps_score <=  806 then '06. 628 - 806'
          when bps_score >   806 then '07. above 806'
          else '99. Else' end as bps_score_grp_2,
        case
          when province in ('BANTEN','DKI JAKARTA','JAWA BARAT',
                          'JAWA TENGAH','JAWA TIMUR') then province
          when province in ('ACEH','BENGKULU', 'JAMBI', 'KEP. BANGKA BELITUNG', 
                          'LAMPUNG','RIAU', 'KEPULAUAN RIAU') then 'SUMATERA'
          when province like '%SUMATERA%' then 'SUMATERA'
          when province in ('BALI') then 'BALI'
          when province in ('DAERAH ISTIMEWA YOGYAKARTA') then 'JAWA TENGAH'
          when province like '%KALIMANTAN%' then 'KALIMANTAN'
          when province in ('MALUKU','MALUKU UTARA','NUSA TENGGARA BARAT',
                            'NUSA TENGGARA TIMUR','PAPUA', 
                            'PAPUA TENGAH', 'PAPUA BARAT',
                            'PAPUA BARAT DAYA','GORONTALO',
                            'PAPUA SELATAN', 'PAPUA PEGUNUNGAN') then 'Indonesia Bagian Timur'
          when province like '%SULAWESI%' then 'Indonesia Bagian Timur'
          else '99. Others' end as prov_grp,
  from combine
)
, result_incoming as (
select substr(string(mis_date),1,7) as mis_date,
        create_mth,
        status_grp,
        limit_band_grp,
        flagging,
        channel,
        gender,
        customer_range_age,
        No_Provider,
        clik_bureau_info,
        final_risk_level,
        clik_bur_score_grp_2,
        bps_score_grp_2,
        prov_grp,
        sum(1) as count_apps
from comb_band
where status_grp in ('1. Approved/Signed','2. Rejected')
group by substr(string(mis_date),1,7),
        create_mth,
        status_grp,
        limit_band_grp,
        flagging,
        channel,
        gender,
        customer_range_age,
        No_Provider,
        clik_bureau_info,
        final_risk_level,
        clik_bur_score_grp_2,
        bps_score_grp_2,
        prov_grp
)
select *
from result_incoming

'''
	ThruTheDoor = client_con.query(sql).to_dataframe()
	return ThruTheDoor



def limit_v2_3_DBR_perf(client_con):
	
	##DECLARE date1 STRING DEFAULT '2025-07-31';

	sql = '''
With limit_base as (
  select 
      t24_customer_id,
      cast(mis_date as string) as mis_date,
      limit_id,
      cast(limit_amount as numeric) as limit_amt,
      date(datetime(created_at)) as create_dt,
      customer_info_id
  from joey-bi-prod-project.staging.stg_loan__personal_loan_sys_limit_application
  where STRING(mis_date)= date1
    and status = "agreementSigned"
),
disb_base as (
  select arrangement_id,
          limit_id,
          limit_loan_amount as disb_amt,
          tenor,
          product_code,
          collectability,
          created_at as disb_at,
          substr(cast(created_at as STRING),0,10) as disb_dt,
          concat(substr(string(created_at),1,4),'-',substr(string(created_at),6,2)) as disb_mth,
          extract(hour from created_at) as disb_hr
  from joey-bi-prod-project.staging.stg_collection__limit_loan disb
  where STRING(mis_date)= date1
),
funded_detail as (
  select t24_customer_id,
        max(ever_funded) as ever_funded
  from joey-bi-ss-risk-fraud-project.credit.funded_info_2025_04
  group by t24_customer_id
),
repay_data as (
  select a.arrangement_id,
          cast(a.mis_date as string) as mis_date,
          sum(a.owed_amount) as sum_owed,
          max(a.dpd) as DPD_acct,
          b.limit_id
  from joey-bi-prod-project.staging.stg_collection__repayment_schedule a
  left join joey-bi-prod-project.staging.stg_collection__limit_loan b
        on a.arrangement_id = b.arrangement_id
        and String(b.mis_date) = date1
  where a.mis_date = last_day(a.mis_date)
  group by a.arrangement_id, a.mis_date, b.limit_id
),
bur_detail as (
    SELECT customer_id,
          count_num,
          provider_code_desc,
          contract_phase_desc,
          contract_type,
          contract_type_desc,
          start_date,
          due_date,
          past_due_status_code as latest_kol,
          past_due_status_desc,
          debit_bal as latest_os,
          dpd as latest_dpd,
          max_dpd,
          worst_status,
          worst_status_desc,
          interest_rate,
          credit_limit,
          collaterals_num,
          tot_collateral_value,
          (cast(substr(due_date,1,4) as int)*12 + cast(substr(due_date,6,2) as int))
          - (cast(substr(start_date,1,4) as int)*12 + cast(substr(start_date,6,2) as int)
          ) as tenor,
          init_credit_lim,
          case
            when interest_rate is null
                or interest_rate = 0 then 0.0001
            else interest_rate end as int_cast

  FROM joey-bi-prod-project.staging.stg_credit_report__clik_credit_granted__hive_merge
  where string(mis_date) = date1
),
bur_detail_2 as (
  select *,
        case
          when tot_collateral_value > 0
              and provider_code_desc in ('PT Atome Finance Indonesia') then 9
          when tot_collateral_value > 0
              and provider_code_desc in ('PT Bank Tabungan Negara (Persero) Tbk') then 7
          when tot_collateral_value > 0
              and provider_code_desc in ('PT Bank Danamon Indonesia Tbk','PT Bank Central Asia Tbk',
              'PT Bank Mandiri (Persero) Tbk','PT Bank Syariah Indonesia','PT Bank Mega Tbk',
              'PT Bank Mega Syariah','PT Bank Rakyat Indonesia (Persero) Tbk',
              'PT Bank Negara Indonesia (Persero) Tbk') 
              and tenor > 60 then 7
          when tot_collateral_value > 0
              and provider_code_desc in ('PT Astra Sedaya Finance') then 6
          when tot_collateral_value > 0
              and provider_code_desc in ('PT Bank Danamon Indonesia Tbk','PT Bank Central Asia Tbk',
              'PT Bank Mandiri (Persero) Tbk','PT Bank Syariah Indonesia','PT Bank Mega Tbk',
              'PT Bank Mega Syariah','PT Bank Rakyat Indonesia (Persero) Tbk',
              'PT Bank Negara Indonesia (Persero) Tbk') 
              and tenor <= 60 then 6
          when tot_collateral_value > 0
              and provider_code_desc in ('PT BCA Finance','PT KB Finansia Multi Finance',
              'PT Mandiri Utama Finance','PT Wahana Ottomitra Multiartha Tbk','PT Mega Central Finance',
              'PT Bussan Auto Finance (UUS)','PT Mandiri Tunas Finance','PT Mega Auto Finance',
              'PT Mega Finance','PT Bussan Auto Finance')
              and tenor > 24 then 6
          when tot_collateral_value > 0
              and provider_code_desc in ('PT Adira Dinamika Multi Finance Tbk',
              'PT Federal International Finance','PT BFI Finance Indonesia',
              'PT Summit Oto Finance','PT Adira Dinamika Multi Finance (UUS)',
              'PT Federal International Finance (UUS)') then 5
          when tot_collateral_value > 0
              and provider_code_desc in ('PT BCA Finance','PT KB Finansia Multi Finance',
              'PT Mandiri Utama Finance','PT Wahana Ottomitra Multiartha Tbk','PT Mega Central Finance',
              'PT Bussan Auto Finance (UUS)','PT Mandiri Tunas Finance','PT Mega Auto Finance',
              'PT Mega Finance','PT Bussan Auto Finance')
              and tenor <= 24 then 5
          when tot_collateral_value > 0
              and provider_code_desc like '%Rakyat%' then 2
          when tot_collateral_value > 0
              and provider_code_desc like '%BPR%' then 2
          when tot_collateral_value > 0
              and provider_code_desc like '%BPD%' then 2
          when tot_collateral_value > 0 then 1
          else 0 end as coll_type,
      case
        when tenor <= 0 then 1
        else tenor end as tenor_cast

  from bur_detail
),
bur_detail_client_list as(
  select customer_id,
        max(count_num) as count_num
  from bur_detail
  group by customer_id 
),
bur_detail_client_info as (
  select customer_id,
        count_num,
        max(case
          when contract_type_desc = 'Credit Card' then credit_limit
          else 0 end) as max_CC_limit,
        sum(case
          when contract_type_desc = 'Credit Card' then credit_limit
          else 0 end) as sum_CC_limit,
        sum(case
          when contract_type_desc = 'Credit Card' then latest_os
          else 0 end) as sum_CC_OS,
        sum(case
          when contract_type_desc = 'Credit Card' then 1
          else 0 end) as count_CC_issuer,

        max(tot_collateral_value) as collateral_flag,
        max(coll_type) as collateral_type_code,
        max(case
          when tot_collateral_value >0 then latest_os
          else 0 end) as max_collateral_loan_OS,
        max(case
          when tot_collateral_value >0 then 
                  init_credit_lim / (power(1+int_cast/100/12,tenor_cast)-1) * 
                    (int_cast/100/12*power(1+int_cast/100/12,tenor_cast))
          else 0 end) as max_collateral_loan_instalment,
        max(case
          when tot_collateral_value >0 then 
                  tenor
          else 0 end) as max_collateral_loan_tenor,
        
        sum(case
          when contract_type_desc in ('Others','Loans provided')
              and tot_collateral_value = 0 then latest_os
            else 0 end) as sum_kta_loan_OS,
        max(case
          when contract_type_desc in ('Others','Loans provided')
              and tot_collateral_value = 0 then interest_rate
            else 0 end) as max_kta_int_rate,
        max(case
          when contract_type_desc in ('Others','Loans provided')
              and tot_collateral_value = 0 then 
                  init_credit_lim / (power(1+int_cast/100/12,tenor_cast)-1) * 
                    (int_cast/100/12*power(1+int_cast/100/12,tenor_cast))
          else 0 end) as max_kta_loan_instalment,
        
        sum(case
          when contract_type in ('10','20','P01','P10')
              and tot_collateral_value = 0 then latest_os
              else 0 end) as sum_fintech_loan_OS,
        max(case
          when contract_type in ('10','20','P01','P10')
              and tot_collateral_value = 0 then interest_rate
              else 0 end) as max_fintech_int_rate,
        max(case
          when contract_type in ('10','20','P01','P10')
              and tot_collateral_value = 0 then 
                  init_credit_lim / (power(1+int_cast/100/12,tenor_cast)-1) * 
                    (int_cast/100/12*power(1+int_cast/100/12,tenor_cast))
          else 0 end) as max_fintech_instalment,
        max(case
          when (contract_type in ('10','20','P01','P10')
                or contract_type_desc in ('Others','Loans provided'))
              and tot_collateral_value = 0 then 
                  init_credit_lim / (power(1+int_cast/100/12,tenor_cast)-1) * 
                    (int_cast/100/12*power(1+int_cast/100/12,tenor_cast))
          else 0 end) as max_oth_loan_instalment,
        sum(case
          when (contract_type in ('10','20','P01','P10')
                or contract_type_desc in ('Others','Loans provided'))
              and tot_collateral_value = 0 then 
                  init_credit_lim / (power(1+int_cast/100/12,tenor_cast)-1) * 
                    (int_cast/100/12*power(1+int_cast/100/12,tenor_cast))
          else 0 end) as sum_oth_loan_instalment,
        
        max(case
          when tot_collateral_value = 0 then
                init_credit_lim
                else 0 end) as max_non_collateral_loan_lim

  from bur_detail_2
  where contract_phase_desc = 'Active'
  group by customer_id, count_num
),
cred_summary as (
  select application_id,
          bureau_type as clik_bureau_info,
          income_verified as izi_income_verification,
          final_income,
          monthly_debt_repayment,
          final_dbr,
          internal_exposure,
          total_exposure,
          internal_exposure_cap,
          total_exposure_cap,
          exposure_dbr,
          final_risk_level,
          credit_bureau_score,
          alternative_advance_ai_score
  from joey-bi-prod-project.staging.stg_loan__loan_credit_summary
  where STRING(mis_date)=date1
),
izi_salary as (
  select loan_id,
        salary,
        income,
        RANK() OVER (PARTITION BY loan_id ORDER BY pull_datetime DESC)
          AS data_rank
  from staging.stg_loan__sake_izi_salary
  where STRING(mis_date)=date1
),
izi_salary_list as (
  select loan_id,
          max(data_rank) as data_rank
  from izi_salary
  group by loan_id
),
work_info as (
  select id,
        declared_income
  from joey-bi-prod-project.staging.stg_loan__sake_limit_work_info
  where STRING(mis_date)=date1
),
combine as (
  select a.*,
          b.disb_amt,
          b.disb_mth,
          b.arrangement_id,
          d.ever_funded,
          g.count_num,
          h.clik_bureau_info,
          h.izi_income_verification,
          h.final_income,
          h.monthly_debt_repayment,
          h.final_dbr,
          h.internal_exposure,
          h.total_exposure,
          h.internal_exposure_cap,
          h.total_exposure_cap,
          h.exposure_dbr,
          h.final_risk_level,
          h.credit_bureau_score,
          h.alternative_advance_ai_score,
          M.DPD_acct as dpd_mob_2,
          R.data_rank,
          S.salary as salary_izi_score,
          S.income as izi_income_range,
          T.declared_income
  from limit_base a
  left join disb_base b on a.limit_id = cast(b.limit_id as STRING)
  left join funded_detail d on a.t24_customer_id = cast(d.t24_customer_id as string) 
  left join bur_detail_client_list g on a.customer_info_id = g.customer_id 
  left join cred_summary h on a.limit_id = h.application_id 
  left join repay_data M on b.arrangement_id = M.arrangement_id
      and cast(substr(b.disb_mth,1,4) as int)*12 + cast(substr(b.disb_mth,6,2) as int)
          = cast(substr(M.mis_date,1,4) as int)*12 + cast(substr(M.mis_date,6,2) as int) -2
  left join izi_salary_list R on a.limit_id = R.loan_id
  left join izi_salary S on R.loan_id = S.loan_id
                        and R.data_rank = S.data_rank
  left join work_info T on a.limit_id = T.id

),
combine_2 as (
  select a.*,
        case
          when a.izi_income_range = '0-3999999' then 1000000
          when a.izi_income_range = '4000000-7999999' then 4000000
          when a.izi_income_range = '8000000-9999999' then 8000000
          when a.izi_income_range = '10000000-999999999999' then 10000000
          else 0 end as min_izi_income_range,

        b.max_CC_limit,
        b.sum_CC_limit,
        b.sum_CC_OS,
        b.count_CC_issuer,

        b.collateral_flag,
        b.collateral_type_code,
        b.max_collateral_loan_OS,
        b.max_collateral_loan_instalment,
        b.max_collateral_loan_tenor,
        
        b.sum_kta_loan_OS,
        b.max_kta_int_rate,
        b.max_kta_loan_instalment,
        
        b.sum_fintech_loan_OS,
        b.max_fintech_int_rate,
        b.max_fintech_instalment,

        b.max_oth_loan_instalment,
        b.sum_oth_loan_instalment,
        b.max_non_collateral_loan_lim
  from combine a
  left join bur_detail_client_info b on a.customer_info_id = b.customer_id
          and a.count_num = b.count_num 
),
comb_band as (
  select *,
        case
          when max_CC_limit > limit_amt then '01. Have CC with higher limit'
          when sum_kta_loan_OS > limit_amt then '02. Have KTA with higher limit'
          when sum_fintech_loan_OS > limit_amt then '03. Have fintech with higher limit'
          when max_kta_int_rate <= 1.99 then '04. Have KTA with lower interest'
          when max_fintech_int_rate <= 1.99 then '05. Have fintech with lower interest'
          else '99. Other reason' end as Reason_not_used,
        case
          when max_CC_limit > 100000000 then '01. > 100 mio'
          when max_CC_limit >  70000000 then '02. 70 - 100 mio'
          when max_CC_limit >  50000000 then '03. 50 - 70 mio'
          when max_CC_limit >  30000000 then '04. 30 - 50 mio'
          when max_CC_limit >  20000000 then '05. 20 - 30 mio'
          when max_CC_limit >  10000000 then '06. 10 - 20 mio'
          when max_CC_limit >   5000000 then '07.  5 - 10 mio'
          when max_CC_limit >         0 then '08.  1 -  5 mio'
          else '09. Else' end as CC_limit_grp,
        case
          when sum_kta_loan_OS > 100000000 then '01. > 100 mio'
          when sum_kta_loan_OS >  70000000 then '02. 70 - 100 mio'
          when sum_kta_loan_OS >  50000000 then '03. 50 - 70 mio'
          when sum_kta_loan_OS >  30000000 then '04. 30 - 50 mio'
          when sum_kta_loan_OS >  20000000 then '05. 20 - 30 mio'
          when sum_kta_loan_OS >  10000000 then '06. 10 - 20 mio'
          when sum_kta_loan_OS >   5000000 then '07.  5 - 10 mio'
          when sum_kta_loan_OS >         0 then '08.  1 -  5 mio'
          else '09. Else' end as KTA_loan_grp,
        case
          when sum_fintech_loan_OS > 100000000 then '01. > 100 mio'
          when sum_fintech_loan_OS >  70000000 then '02. 70 - 100 mio'
          when sum_fintech_loan_OS >  50000000 then '03. 50 - 70 mio'
          when sum_fintech_loan_OS >  30000000 then '04. 30 - 50 mio'
          when sum_fintech_loan_OS >  20000000 then '05. 20 - 30 mio'
          when sum_fintech_loan_OS >  10000000 then '06. 10 - 20 mio'
          when sum_fintech_loan_OS >   5000000 then '07.  5 - 10 mio'
          when sum_fintech_loan_OS >         0 then '08.  1 -  5 mio'
          else '09. Else' end as fintech_loan_grp,
        case
          when max_kta_int_rate > 1.99 then '01. > 1.99%'
          when max_kta_int_rate > 1.70 then '02. 1.70 - 1.99%'
          when max_kta_int_rate > 1.50 then '03. 1.50 - 1.70%'
          when max_kta_int_rate > 1.30 then '04. 1.30 - 1.50%'
          when max_kta_int_rate > 1.10 then '05. 1.10 - 1.30%'
          when max_kta_int_rate > 0.90 then '06. 0.90 - 1.10%'
          when max_kta_int_rate > 0.70 then '07. 0.70 - 0.90%'
          when max_kta_int_rate > 0.50 then '08. 0.50 - 0.70%'
          when max_kta_int_rate > 0.25 then '09. 0.25 - 0.50%'
          when max_kta_int_rate > 0 then '10. 0.01 - 0.25%'
          when max_kta_int_rate = 0 then '11. 0%'
          else '12. Else' end as kta_int_grp,
        case
          when max_fintech_int_rate > 1.99 then '01. > 1.99%'
          when max_fintech_int_rate > 1.70 then '02. 1.70 - 1.99%'
          when max_fintech_int_rate > 1.50 then '03. 1.50 - 1.70%'
          when max_fintech_int_rate > 1.30 then '04. 1.30 - 1.50%'
          when max_fintech_int_rate > 1.10 then '05. 1.10 - 1.30%'
          when max_fintech_int_rate > 0.90 then '06. 0.90 - 1.10%'
          when max_fintech_int_rate > 0.70 then '07. 0.70 - 0.90%'
          when max_fintech_int_rate > 0.50 then '08. 0.50 - 0.70%'
          when max_fintech_int_rate > 0.25 then '09. 0.25 - 0.50%'
          when max_fintech_int_rate > 0 then '10. 0.01 - 0.25%'
          when max_fintech_int_rate = 0 then '11. 0%'
          else '12. Else' end as fintech_int_grp,
        case 
          when dpd_mob_2 >= 14 then disb_amt
          else 0 end as FPD14,
        case 
          when dpd_mob_2 >= 30 then disb_amt
          else 0 end as SPD30,
        case
          when final_dbr = 0 then '01. DBR 0%'
          when final_dbr <= 10 then '02. DBR 1 - 10%'
          when final_dbr <= 20 then '03. DBR 11 - 20%'
          when final_dbr <= 30 then '04. DBR 21 - 30%'
          when final_dbr <= 40 then '05. DBR 31 - 40%'
          when final_dbr <= 50 then '06. DBR 41 - 50%'
          when final_dbr <= 60 then '07. DBR 51 - 60%'
          when final_dbr <= 80 then '08. DBR 61 - 80%'
          when final_dbr <= 100 then '09. DBR 81 - 100%'
          when final_dbr > 100 then '10. DBR > 100%'
          else '99. Else' end as DBR_perc,
        case
          when final_income is null or 
                final_income = 0 then '00. No income data'
          when total_exposure is null or 
                total_exposure = 0 then '00. 0%'
          when total_exposure/(total_exposure_cap*final_income) = 0 then '00. 0%'
          when total_exposure/(total_exposure_cap*final_income) <= 0.1 then '01. 0 - 10%'
          when total_exposure/(total_exposure_cap*final_income) <= 0.2 then '02. 10 - 20%'
          when total_exposure/(total_exposure_cap*final_income) <= 0.3 then '03. 20 - 30%'
          when total_exposure/(total_exposure_cap*final_income) <= 0.4 then '04. 30 - 40%'
          when total_exposure/(total_exposure_cap*final_income) <= 0.5 then '05. 40 - 50%'
          when total_exposure/(total_exposure_cap*final_income) <= 0.6 then '06. 50 - 60%'
          when total_exposure/(total_exposure_cap*final_income) <= 0.7 then '07. 60 - 70%'
          when total_exposure/(total_exposure_cap*final_income) <= 0.8 then '08. 70 - 80%'
          when total_exposure/(total_exposure_cap*final_income) <= 0.9 then '09. 80 - 90%'
          when total_exposure/(total_exposure_cap*final_income) <= 1.0 then '10. 90 - 100%'
          when total_exposure/(total_exposure_cap*final_income) > 1.0 then '11. > 100%'
          else '99. Else' end as exposure_over_cap_grp,
        case 
          when dpd_mob_2 >= 30 then 1
          else 0 end as bad_tag,
        case
          when final_income is null then '00. income: 0'
          when final_income =        0  then '00. income: 0'
          when final_income <=  3000000 then '01. income: 1 - 3 mio'
          when final_income <=  4500000 then '02. income: 3 - 4.5 mio'
          when final_income <=  8000000 then '03. income: 4.5 - 8 mio'
          when final_income <= 15000000 then '04. income: 8 - 15 mio'
          when final_income <= 25000000 then '05. income: 15 - 25 mio'
          when final_income <= 50000000 then '06. income: 25 - 50 mio'
          when final_income > 50000000 then '07. income: > 50 mio'
          else '99. Else' end as final_income_grp,
        case
          when max_CC_limit >= 50000000 then '01. CC lim > 50 mio rule'
          when collateral_flag > 0 then '02. Have Collateral Loan rule'
          when max_CC_limit > 0 then '03. CC lim < 50 mio rule'
          when min_izi_income_range = 0 then '04. Inc not verified- null izi data'
          when declared_income is not null and
                least(min_izi_income_range, declared_income) = min_izi_income_range then '05. Min Limit from Izi'      
          when declared_income is null and
                min_izi_income_range > 0 then '06. Min Limit from Izi - declared inc. null'
          when declared_income is not null and
                least(min_izi_income_range, declared_income) = declared_income then '07. Take Declared Income'      
          else '99. Else' end as proxy_income_rule,
        case
          when max_CC_limit >= 50000000 then '01. CC lim > 50 mio rule' -- income = lim/2
          when max_CC_limit <= 50000000
              and max_CC_limit >= 30000000 then '02. CC lim > 30 mio rule' --income = lim/2
          when count_CC_issuer > 2 then '03. CC issuer > 2' -- income 10 mio
          when max_CC_limit >= 10000000 then '04. CC lim 10-30' -- income = lim/3
          when collateral_flag > 0 then '05. Have Collateral Loan rule'
          when count_CC_issuer > 0 then '06. Have at least 1 CC' -- income = 3 mio
          when max_oth_loan_instalment > 0 then '07. Have Unsecured Loan'
          else '08. Base Limit' end as new_adj_proxy_income_rule,
        case
          when collateral_type_code = 9 then '00. Atome'
          when collateral_type_code = 7 then '01. Mortgage'
          when collateral_type_code = 6 then '02. Auto - Car'
          when collateral_type_code = 5 then '03. Auto - Bike'
          when collateral_type_code = 2 then '04. BPR/BPD'
          when collateral_type_code = 1 then '05. Others'
          when collateral_type_code = 0 then '06. No Collateral'
          else '99. Else' end as Coll_type,
        case
          when max_collateral_loan_instalment  =        0 then '00. 0 '
          when max_collateral_loan_instalment <=  2000000 then '01. 1-2 Mio'
          when max_collateral_loan_instalment <=  5000000 then '02. 2-5 Mio'
          when max_collateral_loan_instalment <=  6000000 then '03. 5-6 Mio'
          when max_collateral_loan_instalment <=  8000000 then '04. 6-8 Mio'
          when max_collateral_loan_instalment <= 10000000 then '05. 8-10 Mio'
          when max_collateral_loan_instalment  > 10000000 then '06. >10 Mio'
          else '99. Else' end as coll_loan_instalment_grp,
        case
          when max_collateral_loan_tenor = 0 then '00. Tenor 0'
          when max_collateral_loan_tenor <= 6 then '01. Tenor 1-6 mths'
          when max_collateral_loan_tenor <= 12 then '02. Tenor 7-12 mths'
          when max_collateral_loan_tenor <= 18 then '03. Tenor 13-18 mths'
          when max_collateral_loan_tenor <= 24 then '04. Tenor 19-24 mths'
          when max_collateral_loan_tenor <= 36 then '05. Tenor 25-36 mths'
          when max_collateral_loan_tenor <= 48 then '06. Tenor 37-48 mths'
          when max_collateral_loan_tenor <= 60 then '07. Tenor 49-60 mths'
          when max_collateral_loan_tenor > 60 then '08. Tenor >60 mths'
          else '99. else' end as coll_loan_tenor_grp,
        case
          when max_kta_loan_instalment  =        0 then '00. 0 '
          when max_kta_loan_instalment <=  2000000 then '01. 1-2 Mio'
          when max_kta_loan_instalment <=  4000000 then '02. 2-4 Mio'
          when max_kta_loan_instalment <=  6000000 then '03. 4-6 Mio'
          when max_kta_loan_instalment <=  8000000 then '04. 6-8 Mio'
          when max_kta_loan_instalment <= 10000000 then '05. 8-10 Mio'
          when max_kta_loan_instalment  > 10000000 then '06. >10 Mio'
          else '99. Else' end as kta_loan_instalment_grp,
        case
          when max_fintech_instalment  =        0 then '00. 0 '
          when max_fintech_instalment <=  2000000 then '01. 1-2 Mio'
          when max_fintech_instalment <=  4000000 then '02. 2-4 Mio'
          when max_fintech_instalment <=  6000000 then '03. 4-6 Mio'
          when max_fintech_instalment <=  8000000 then '04. 6-8 Mio'
          when max_fintech_instalment <= 10000000 then '05. 8-10 Mio'
          when max_fintech_instalment  > 10000000 then '06. >10 Mio'
          else '99. Else' end as fintech_instalment_grp,
        case
          when max_oth_loan_instalment  =        0 then '00. 0 '
          when max_oth_loan_instalment <=  2000000 then '01. 1-2 Mio'
          when max_oth_loan_instalment <=  4000000 then '02. 2-4 Mio'
          when max_oth_loan_instalment <=  6000000 then '03. 4-6 Mio'
          when max_oth_loan_instalment <=  8000000 then '04. 6-8 Mio'
          when max_oth_loan_instalment <= 10000000 then '05. 8-10 Mio'
          when max_oth_loan_instalment  > 10000000 then '06. >10 Mio'
          else '99. Else' end as oth_loan_instalment_grp,
        case
          when limit_amt  =        0 then '00. 0 '
          when limit_amt <=  2000000 then '01. 1 - 2 Mio'
          when limit_amt <=  5000000 then '02. 2 - 5 Mio'
          when limit_amt <= 10000000 then '03. 5 - 10 Mio'
          when limit_amt <= 15000000 then '04. 10 - 15 Mio'
          when limit_amt <= 20000000 then '05. 15 - 20 Mio'
          when limit_amt <= 25000000 then '06. 20 - 25 Mio'
          when limit_amt <= 30000000 then '07. 25 - 30 Mio'
          else '99. Else' end as SK_limit_amt_grp,
        case
          when max_non_collateral_loan_lim  =        0 then '00. 0 '
          when max_non_collateral_loan_lim <=  2000000 then '01. 1 - 2 Mio'
          when max_non_collateral_loan_lim <=  5000000 then '02. 2 - 5 Mio'
          when max_non_collateral_loan_lim <= 10000000 then '03. 5 - 10 Mio'
          when max_non_collateral_loan_lim <= 15000000 then '04. 10 - 15 Mio'
          when max_non_collateral_loan_lim <= 20000000 then '05. 15 - 20 Mio'
          when max_non_collateral_loan_lim <= 25000000 then '06. 20 - 25 Mio'
          when max_non_collateral_loan_lim <= 30000000 then '07. 25 - 30 Mio'
          else '99. Else' end as non_collateral_limit_amt_grp,
        case
          when clik_bureau_info = 'NoBureau' and 
              (  salary_izi_score is null
              or salary_izi_score = 0) then 'NoBureau-No Alt Score'
          else clik_bureau_info end as clik_bureau_info_2,
        case
          when credit_bureau_score is null then '00. Null'
          when credit_bureau_score  =   0 then '00. 0'
          when credit_bureau_score <= 149 then '00. 1 - 149'
          when credit_bureau_score <= 267 then '01. 150 - 267'
          when credit_bureau_score <= 379 then '02. 268 - 379'
          when credit_bureau_score <= 519 then '03. 380 - 519'
          when credit_bureau_score <= 545 then '04. 520 - 545'
          when credit_bureau_score <= 561 then '05. 546 - 561'
          when credit_bureau_score  > 561 then '06. above 561' --up to 659 based on docs....
          else '99. else' end as clik_bur_score_grp_2,
        case
          when alternative_advance_ai_score is null then '00. null'          
          when alternative_advance_ai_score  =    0 then '00. 0'
          when alternative_advance_ai_score <=  499 then '01. 370 - 499'
          when alternative_advance_ai_score <=  530 then '02. 500 - 530'
          when alternative_advance_ai_score <=  559 then '03. 531 - 559'
          when alternative_advance_ai_score <=  593 then '04. 560 - 593'
          when alternative_advance_ai_score <=  627 then '05. 594 - 627'
          when alternative_advance_ai_score <=  806 then '06. 628 - 806'
          when alternative_advance_ai_score >   806 then '07. above 806'
          else '99. Else' end as bps_score_grp_2,
        case
          when credit_bureau_score is null then '00. Null'
          when credit_bureau_score  =   0 then '00. 0'
          when credit_bureau_score <= 320 then '00. 1 - 320'
          when credit_bureau_score <= 378 then '01. 320 - 378'
          when credit_bureau_score <= 550 then '02. 378 - 550'
          when credit_bureau_score  > 550 then '03. above 550' --up to 659 based on docs....
          else '99. else' end as clik_bur_score_old_grp,
        case
          when alternative_advance_ai_score is null then '00. null'          
          when alternative_advance_ai_score  =    0 then '00. 0'
          when alternative_advance_ai_score <=  500 then '01. 370 - 500'
          when alternative_advance_ai_score <=  538 then '02. 500 - 538'
          when alternative_advance_ai_score <=  575 then '03. 539 - 575'
          when alternative_advance_ai_score >   575 then '04. above 575'
          else '99. Else' end as bps_score_old_grp_1,
        case
          when alternative_advance_ai_score is null then '00. null'          
          when alternative_advance_ai_score  =    0 then '00. 0'
          when alternative_advance_ai_score <=  564 then '01. 370 - 564'
          when alternative_advance_ai_score <=  575 then '02. 564 - 575'
          when alternative_advance_ai_score <=  606 then '03. 575 - 606'
          when alternative_advance_ai_score >   606 then '07. above 606'
          else '99. Else' end as bps_score_old_grp_2,
        case
          when clik_bureau_info in ('Thick')
              and credit_bureau_score < 380 then '01. High Risk'
          when clik_bureau_info in ('Thick')
              and credit_bureau_score < 520 then '02. Medium Risk'
          when clik_bureau_info in ('Thick')
              and credit_bureau_score < 546 then '03. Low2 Risk'
          when clik_bureau_info in ('Thick')
              and credit_bureau_score < 562 then '04. Low1 Risk'
          when clik_bureau_info in ('Thick')
              and credit_bureau_score >= 562 then '05. Very Low Risk'
          
          when clik_bureau_info in ('Thin')
              and credit_bureau_score < 546 
              and alternative_advance_ai_score < 565 then '01. High Risk'
          when clik_bureau_info in ('Thin')
              and credit_bureau_score < 520 
              and alternative_advance_ai_score < 576 then '01. High Risk'
          when clik_bureau_info in ('Thin')
              and credit_bureau_score < 380 
              and alternative_advance_ai_score < 607 then '01. High Risk'
          when clik_bureau_info in ('Thin')
              and credit_bureau_score < 900 
              and alternative_advance_ai_score < 565 then '02. Medium Risk'
          when clik_bureau_info in ('Thin')
              and credit_bureau_score < 562 
              and alternative_advance_ai_score < 576 then '02. Medium Risk'
          when clik_bureau_info in ('Thin')
              and credit_bureau_score < 562 
              and alternative_advance_ai_score < 607 then '02. Medium Risk'
          when clik_bureau_info in ('Thin')
              and credit_bureau_score < 520
              and alternative_advance_ai_score < 1000 then '02. Medium Risk'
          when clik_bureau_info in ('Thin') then '03. Low2 Risk'
          else '99. Non Bureau' end as new_final_risk_level
          
  from combine_2
  where t24_customer_id not in ('101174407','101858872','102072000',
          '102084939','102072802','102096422') -- fraud
),
comb_band_2 as (
  select *,
        case
          when clik_bureau_info in ('Thick')
              and new_final_risk_level = '05. Very Low Risk' then 12
          when clik_bureau_info in ('Thick')
              and new_final_risk_level = '04. Low1 Risk' then 11
          when clik_bureau_info in ('Thick')
              and new_final_risk_level = '03. Low2 Risk' then 10
          when clik_bureau_info in ('Thick')
              and new_final_risk_level = '02. Medium Risk' then 9
          when clik_bureau_info in ('Thin')
              and new_final_risk_level = '03. Low2 Risk' then 10
          when clik_bureau_info in ('Thin')
              and new_final_risk_level = '02. Medium Risk' then 9
          else 0 end as new_limit_multiplier,
        case
          when clik_bureau_info in ('Thick')
              and new_final_risk_level = '05. Very Low Risk' then 70
          when clik_bureau_info in ('Thick')
              and new_final_risk_level = '04. Low1 Risk' then 65
          when clik_bureau_info in ('Thick')
              and new_final_risk_level = '03. Low2 Risk' then 60
          when clik_bureau_info in ('Thick')
              and new_final_risk_level = '02. Medium Risk' then 55
          when clik_bureau_info in ('Thin')
              and new_final_risk_level = '03. Low2 Risk' then 60
          when clik_bureau_info in ('Thin')
              and new_final_risk_level = '02. Medium Risk' then 55
          else 0 end as new_dbr_exposure,
        case
          when clik_bureau_info in ('Thick')
              and new_final_risk_level = '05. Very Low Risk' then 14
          when clik_bureau_info in ('Thick')
              and new_final_risk_level = '04. Low1 Risk' then 12
          when clik_bureau_info in ('Thick')
              and new_final_risk_level = '03. Low2 Risk' then 10
          when clik_bureau_info in ('Thick')
              and new_final_risk_level = '02. Medium Risk' then 8
          when clik_bureau_info in ('Thin')
              and new_final_risk_level = '03. Low2 Risk' then 10
          when clik_bureau_info in ('Thin')
              and new_final_risk_level = '02. Medium Risk' then 8
          else 0 end as new_total_exposure_cap,
        case
          when clik_bureau_info in ('Thick')
              and new_final_risk_level = '05. Very Low Risk' then 7
          when clik_bureau_info in ('Thick')
              and new_final_risk_level = '04. Low1 Risk' then 6
          when clik_bureau_info in ('Thick')
              and new_final_risk_level = '03. Low2 Risk' then 5
          when clik_bureau_info in ('Thick')
              and new_final_risk_level = '02. Medium Risk' then 4
          when clik_bureau_info in ('Thin')
              and new_final_risk_level = '03. Low2 Risk' then 5
          when clik_bureau_info in ('Thin')
              and new_final_risk_level = '02. Medium Risk' then 4
          else 0 end as new_internal_exposure_cap,

        case
          when new_adj_proxy_income_rule in ('01. CC lim > 50 mio rule','02. CC lim > 30 mio rule')
                then max_CC_limit/2
          when new_adj_proxy_income_rule in ('03. CC issuer > 2') then 10000000
          when new_adj_proxy_income_rule in ('04. CC lim 10-30') then max_CC_limit/3
          when new_adj_proxy_income_rule in ('05. Have Collateral Loan rule') then max_collateral_loan_instalment*2.5
          when new_adj_proxy_income_rule in ('06. Have at least 1 CC') then 3000000
          when new_adj_proxy_income_rule in ('07. Have Unsecured Loan') then sum_oth_loan_instalment*1.6
          else 0 end as new_proxy_income_calc,
  from comb_band
),
comb_band_3 as (
  select *,
        case
          when (new_dbr_exposure-final_dbr) < 0 then 0
          else (new_dbr_exposure-final_dbr)*new_proxy_income_calc*new_limit_multiplier end as new_limit_calculation,
        greatest(0,(new_dbr_exposure-final_dbr)*new_proxy_income_calc) as new_available_instalment
  from comb_band_2
),
comb_band_4 as (
  select *,
        least(30000000,
            greatest((new_proxy_income_calc*new_internal_exposure_cap)-internal_exposure,0),
            greatest((new_proxy_income_calc*new_total_exposure_cap)-total_exposure,0),
            greatest((new_available_instalment*new_limit_multiplier)/(1.02*greatest(new_limit_multiplier,1)),0)
            ) as new_limit_calc_tbu

        -- case
        --   when limit_amt = 30000000 then 30000000
        --   when new_limit_calculation > 30000000 then 30000000
        --   when new_limit_calculation > internal_exposure_cap * final_income then least(internal_exposure_cap * final_income,30000000)
        --   when new_limit_calculation > total_exposure_cap * final_income then least(total_exposure_cap * final_income,30000000)
        --   else new_limit_calculation end as new_limit_calc_tbu
  from comb_band_3
),
limit_cust_level as (
  select limit_id,
          new_adj_proxy_income_rule,
          max(limit_amt) as existing_limit,
          max(new_limit_calc_tbu) as new_limit_calc,
          max(final_income) as final_income,
          max(new_proxy_income_calc) as new_proxy_income_calc,
  from comb_band_4
  group by limit_id,new_adj_proxy_income_rule
),
raw_3 as (
  select 
      disb_mth,
      ever_funded,
      DBR_perc,
      final_income_grp,
      proxy_income_rule,
      clik_bureau_info_2,
      exposure_over_cap_grp,
      Coll_type,
      coll_loan_instalment_grp,
      oth_loan_instalment_grp,
      SK_limit_amt_grp,
      non_collateral_limit_amt_grp,
      final_risk_level,
      clik_bur_score_grp_2,
      bps_score_grp_2,
      clik_bur_score_old_grp,
      bps_score_old_grp_1,
      bps_score_old_grp_2,
      new_final_risk_level,
      new_adj_proxy_income_rule,
      coll_loan_tenor_grp,
      sum(disb_amt) as total_disb,
      sum(FPD14) as FPD14,
      sum(SPD30) as SPD30,
      count(distinct t24_customer_id) as count_client,
      sum(limit_amt) as total_limit,
      sum(new_limit_calc_tbu) as total_new_limit_tbu
    
from comb_band_4
group by 
      disb_mth,
      ever_funded,
      DBR_perc,
      final_income_grp,
      proxy_income_rule,
      clik_bureau_info_2,
      exposure_over_cap_grp,
      Coll_type,
      coll_loan_instalment_grp,
      KTA_loan_grp,
      oth_loan_instalment_grp,
      SK_limit_amt_grp,
      non_collateral_limit_amt_grp,
      final_risk_level,
      clik_bur_score_grp_2,
      bps_score_grp_2,
      clik_bur_score_old_grp,
      bps_score_old_grp_1,
      bps_score_old_grp_2,
      new_final_risk_level,
      new_adj_proxy_income_rule,
      coll_loan_tenor_grp
),
raw_3_bin as (
  select final_dbr,
        arrangement_id,
        final_income,
        salary_izi_score,
        credit_bureau_score,
        alternative_advance_ai_score,
        clik_bureau_info_2,
        bad_tag
  from comb_band
),
raw_3_check as (
  select *
  from comb_band
  where customer_info_id in (
    select customer_id
    from bur_detail_2
    where collaterals_num > 0
      and provider_code_desc in ('PT Atome Finance Indonesia')
  )
  and proxy_income_rule not in ('01. Have CC with higher limit')
),
cust_limit_grp as (
  select new_adj_proxy_income_rule,
          sum(1) as count_client,
          sum(existing_limit) as sum_Existing_limit,
          sum(new_limit_calc) as sum_new_limit_calc,
          sum(final_income) as sum_final_income_old,
          sum(new_proxy_income_calc) as sum_new_proxy_income_calc
  from limit_cust_level
  group by new_adj_proxy_income_rule
)
select *
from 
-- cust_limit_grp
raw_3
'''
	limit_dbr = client_con.query(sql).to_dataframe()
	return limit_dbr


def reject_excap_bdown(client_con):
	
	##DECLARE date1 STRING DEFAULT '2025-07-31';

	sql = '''
With limit_base as (
  select 
      t24_customer_id,
      limit_application_id,
      cast(mis_date as string) as mis_date,
      limit_id,
      cast(limit_amount as numeric) as limit_amt,
      date(datetime(created_at)) as create_dt,
      customer_info_id,
      reason
  from joey-bi-prod-project.staging.stg_loan__personal_loan_sys_limit_application
  where string(mis_date) = date1
    and status = "rejected"
),
rej_reason_dict as (
  select *
  from joey-bi-ss-risk-fraud-project.credit.Saku_kredit_reject_reason_desc_2025_04_30
),
funded_detail as (
  select t24_customer_id,
        max(ever_funded) as ever_funded
  from joey-bi-ss-risk-fraud-project.credit.funded_info_2025_04
  group by t24_customer_id
),
bur_detail as (
    SELECT customer_id,
          count_num,
          provider_code_desc,
          contract_phase_desc,
          contract_type,
          contract_type_desc,
          start_date,
          due_date,
          past_due_status_code as latest_kol,
          past_due_status_desc,
          debit_bal as latest_os,
          dpd as latest_dpd,
          max_dpd,
          worst_status,
          worst_status_desc,
          interest_rate,
          credit_limit,
          collaterals_num,
          tot_collateral_value,
          (cast(substr(due_date,1,4) as int)*12 + cast(substr(due_date,6,2) as int))
          - (cast(substr(start_date,1,4) as int)*12 + cast(substr(start_date,6,2) as int)
          ) as tenor,
          init_credit_lim,
          case
            when interest_rate is null
                or interest_rate = 0 then 0.0001
            else interest_rate end as int_cast

  FROM joey-bi-prod-project.staging.stg_credit_report__clik_credit_granted__hive_merge
  where string(mis_date) = date1
),
bur_detail_2 as (
  select *,
        case
          when tot_collateral_value > 0
              and provider_code_desc in ('PT Atome Finance Indonesia') then 9
          when tot_collateral_value > 0
              and provider_code_desc in ('PT Bank Tabungan Negara (Persero) Tbk') then 7
          when tot_collateral_value > 0
              and provider_code_desc in ('PT Bank Danamon Indonesia Tbk','PT Bank Central Asia Tbk',
              'PT Bank Mandiri (Persero) Tbk','PT Bank Syariah Indonesia','PT Bank Mega Tbk',
              'PT Bank Mega Syariah','PT Bank Rakyat Indonesia (Persero) Tbk',
              'PT Bank Negara Indonesia (Persero) Tbk') 
              and tenor > 60 then 7
          when tot_collateral_value > 0
              and provider_code_desc in ('PT Astra Sedaya Finance') then 6
          when tot_collateral_value > 0
              and provider_code_desc in ('PT Bank Danamon Indonesia Tbk','PT Bank Central Asia Tbk',
              'PT Bank Mandiri (Persero) Tbk','PT Bank Syariah Indonesia','PT Bank Mega Tbk',
              'PT Bank Mega Syariah','PT Bank Rakyat Indonesia (Persero) Tbk',
              'PT Bank Negara Indonesia (Persero) Tbk') 
              and tenor <= 60 then 6
          when tot_collateral_value > 0
              and provider_code_desc in ('PT BCA Finance','PT KB Finansia Multi Finance',
              'PT Mandiri Utama Finance','PT Wahana Ottomitra Multiartha Tbk','PT Mega Central Finance',
              'PT Bussan Auto Finance (UUS)','PT Mandiri Tunas Finance','PT Mega Auto Finance',
              'PT Mega Finance','PT Bussan Auto Finance')
              and tenor > 24 then 6
          when tot_collateral_value > 0
              and provider_code_desc in ('PT Adira Dinamika Multi Finance Tbk',
              'PT Federal International Finance','PT BFI Finance Indonesia',
              'PT Summit Oto Finance','PT Adira Dinamika Multi Finance (UUS)',
              'PT Federal International Finance (UUS)') then 5
          when tot_collateral_value > 0
              and provider_code_desc in ('PT BCA Finance','PT KB Finansia Multi Finance',
              'PT Mandiri Utama Finance','PT Wahana Ottomitra Multiartha Tbk','PT Mega Central Finance',
              'PT Bussan Auto Finance (UUS)','PT Mandiri Tunas Finance','PT Mega Auto Finance',
              'PT Mega Finance','PT Bussan Auto Finance')
              and tenor <= 24 then 5
          when tot_collateral_value > 0
              and provider_code_desc like '%Rakyat%' then 2
          when tot_collateral_value > 0
              and provider_code_desc like '%BPR%' then 2
          when tot_collateral_value > 0
              and provider_code_desc like '%BPD%' then 2
          when tot_collateral_value > 0 then 1
          else 0 end as coll_type,
      case
        when tenor <= 0 then 1
        else tenor end as tenor_cast

  from bur_detail
),
bur_detail_client_list as(
  select customer_id,
        max(count_num) as count_num
  from bur_detail
  group by customer_id 
),
bur_detail_client_info as (
  select customer_id,
        count_num,
        max(case
          when contract_type_desc = 'Credit Card' then credit_limit
          else 0 end) as max_CC_limit,
        sum(case
          when contract_type_desc = 'Credit Card' then credit_limit
          else 0 end) as sum_CC_limit,
        sum(case
          when contract_type_desc = 'Credit Card' then latest_os
          else 0 end) as sum_CC_OS,

        max(tot_collateral_value) as collateral_flag,
        max(coll_type) as collateral_type_code,
        max(case
          when tot_collateral_value >0 then latest_os
          else 0 end) as max_collateral_loan_OS,
        max(case
          when tot_collateral_value >0 then 
                  init_credit_lim / (power(1+int_cast/100/12,tenor_cast)-1) * 
                    (int_cast/100/12*power(1+int_cast/100/12,tenor_cast))
          else 0 end) as max_collateral_loan_instalment,
        
        sum(case
          when contract_type_desc in ('Others','Loans provided')
              and tot_collateral_value = 0 then latest_os
            else 0 end) as sum_kta_loan_OS,
        max(case
          when contract_type_desc in ('Others','Loans provided')
              and tot_collateral_value = 0 then interest_rate
            else 0 end) as max_kta_int_rate,
        max(case
          when contract_type_desc in ('Others','Loans provided')
              and tot_collateral_value = 0 then 
                  init_credit_lim / (power(1+int_cast/100/12,tenor_cast)-1) * 
                    (int_cast/100/12*power(1+int_cast/100/12,tenor_cast))
          else 0 end) as max_kta_loan_instalment,
        
        sum(case
          when contract_type in ('10','20','P01','P10')
              and tot_collateral_value = 0 then latest_os
              else 0 end) as sum_fintech_loan_OS,
        max(case
          when contract_type in ('10','20','P01','P10')
              and tot_collateral_value = 0 then interest_rate
              else 0 end) as max_fintech_int_rate,
        max(case
          when contract_type in ('10','20','P01','P10')
              and tot_collateral_value = 0 then 
                  init_credit_lim / (power(1+int_cast/100/12,tenor_cast)-1) * 
                    (int_cast/100/12*power(1+int_cast/100/12,tenor_cast))
          else 0 end) as max_fintech_instalment,
        max(case
          when (contract_type in ('10','20','P01','P10')
                or contract_type_desc in ('Others','Loans provided'))
              and tot_collateral_value = 0 then 
                  init_credit_lim / (power(1+int_cast/100/12,tenor_cast)-1) * 
                    (int_cast/100/12*power(1+int_cast/100/12,tenor_cast))
          else 0 end) as max_oth_loan_instalment,
        
        max(case
          when tot_collateral_value = 0 then
                init_credit_lim
                else 0 end) as max_non_collateral_loan_lim,
        max(case
          when contract_type_desc = 'Credit Card' then latest_os*5/100
          else init_credit_lim / (power(1+int_cast/100/12,tenor_cast)-1) * 
                    (int_cast/100/12*power(1+int_cast/100/12,tenor_cast))
           end) as sum_existing_instalment

  from bur_detail_2
  where contract_phase_desc = 'Active'
  group by customer_id, count_num
),
cred_summary as (
  select application_id,
          bureau_type as clik_bureau_info,
          income_verified as izi_income_verification,
          final_income,
          monthly_debt_repayment,
          final_dbr,
          internal_exposure,
          total_exposure,
          internal_exposure_cap,
          total_exposure_cap,
          exposure_dbr,
          final_risk_level
  from joey-bi-prod-project.staging.stg_loan__loan_credit_summary
  where STRING(mis_date)=date1
),
izi_salary as (
  select loan_id,
        salary,
        income,
        RANK() OVER (PARTITION BY loan_id ORDER BY pull_datetime DESC)
          AS data_rank
  from staging.stg_loan__sake_izi_salary
  where STRING(mis_date)=date1
),
izi_salary_list as (
  select loan_id,
          max(data_rank) as data_rank
  from izi_salary
  group by loan_id
),
work_info as (
  select id,
        declared_income
  from joey-bi-prod-project.staging.stg_loan__sake_limit_work_info
  where STRING(mis_date)=date1
),
combine as (
  select a.*,
          b.DescCode,
          d.ever_funded,
          g.count_num,
          h.clik_bureau_info,
          h.izi_income_verification,
          h.final_income,
          h.monthly_debt_repayment,
          h.final_dbr,
          h.internal_exposure,
          h.total_exposure,
          h.internal_exposure_cap,
          h.total_exposure_cap,
          h.exposure_dbr,
          h.final_risk_level,
          R.data_rank,
          S.salary as salary_izi_score,
          S.income as izi_income_range,
          T.declared_income
  from limit_base a  
  left join rej_reason_dict b
          on a.reason = b.Reason
  left join funded_detail d on a.t24_customer_id = cast(d.t24_customer_id as string) 
  left join bur_detail_client_list g on a.customer_info_id = g.customer_id 
  left join cred_summary h on a.limit_id = h.application_id 
  left join izi_salary_list R on a.limit_id = R.loan_id
  left join izi_salary S on R.loan_id = S.loan_id
                        and R.data_rank = S.data_rank
  left join work_info T on a.limit_id = T.id

),
combine_2 as (
  select a.*,
        case
          when a.izi_income_range = '0-3999999' then 1000000
          when a.izi_income_range = '4000000-7999999' then 4000000
          when a.izi_income_range = '8000000-9999999' then 8000000
          when a.izi_income_range = '10000000-999999999999' then 10000000
          else 0 end as min_izi_income_range,
        case
          when final_income is null 
              or final_income = 0 then 0
          else internal_exposure_cap * final_income end as MICE_amt,
        case
          when final_income is null 
              or final_income = 0 then 0
          else total_exposure_cap * final_income end as UE_amt,
        case
          when final_income is null 
              or final_income = 0 then 0
          else (exposure_dbr * final_income/100)-sum_existing_instalment end as available_instalment,
        case
          when clik_bureau_info = 'Thick'
              and final_risk_level = 'High' then 6
          when clik_bureau_info = 'Thick'
              and final_risk_level = 'Medium' then 8
          when clik_bureau_info = 'Thick'
              and final_risk_level = 'Low' then 10
          when clik_bureau_info = 'Thin'
              and final_risk_level = 'High' then 4
          when clik_bureau_info = 'Thin'
              and final_risk_level = 'Medium' then 6
          when clik_bureau_info = 'Thin'
              and final_risk_level = 'Low' then 8
          when clik_bureau_info = 'NoBureau'
              and final_risk_level = 'High' then 2
          when clik_bureau_info = 'NoBureau'
              and final_risk_level = 'Medium' then 3
          when clik_bureau_info = 'NoBureau'
              and final_risk_level = 'Low' then 4
          when final_risk_level = 'Low' then 4
          when final_risk_level = 'Medium' then 3
          else 0 end as Limit_tenor_multiplier,
                
        b.max_CC_limit,
        b.sum_CC_limit,
        b.sum_CC_OS,

        b.collateral_flag,
        b.collateral_type_code,
        b.max_collateral_loan_OS,
        b.max_collateral_loan_instalment,
        
        b.sum_kta_loan_OS,
        b.max_kta_int_rate,
        b.max_kta_loan_instalment,
        
        b.sum_fintech_loan_OS,
        b.max_fintech_int_rate,
        b.max_fintech_instalment,

        b.max_oth_loan_instalment,
        b.max_non_collateral_loan_lim,
        b.sum_existing_instalment
  from combine a
  left join bur_detail_client_info b on a.customer_info_id = b.customer_id
          and a.count_num = b.count_num 
),
comb_band as (
  select *,
        case
          when max_CC_limit > limit_amt then '01. Have CC with higher limit'
          when sum_kta_loan_OS > limit_amt then '02. Have KTA with higher limit'
          when sum_fintech_loan_OS > limit_amt then '03. Have fintech with higher limit'
          when max_kta_int_rate <= 1.99 then '04. Have KTA with lower interest'
          when max_fintech_int_rate <= 1.99 then '05. Have fintech with lower interest'
          else '99. Other reason' end as Reason_not_used,
        case
          when max_CC_limit > 100000000 then '01. > 100 mio'
          when max_CC_limit >  70000000 then '02. 70 - 100 mio'
          when max_CC_limit >  50000000 then '03. 50 - 70 mio'
          when max_CC_limit >  30000000 then '04. 30 - 50 mio'
          when max_CC_limit >  20000000 then '05. 20 - 30 mio'
          when max_CC_limit >  10000000 then '06. 10 - 20 mio'
          when max_CC_limit >   5000000 then '07.  5 - 10 mio'
          when max_CC_limit >         0 then '08.  1 -  5 mio'
          else '09. Else' end as CC_limit_grp,
        case
          when sum_kta_loan_OS > 100000000 then '01. > 100 mio'
          when sum_kta_loan_OS >  70000000 then '02. 70 - 100 mio'
          when sum_kta_loan_OS >  50000000 then '03. 50 - 70 mio'
          when sum_kta_loan_OS >  30000000 then '04. 30 - 50 mio'
          when sum_kta_loan_OS >  20000000 then '05. 20 - 30 mio'
          when sum_kta_loan_OS >  10000000 then '06. 10 - 20 mio'
          when sum_kta_loan_OS >   5000000 then '07.  5 - 10 mio'
          when sum_kta_loan_OS >         0 then '08.  1 -  5 mio'
          else '09. Else' end as KTA_loan_grp,
        case
          when sum_fintech_loan_OS > 100000000 then '01. > 100 mio'
          when sum_fintech_loan_OS >  70000000 then '02. 70 - 100 mio'
          when sum_fintech_loan_OS >  50000000 then '03. 50 - 70 mio'
          when sum_fintech_loan_OS >  30000000 then '04. 30 - 50 mio'
          when sum_fintech_loan_OS >  20000000 then '05. 20 - 30 mio'
          when sum_fintech_loan_OS >  10000000 then '06. 10 - 20 mio'
          when sum_fintech_loan_OS >   5000000 then '07.  5 - 10 mio'
          when sum_fintech_loan_OS >         0 then '08.  1 -  5 mio'
          else '09. Else' end as fintech_loan_grp,
        case
          when max_kta_int_rate > 1.99 then '01. > 1.99%'
          when max_kta_int_rate > 1.70 then '02. 1.70 - 1.99%'
          when max_kta_int_rate > 1.50 then '03. 1.50 - 1.70%'
          when max_kta_int_rate > 1.30 then '04. 1.30 - 1.50%'
          when max_kta_int_rate > 1.10 then '05. 1.10 - 1.30%'
          when max_kta_int_rate > 0.90 then '06. 0.90 - 1.10%'
          when max_kta_int_rate > 0.70 then '07. 0.70 - 0.90%'
          when max_kta_int_rate > 0.50 then '08. 0.50 - 0.70%'
          when max_kta_int_rate > 0.25 then '09. 0.25 - 0.50%'
          when max_kta_int_rate > 0 then '10. 0.01 - 0.25%'
          when max_kta_int_rate = 0 then '11. 0%'
          else '12. Else' end as kta_int_grp,
        case
          when max_fintech_int_rate > 1.99 then '01. > 1.99%'
          when max_fintech_int_rate > 1.70 then '02. 1.70 - 1.99%'
          when max_fintech_int_rate > 1.50 then '03. 1.50 - 1.70%'
          when max_fintech_int_rate > 1.30 then '04. 1.30 - 1.50%'
          when max_fintech_int_rate > 1.10 then '05. 1.10 - 1.30%'
          when max_fintech_int_rate > 0.90 then '06. 0.90 - 1.10%'
          when max_fintech_int_rate > 0.70 then '07. 0.70 - 0.90%'
          when max_fintech_int_rate > 0.50 then '08. 0.50 - 0.70%'
          when max_fintech_int_rate > 0.25 then '09. 0.25 - 0.50%'
          when max_fintech_int_rate > 0 then '10. 0.01 - 0.25%'
          when max_fintech_int_rate = 0 then '11. 0%'
          else '12. Else' end as fintech_int_grp,
        case
          when final_dbr = 0 then '01. DBR 0%'
          when final_dbr <= 10 then '02. DBR 1 - 10%'
          when final_dbr <= 20 then '03. DBR 11 - 20%'
          when final_dbr <= 30 then '04. DBR 21 - 30%'
          when final_dbr <= 40 then '05. DBR 31 - 40%'
          when final_dbr <= 50 then '06. DBR 41 - 50%'
          when final_dbr <= 60 then '07. DBR 51 - 60%'
          when final_dbr <= 80 then '08. DBR 61 - 80%'
          when final_dbr <= 100 then '09. DBR 81 - 100%'
          when final_dbr > 100 then '10. DBR > 100%'
          else '99. Else' end as DBR_perc,
        case
          when final_income is null or 
                final_income = 0 then '00. No income data'
          when total_exposure is null or 
                total_exposure = 0 then '00. 0%'
          when total_exposure/(total_exposure_cap*final_income) = 0 then '00. 0%'
          when total_exposure/(total_exposure_cap*final_income) <= 0.1 then '01. 0 - 10%'
          when total_exposure/(total_exposure_cap*final_income) <= 0.2 then '02. 10 - 20%'
          when total_exposure/(total_exposure_cap*final_income) <= 0.3 then '03. 20 - 30%'
          when total_exposure/(total_exposure_cap*final_income) <= 0.4 then '04. 30 - 40%'
          when total_exposure/(total_exposure_cap*final_income) <= 0.5 then '05. 40 - 50%'
          when total_exposure/(total_exposure_cap*final_income) <= 0.6 then '06. 50 - 60%'
          when total_exposure/(total_exposure_cap*final_income) <= 0.7 then '07. 60 - 70%'
          when total_exposure/(total_exposure_cap*final_income) <= 0.8 then '08. 70 - 80%'
          when total_exposure/(total_exposure_cap*final_income) <= 0.9 then '09. 80 - 90%'
          when total_exposure/(total_exposure_cap*final_income) <= 1.0 then '10. 90 - 100%'
          when total_exposure/(total_exposure_cap*final_income) > 1.0 then '11. > 100%'
          else '99. Else' end as exposure_over_cap_grp,
        case
          when final_income is null then '00. income: 0'
          when final_income =        0  then '00. income: 0'
          when final_income <=  3000000 then '01. income: 1 - 3 mio'
          when final_income <=  4500000 then '02. income: 3 - 4.5 mio'
          when final_income <=  8000000 then '03. income: 4.5 - 8 mio'
          when final_income <= 15000000 then '04. income: 8 - 15 mio'
          when final_income <= 25000000 then '05. income: 15 - 25 mio'
          when final_income <= 50000000 then '06. income: 25 - 50 mio'
          when final_income > 50000000 then '07. income: > 50 mio'
          else '99. Else' end as final_income_grp,
        case
          when max_CC_limit >= 50000000 then '01. CC lim > 50 mio rule'
          when collateral_flag > 0 then '02. Have Collateral Loan rule'
          when max_CC_limit > 0 then '03. CC lim < 50 mio rule'
          when min_izi_income_range = 0 then '04. Inc not verified- null izi data'
          when declared_income is not null and
                least(min_izi_income_range, declared_income) = min_izi_income_range then '05. Min Limit from Izi'      
          when declared_income is null and
                min_izi_income_range > 0 then '06. Min Limit from Izi - declared inc. null'
          when declared_income is not null and
                least(min_izi_income_range, declared_income) = declared_income then '07. Take Declared Income'      
          else '99. Else' end as proxy_income_rule,
        case
          when collateral_type_code = 9 then '00. Atome'
          when collateral_type_code = 7 then '01. Mortgage'
          when collateral_type_code = 6 then '02. Auto - Car'
          when collateral_type_code = 5 then '03. Auto - Bike'
          when collateral_type_code = 2 then '04. BPR/BPD'
          when collateral_type_code = 1 then '05. Others'
          when collateral_type_code = 0 then '06. No Collateral'
          else '99. Else' end as Coll_type,
        case
          when max_collateral_loan_instalment  =        0 then '00. 0 '
          when max_collateral_loan_instalment <=  2000000 then '01. 1-2 Mio'
          when max_collateral_loan_instalment <=  5000000 then '02. 2-5 Mio'
          when max_collateral_loan_instalment <=  6000000 then '03. 5-6 Mio'
          when max_collateral_loan_instalment <=  8000000 then '04. 6-8 Mio'
          when max_collateral_loan_instalment <= 10000000 then '05. 8-10 Mio'
          when max_collateral_loan_instalment  > 10000000 then '06. >10 Mio'
          else '99. Else' end as coll_loan_instalment_grp,
        case
          when max_kta_loan_instalment  =        0 then '00. 0 '
          when max_kta_loan_instalment <=  2000000 then '01. 1-2 Mio'
          when max_kta_loan_instalment <=  4000000 then '02. 2-4 Mio'
          when max_kta_loan_instalment <=  6000000 then '03. 4-6 Mio'
          when max_kta_loan_instalment <=  8000000 then '04. 6-8 Mio'
          when max_kta_loan_instalment <= 10000000 then '05. 8-10 Mio'
          when max_kta_loan_instalment  > 10000000 then '06. >10 Mio'
          else '99. Else' end as kta_loan_instalment_grp,
        case
          when max_fintech_instalment  =        0 then '00. 0 '
          when max_fintech_instalment <=  2000000 then '01. 1-2 Mio'
          when max_fintech_instalment <=  4000000 then '02. 2-4 Mio'
          when max_fintech_instalment <=  6000000 then '03. 4-6 Mio'
          when max_fintech_instalment <=  8000000 then '04. 6-8 Mio'
          when max_fintech_instalment <= 10000000 then '05. 8-10 Mio'
          when max_fintech_instalment  > 10000000 then '06. >10 Mio'
          else '99. Else' end as fintech_instalment_grp,
        case
          when max_oth_loan_instalment  =        0 then '00. 0 '
          when max_oth_loan_instalment <=  2000000 then '01. 1-2 Mio'
          when max_oth_loan_instalment <=  4000000 then '02. 2-4 Mio'
          when max_oth_loan_instalment <=  6000000 then '03. 4-6 Mio'
          when max_oth_loan_instalment <=  8000000 then '04. 6-8 Mio'
          when max_oth_loan_instalment <= 10000000 then '05. 8-10 Mio'
          when max_oth_loan_instalment  > 10000000 then '06. >10 Mio'
          else '99. Else' end as oth_loan_instalment_grp,
        case
          when limit_amt  =        0 then '00. 0 '
          when limit_amt <=  2000000 then '01. 1 - 2 Mio'
          when limit_amt <=  5000000 then '02. 2 - 5 Mio'
          when limit_amt <= 10000000 then '03. 5 - 10 Mio'
          when limit_amt <= 15000000 then '04. 10 - 15 Mio'
          when limit_amt <= 20000000 then '05. 15 - 20 Mio'
          when limit_amt <= 25000000 then '06. 20 - 25 Mio'
          when limit_amt <= 30000000 then '07. 25 - 30 Mio'
          else '99. Else' end as SK_limit_amt_grp,
        case
          when max_non_collateral_loan_lim  =        0 then '00. 0 '
          when max_non_collateral_loan_lim <=  2000000 then '01. 1 - 2 Mio'
          when max_non_collateral_loan_lim <=  5000000 then '02. 2 - 5 Mio'
          when max_non_collateral_loan_lim <= 10000000 then '03. 5 - 10 Mio'
          when max_non_collateral_loan_lim <= 15000000 then '04. 10 - 15 Mio'
          when max_non_collateral_loan_lim <= 20000000 then '05. 15 - 20 Mio'
          when max_non_collateral_loan_lim <= 25000000 then '06. 20 - 25 Mio'
          when max_non_collateral_loan_lim <= 30000000 then '07. 25 - 30 Mio'
          else '99. Else' end as non_collateral_limit_amt_grp,
        case
          when MICE_amt - internal_exposure < 0 then '01. Limit by MICE cap'
          when UE_Amt - total_exposure < 0 then '02. Limit by UE cap'
          when final_income is null
                or final_income = 0 then '03. no income data'
          when (final_dbr) > exposure_dbr/100  then '04. Limit by DBR cap'
          -- when (sum_existing_instalment/final_income) > exposure_dbr/100  then '04. Limit by DBR cap'
          else '99. Else' end as limit_class_det
          
          
  from combine_2
  where t24_customer_id not in ('101174407','101858872','102072000',
          '102084939','102072802','102096422') -- fraud
),
comb_band_2 as (
  select *,
        case
          when proxy_income_rule in ('04. Inc not verified- null izi data','05. Min Limit from Izi'      
          ,'06. Min Limit from Izi - declared inc. null','07. Take Declared Income')
            and (sum_existing_instalment is not null
                or sum_existing_instalment != 0)
             then sum_existing_instalment
          else final_income end as new_proxy_income
  from comb_band
),
comb_band_3 as (
  select *,
        case 
          when internal_exposure > internal_exposure_cap * new_proxy_income then '01. Hit MICE'
          when total_exposure > total_exposure_cap * new_proxy_income then '02. Hit Excap'
          when final_dbr > exposure_dbr then '03. Hit DBR'
          when (exposure_dbr-final_dbr)*new_proxy_income*Limit_tenor_multiplier < 1000000 then '04. Hit min Limit'
          else '05. Passed by new proxy income' end as new_proxy_income_rej_rule,
        case
          when (exposure_dbr-final_dbr) < 0 then 0
          else (exposure_dbr-final_dbr)*new_proxy_income*Limit_tenor_multiplier end as new_proxy_income_new_limit
  from comb_band_2
),
raw_3b as (
  select 
      ever_funded,
      DescCode,
      DBR_perc,
      final_income_grp,
      proxy_income_rule,
      clik_bureau_info,
      exposure_over_cap_grp,
      Coll_type,
      coll_loan_instalment_grp,
      oth_loan_instalment_grp,
      SK_limit_amt_grp,
      non_collateral_limit_amt_grp,
      limit_class_det,
      new_proxy_income_rej_rule,
      count(distinct t24_customer_id) as count_client,
      sum(new_proxy_income_new_limit) as total_new_proxy_income_new_limit
      
from comb_band_3
group by 
      ever_funded,
      DescCode,
      DBR_perc,
      final_income_grp,
      proxy_income_rule,
      clik_bureau_info,
      exposure_over_cap_grp,
      Coll_type,
      coll_loan_instalment_grp,
      KTA_loan_grp,
      oth_loan_instalment_grp,
      SK_limit_amt_grp,
      non_collateral_limit_amt_grp,
      limit_class_det,
      new_proxy_income_rej_rule
)
select *
from raw_3b
-- from comb_band
-- where DescCode = 'DBR/Excap'

'''
	reject_excap_bdown = client_con.query(sql).to_dataframe()
	return reject_excap_bdown

def New_table_New_Risk_level(client_con):
	
	##DECLARE date1 STRING DEFAULT '2025-07-31';

	sql = '''
-- drop table joey-bi-ss-risk-fraud-project.credit.new_risk_level_2025_07;
create table joey-bi-ss-risk-fraud-project.credit.new_risk_level_2025_07 as
With limit_base as (
  select 
      t24_customer_id,
      cast(mis_date as string) as mis_date,
      limit_id,
      cast(limit_amount as numeric) as limit_amt,
      date(datetime(created_at)) as create_dt,
      customer_info_id,
      status
  from joey-bi-prod-project.staging.stg_loan__personal_loan_sys_limit_application
  where STRING(mis_date)= date1
    -- and status = "agreementSigned"
),
-- disb_base as (
--   select arrangement_id,
--           limit_id,
--           limit_loan_amount as disb_amt,
--           tenor,
--           product_code,
--           collectability,
--           created_at as disb_at,
--           substr(cast(created_at as STRING),0,10) as disb_dt,
--           concat(substr(string(created_at),1,4),'-',substr(string(created_at),6,2)) as disb_mth,
--           extract(hour from created_at) as disb_hr
--   from joey-bi-prod-project.staging.stg_collection__limit_loan disb
--   where STRING(mis_date)= date1
-- ),
flagging_fund_ntb as (
  select cast(t24_customer_id as string) as t24_customer_id,
        min(ever_funded) as flagging,
        -- max(uninstall) as uninstall
  from ( select t24_customer_id,
                ever_funded,
                -- uninstall
        from joey-bi-ss-risk-fraud-project.credit.funded_info_2025_06
        
        UNION ALL

        select t24_customer_id,
                ever_funded,
                -- cast(uninstall as int)
        from joey-bi-ss-risk-fraud-project.credit.funded_info_2025_08
  )  group by t24_customer_id

  union ALL
  
  select t24_customer_id,
        max(2) as flagging, --NTB
        -- max(0) as uninstall
  from joey-bi-prod-project.staging.ntb_prefilter_saku_kredit
  group by t24_customer_id
),
flagging_grp as (
  select t24_customer_id,
      case
        when max(flagging) = 2 then 'NTB'
        when max(flagging) = 1 then '1'
        when max(flagging) = 0 then '0'
        else '' end as flagging,
      -- max(uninstall) as uninstall
  from flagging_fund_ntb
  group by t24_customer_id
),
-- repay_data as (
--   select a.arrangement_id,
--           cast(a.mis_date as string) as mis_date,
--           sum(a.owed_amount) as sum_owed,
--           max(a.dpd) as DPD_acct,
--           b.limit_id
--   from joey-bi-prod-project.staging.stg_collection__repayment_schedule a
--   left join joey-bi-prod-project.staging.stg_collection__limit_loan b
--         on a.arrangement_id = b.arrangement_id
--         and String(b.mis_date) = date1
--   where a.mis_date = last_day(a.mis_date)
--   group by a.arrangement_id, a.mis_date, b.limit_id
-- ),
bur_detail as (
    SELECT customer_id,
          count_num,
          provider_code_desc,
          contract_phase_desc,
          contract_type,
          contract_type_desc,
          start_date,
          due_date,
          past_due_status_code as latest_kol,
          past_due_status_desc,
          debit_bal as latest_os,
          dpd as latest_dpd,
          max_dpd,
          worst_status,
          worst_status_desc,
          interest_rate,
          credit_limit,
          collaterals_num,
          tot_collateral_value,
          (cast(substr(due_date,1,4) as int)*12 + cast(substr(due_date,6,2) as int))
          - (cast(substr(start_date,1,4) as int)*12 + cast(substr(start_date,6,2) as int)
          ) as tenor,
          init_credit_lim,
          case
            when interest_rate is null
                or interest_rate = 0 then 0.0001
            else interest_rate end as int_cast

  FROM joey-bi-prod-project.staging.stg_credit_report__clik_credit_granted__hive_merge
  where string(mis_date) = date1
),
bur_detail_2 as (
  select *,
        case
          when tot_collateral_value > 0
              and provider_code_desc in ('PT Atome Finance Indonesia') then 9
          when tot_collateral_value > 0
              and provider_code_desc in ('PT Bank Tabungan Negara (Persero) Tbk') then 7
          when tot_collateral_value > 0
              and provider_code_desc in ('PT Bank Danamon Indonesia Tbk','PT Bank Central Asia Tbk',
              'PT Bank Mandiri (Persero) Tbk','PT Bank Syariah Indonesia','PT Bank Mega Tbk',
              'PT Bank Mega Syariah','PT Bank Rakyat Indonesia (Persero) Tbk',
              'PT Bank Negara Indonesia (Persero) Tbk') 
              and tenor > 60 then 7
          when tot_collateral_value > 0
              and provider_code_desc in ('PT Astra Sedaya Finance') then 6
          when tot_collateral_value > 0
              and provider_code_desc in ('PT Bank Danamon Indonesia Tbk','PT Bank Central Asia Tbk',
              'PT Bank Mandiri (Persero) Tbk','PT Bank Syariah Indonesia','PT Bank Mega Tbk',
              'PT Bank Mega Syariah','PT Bank Rakyat Indonesia (Persero) Tbk',
              'PT Bank Negara Indonesia (Persero) Tbk') 
              and tenor <= 60 then 6
          when tot_collateral_value > 0
              and provider_code_desc in ('PT BCA Finance','PT KB Finansia Multi Finance',
              'PT Mandiri Utama Finance','PT Wahana Ottomitra Multiartha Tbk','PT Mega Central Finance',
              'PT Bussan Auto Finance (UUS)','PT Mandiri Tunas Finance','PT Mega Auto Finance',
              'PT Mega Finance','PT Bussan Auto Finance')
              and tenor > 24 then 6
          when tot_collateral_value > 0
              and provider_code_desc in ('PT Adira Dinamika Multi Finance Tbk',
              'PT Federal International Finance','PT BFI Finance Indonesia',
              'PT Summit Oto Finance','PT Adira Dinamika Multi Finance (UUS)',
              'PT Federal International Finance (UUS)') then 5
          when tot_collateral_value > 0
              and provider_code_desc in ('PT BCA Finance','PT KB Finansia Multi Finance',
              'PT Mandiri Utama Finance','PT Wahana Ottomitra Multiartha Tbk','PT Mega Central Finance',
              'PT Bussan Auto Finance (UUS)','PT Mandiri Tunas Finance','PT Mega Auto Finance',
              'PT Mega Finance','PT Bussan Auto Finance')
              and tenor <= 24 then 5
          when tot_collateral_value > 0
              and provider_code_desc like '%Rakyat%' then 2
          when tot_collateral_value > 0
              and provider_code_desc like '%BPR%' then 2
          when tot_collateral_value > 0
              and provider_code_desc like '%BPD%' then 2
          when tot_collateral_value > 0 then 1
          else 0 end as coll_type,
      case
        when tenor <= 0 then 1
        else tenor end as tenor_cast

  from bur_detail
),
bur_detail_client_list as(
  select customer_id,
        max(count_num) as count_num
  from bur_detail
  group by customer_id 
),
bur_detail_client_info as (
  select customer_id,
        count_num,
        max(case
          when contract_type_desc = 'Credit Card' then credit_limit
          else 0 end) as max_CC_limit,
        sum(case
          when contract_type_desc = 'Credit Card' then credit_limit
          else 0 end) as sum_CC_limit,
        sum(case
          when contract_type_desc = 'Credit Card' then latest_os
          else 0 end) as sum_CC_OS,
        sum(case
          when contract_type_desc = 'Credit Card' then 1
          else 0 end) as count_CC_issuer,

        max(tot_collateral_value) as collateral_flag,
        max(coll_type) as collateral_type_code,
        max(case
          when tot_collateral_value >0 then latest_os
          else 0 end) as max_collateral_loan_OS,
        max(case
          when tot_collateral_value >0 then 
                  init_credit_lim / (power(1+int_cast/100/12,tenor_cast)-1) * 
                    (int_cast/100/12*power(1+int_cast/100/12,tenor_cast))
          else 0 end) as max_collateral_loan_instalment,
        max(case
          when tot_collateral_value >0 then 
                  tenor
          else 0 end) as max_collateral_loan_tenor,
        
        sum(case
          when contract_type_desc in ('Others','Loans provided')
              and tot_collateral_value = 0 then latest_os
            else 0 end) as sum_kta_loan_OS,
        max(case
          when contract_type_desc in ('Others','Loans provided')
              and tot_collateral_value = 0 then interest_rate
            else 0 end) as max_kta_int_rate,
        max(case
          when contract_type_desc in ('Others','Loans provided')
              and tot_collateral_value = 0 then 
                  init_credit_lim / (power(1+int_cast/100/12,tenor_cast)-1) * 
                    (int_cast/100/12*power(1+int_cast/100/12,tenor_cast))
          else 0 end) as max_kta_loan_instalment,
        
        sum(case
          when contract_type in ('10','20','P01','P10')
              and tot_collateral_value = 0 then latest_os
              else 0 end) as sum_fintech_loan_OS,
        max(case
          when contract_type in ('10','20','P01','P10')
              and tot_collateral_value = 0 then interest_rate
              else 0 end) as max_fintech_int_rate,
        max(case
          when contract_type in ('10','20','P01','P10')
              and tot_collateral_value = 0 then 
                  init_credit_lim / (power(1+int_cast/100/12,tenor_cast)-1) * 
                    (int_cast/100/12*power(1+int_cast/100/12,tenor_cast))
          else 0 end) as max_fintech_instalment,
        max(case
          when (contract_type in ('10','20','P01','P10')
                or contract_type_desc in ('Others','Loans provided'))
              and tot_collateral_value = 0 then 
                  init_credit_lim / (power(1+int_cast/100/12,tenor_cast)-1) * 
                    (int_cast/100/12*power(1+int_cast/100/12,tenor_cast))
          else 0 end) as max_oth_loan_instalment,
        sum(case
          when (contract_type in ('10','20','P01','P10')
                or contract_type_desc in ('Others','Loans provided'))
              and tot_collateral_value = 0 then 
                  init_credit_lim / (power(1+int_cast/100/12,tenor_cast)-1) * 
                    (int_cast/100/12*power(1+int_cast/100/12,tenor_cast))
          else 0 end) as sum_oth_loan_instalment,
        
        max(case
          when tot_collateral_value = 0 then
                init_credit_lim
                else 0 end) as max_non_collateral_loan_lim

  from bur_detail_2
  where contract_phase_desc = 'Active'
  group by customer_id, count_num
),
cred_summary as (
  select application_id,
          bureau_type as clik_bureau_info,
          income_verified as izi_income_verification,
          final_income,
          monthly_debt_repayment,
          final_dbr,
          internal_exposure,
          total_exposure,
          internal_exposure_cap,
          total_exposure_cap,
          exposure_dbr,
          final_risk_level,
          credit_bureau_score,
          alternative_advance_ai_score
  from joey-bi-prod-project.staging.stg_loan__loan_credit_summary
  where STRING(mis_date)=date1
),
izi_salary as (
  select loan_id,
        salary,
        income,
        RANK() OVER (PARTITION BY loan_id ORDER BY pull_datetime DESC)
          AS data_rank
  from staging.stg_loan__sake_izi_salary
  where STRING(mis_date)=date1
),
izi_salary_list as (
  select loan_id,
          max(data_rank) as data_rank
  from izi_salary
  group by loan_id
),
work_info as (
  select id,
        declared_income
  from joey-bi-prod-project.staging.stg_loan__sake_limit_work_info
  where STRING(mis_date)=date1
),
combine as (
  select a.*,
          -- b.disb_amt,
          -- b.disb_mth,
          -- b.arrangement_id,
          d.flagging,
          g.count_num,
          h.clik_bureau_info,
          h.izi_income_verification,
          h.final_income,
          h.monthly_debt_repayment,
          h.final_dbr,
          h.internal_exposure,
          h.total_exposure,
          h.internal_exposure_cap,
          h.total_exposure_cap,
          h.exposure_dbr,
          h.final_risk_level,
          h.credit_bureau_score,
          h.alternative_advance_ai_score,
          -- M.DPD_acct as dpd_mob_2,
          R.data_rank,
          S.salary as salary_izi_score,
          S.income as izi_income_range,
          T.declared_income
  from limit_base a
  -- left join disb_base b on a.limit_id = cast(b.limit_id as STRING)
  left join flagging_grp d on a.t24_customer_id = cast(d.t24_customer_id as string) 
  left join bur_detail_client_list g on a.customer_info_id = g.customer_id 
  left join cred_summary h on a.limit_id = h.application_id 
  -- left join repay_data M on b.arrangement_id = M.arrangement_id
  --     and cast(substr(b.disb_mth,1,4) as int)*12 + cast(substr(b.disb_mth,6,2) as int)
  --         = cast(substr(M.mis_date,1,4) as int)*12 + cast(substr(M.mis_date,6,2) as int) -2
  left join izi_salary_list R on a.limit_id = R.loan_id
  left join izi_salary S on R.loan_id = S.loan_id
                        and R.data_rank = S.data_rank
  left join work_info T on a.limit_id = T.id

),
combine_2 as (
  select a.*,
        case
          when a.izi_income_range = '0-3999999' then 1000000
          when a.izi_income_range = '4000000-7999999' then 4000000
          when a.izi_income_range = '8000000-9999999' then 8000000
          when a.izi_income_range = '10000000-999999999999' then 10000000
          else 0 end as min_izi_income_range,

        b.max_CC_limit,
        b.sum_CC_limit,
        b.sum_CC_OS,
        b.count_CC_issuer,

        b.collateral_flag,
        b.collateral_type_code,
        b.max_collateral_loan_OS,
        b.max_collateral_loan_instalment,
        b.max_collateral_loan_tenor,
        
        b.sum_kta_loan_OS,
        b.max_kta_int_rate,
        b.max_kta_loan_instalment,
        
        b.sum_fintech_loan_OS,
        b.max_fintech_int_rate,
        b.max_fintech_instalment,

        b.max_oth_loan_instalment,
        b.sum_oth_loan_instalment,
        b.max_non_collateral_loan_lim
  from combine a
  left join bur_detail_client_info b on a.customer_info_id = b.customer_id
          and a.count_num = b.count_num 
),
comb_band as (
  select *,
        case
          when max_CC_limit > limit_amt then '01. Have CC with higher limit'
          when sum_kta_loan_OS > limit_amt then '02. Have KTA with higher limit'
          when sum_fintech_loan_OS > limit_amt then '03. Have fintech with higher limit'
          when max_kta_int_rate <= 1.99 then '04. Have KTA with lower interest'
          when max_fintech_int_rate <= 1.99 then '05. Have fintech with lower interest'
          else '99. Other reason' end as Reason_not_used,
        case
          when max_CC_limit > 100000000 then '01. > 100 mio'
          when max_CC_limit >  70000000 then '02. 70 - 100 mio'
          when max_CC_limit >  50000000 then '03. 50 - 70 mio'
          when max_CC_limit >  30000000 then '04. 30 - 50 mio'
          when max_CC_limit >  20000000 then '05. 20 - 30 mio'
          when max_CC_limit >  10000000 then '06. 10 - 20 mio'
          when max_CC_limit >   5000000 then '07.  5 - 10 mio'
          when max_CC_limit >         0 then '08.  1 -  5 mio'
          else '09. Else' end as CC_limit_grp,
        case
          when sum_kta_loan_OS > 100000000 then '01. > 100 mio'
          when sum_kta_loan_OS >  70000000 then '02. 70 - 100 mio'
          when sum_kta_loan_OS >  50000000 then '03. 50 - 70 mio'
          when sum_kta_loan_OS >  30000000 then '04. 30 - 50 mio'
          when sum_kta_loan_OS >  20000000 then '05. 20 - 30 mio'
          when sum_kta_loan_OS >  10000000 then '06. 10 - 20 mio'
          when sum_kta_loan_OS >   5000000 then '07.  5 - 10 mio'
          when sum_kta_loan_OS >         0 then '08.  1 -  5 mio'
          else '09. Else' end as KTA_loan_grp,
        case
          when sum_fintech_loan_OS > 100000000 then '01. > 100 mio'
          when sum_fintech_loan_OS >  70000000 then '02. 70 - 100 mio'
          when sum_fintech_loan_OS >  50000000 then '03. 50 - 70 mio'
          when sum_fintech_loan_OS >  30000000 then '04. 30 - 50 mio'
          when sum_fintech_loan_OS >  20000000 then '05. 20 - 30 mio'
          when sum_fintech_loan_OS >  10000000 then '06. 10 - 20 mio'
          when sum_fintech_loan_OS >   5000000 then '07.  5 - 10 mio'
          when sum_fintech_loan_OS >         0 then '08.  1 -  5 mio'
          else '09. Else' end as fintech_loan_grp,
        case
          when max_kta_int_rate > 1.99 then '01. > 1.99%'
          when max_kta_int_rate > 1.70 then '02. 1.70 - 1.99%'
          when max_kta_int_rate > 1.50 then '03. 1.50 - 1.70%'
          when max_kta_int_rate > 1.30 then '04. 1.30 - 1.50%'
          when max_kta_int_rate > 1.10 then '05. 1.10 - 1.30%'
          when max_kta_int_rate > 0.90 then '06. 0.90 - 1.10%'
          when max_kta_int_rate > 0.70 then '07. 0.70 - 0.90%'
          when max_kta_int_rate > 0.50 then '08. 0.50 - 0.70%'
          when max_kta_int_rate > 0.25 then '09. 0.25 - 0.50%'
          when max_kta_int_rate > 0 then '10. 0.01 - 0.25%'
          when max_kta_int_rate = 0 then '11. 0%'
          else '12. Else' end as kta_int_grp,
        case
          when max_fintech_int_rate > 1.99 then '01. > 1.99%'
          when max_fintech_int_rate > 1.70 then '02. 1.70 - 1.99%'
          when max_fintech_int_rate > 1.50 then '03. 1.50 - 1.70%'
          when max_fintech_int_rate > 1.30 then '04. 1.30 - 1.50%'
          when max_fintech_int_rate > 1.10 then '05. 1.10 - 1.30%'
          when max_fintech_int_rate > 0.90 then '06. 0.90 - 1.10%'
          when max_fintech_int_rate > 0.70 then '07. 0.70 - 0.90%'
          when max_fintech_int_rate > 0.50 then '08. 0.50 - 0.70%'
          when max_fintech_int_rate > 0.25 then '09. 0.25 - 0.50%'
          when max_fintech_int_rate > 0 then '10. 0.01 - 0.25%'
          when max_fintech_int_rate = 0 then '11. 0%'
          else '12. Else' end as fintech_int_grp,
        -- case 
        --   when dpd_mob_2 >= 14 then disb_amt
        --   else 0 end as FPD14,
        -- case 
        --   when dpd_mob_2 >= 30 then disb_amt
        --   else 0 end as SPD30,
        case
          when final_dbr = 0 then '01. DBR 0%'
          when final_dbr <= 10 then '02. DBR 1 - 10%'
          when final_dbr <= 20 then '03. DBR 11 - 20%'
          when final_dbr <= 30 then '04. DBR 21 - 30%'
          when final_dbr <= 40 then '05. DBR 31 - 40%'
          when final_dbr <= 50 then '06. DBR 41 - 50%'
          when final_dbr <= 60 then '07. DBR 51 - 60%'
          when final_dbr <= 80 then '08. DBR 61 - 80%'
          when final_dbr <= 100 then '09. DBR 81 - 100%'
          when final_dbr > 100 then '10. DBR > 100%'
          else '99. Else' end as DBR_perc,
        case
          when final_income is null or 
                final_income = 0 then '00. No income data'
          when total_exposure is null or 
                total_exposure = 0 then '00. 0%'
          when total_exposure/(total_exposure_cap*final_income) = 0 then '00. 0%'
          when total_exposure/(total_exposure_cap*final_income) <= 0.1 then '01. 0 - 10%'
          when total_exposure/(total_exposure_cap*final_income) <= 0.2 then '02. 10 - 20%'
          when total_exposure/(total_exposure_cap*final_income) <= 0.3 then '03. 20 - 30%'
          when total_exposure/(total_exposure_cap*final_income) <= 0.4 then '04. 30 - 40%'
          when total_exposure/(total_exposure_cap*final_income) <= 0.5 then '05. 40 - 50%'
          when total_exposure/(total_exposure_cap*final_income) <= 0.6 then '06. 50 - 60%'
          when total_exposure/(total_exposure_cap*final_income) <= 0.7 then '07. 60 - 70%'
          when total_exposure/(total_exposure_cap*final_income) <= 0.8 then '08. 70 - 80%'
          when total_exposure/(total_exposure_cap*final_income) <= 0.9 then '09. 80 - 90%'
          when total_exposure/(total_exposure_cap*final_income) <= 1.0 then '10. 90 - 100%'
          when total_exposure/(total_exposure_cap*final_income) > 1.0 then '11. > 100%'
          else '99. Else' end as exposure_over_cap_grp,
        -- case 
        --   when dpd_mob_2 >= 30 then 1
        --   else 0 end as bad_tag,
        case
          when final_income is null then '00. income: 0'
          when final_income =        0  then '00. income: 0'
          when final_income <=  3000000 then '01. income: 1 - 3 mio'
          when final_income <=  4500000 then '02. income: 3 - 4.5 mio'
          when final_income <=  8000000 then '03. income: 4.5 - 8 mio'
          when final_income <= 15000000 then '04. income: 8 - 15 mio'
          when final_income <= 25000000 then '05. income: 15 - 25 mio'
          when final_income <= 50000000 then '06. income: 25 - 50 mio'
          when final_income > 50000000 then '07. income: > 50 mio'
          else '99. Else' end as final_income_grp,
        case
          when max_CC_limit >= 50000000 then '01. CC lim > 50 mio rule'
          when collateral_flag > 0 then '02. Have Collateral Loan rule'
          when max_CC_limit > 0 then '03. CC lim < 50 mio rule'
          when min_izi_income_range = 0 then '04. Inc not verified- null izi data'
          when declared_income is not null and
                least(min_izi_income_range, declared_income) = min_izi_income_range then '05. Min Limit from Izi'      
          when declared_income is null and
                min_izi_income_range > 0 then '06. Min Limit from Izi - declared inc. null'
          when declared_income is not null and
                least(min_izi_income_range, declared_income) = declared_income then '07. Take Declared Income'      
          else '99. Else' end as proxy_income_rule,
        case
          when max_CC_limit >= 50000000 then '01. CC lim > 50 mio rule' -- income = lim/2
          when max_CC_limit <= 50000000
              and max_CC_limit >= 30000000 then '02. CC lim > 30 mio rule' --income = lim/2
          when count_CC_issuer > 2 then '03. CC issuer > 2' -- income 10 mio
          when max_CC_limit >= 10000000 then '04. CC lim 10-30' -- income = lim/3
          when collateral_flag > 0 then '05. Have Collateral Loan rule'
          when count_CC_issuer > 0 then '06. Have at least 1 CC' -- income = 3 mio
          when max_oth_loan_instalment > 0 then '07. Have Unsecured Loan'
          else '08. Base Limit' end as new_adj_proxy_income_rule,
        case
          when collateral_type_code = 9 then '00. Atome'
          when collateral_type_code = 7 then '01. Mortgage'
          when collateral_type_code = 6 then '02. Auto - Car'
          when collateral_type_code = 5 then '03. Auto - Bike'
          when collateral_type_code = 2 then '04. BPR/BPD'
          when collateral_type_code = 1 then '05. Others'
          when collateral_type_code = 0 then '06. No Collateral'
          else '99. Else' end as Coll_type,
        case
          when max_collateral_loan_instalment  =        0 then '00. 0 '
          when max_collateral_loan_instalment <=  2000000 then '01. 1-2 Mio'
          when max_collateral_loan_instalment <=  5000000 then '02. 2-5 Mio'
          when max_collateral_loan_instalment <=  6000000 then '03. 5-6 Mio'
          when max_collateral_loan_instalment <=  8000000 then '04. 6-8 Mio'
          when max_collateral_loan_instalment <= 10000000 then '05. 8-10 Mio'
          when max_collateral_loan_instalment  > 10000000 then '06. >10 Mio'
          else '99. Else' end as coll_loan_instalment_grp,
        case
          when max_collateral_loan_tenor = 0 then '00. Tenor 0'
          when max_collateral_loan_tenor <= 6 then '01. Tenor 1-6 mths'
          when max_collateral_loan_tenor <= 12 then '02. Tenor 7-12 mths'
          when max_collateral_loan_tenor <= 18 then '03. Tenor 13-18 mths'
          when max_collateral_loan_tenor <= 24 then '04. Tenor 19-24 mths'
          when max_collateral_loan_tenor <= 36 then '05. Tenor 25-36 mths'
          when max_collateral_loan_tenor <= 48 then '06. Tenor 37-48 mths'
          when max_collateral_loan_tenor <= 60 then '07. Tenor 49-60 mths'
          when max_collateral_loan_tenor > 60 then '08. Tenor >60 mths'
          else '99. else' end as coll_loan_tenor_grp,
        case
          when max_kta_loan_instalment  =        0 then '00. 0 '
          when max_kta_loan_instalment <=  2000000 then '01. 1-2 Mio'
          when max_kta_loan_instalment <=  4000000 then '02. 2-4 Mio'
          when max_kta_loan_instalment <=  6000000 then '03. 4-6 Mio'
          when max_kta_loan_instalment <=  8000000 then '04. 6-8 Mio'
          when max_kta_loan_instalment <= 10000000 then '05. 8-10 Mio'
          when max_kta_loan_instalment  > 10000000 then '06. >10 Mio'
          else '99. Else' end as kta_loan_instalment_grp,
        case
          when max_fintech_instalment  =        0 then '00. 0 '
          when max_fintech_instalment <=  2000000 then '01. 1-2 Mio'
          when max_fintech_instalment <=  4000000 then '02. 2-4 Mio'
          when max_fintech_instalment <=  6000000 then '03. 4-6 Mio'
          when max_fintech_instalment <=  8000000 then '04. 6-8 Mio'
          when max_fintech_instalment <= 10000000 then '05. 8-10 Mio'
          when max_fintech_instalment  > 10000000 then '06. >10 Mio'
          else '99. Else' end as fintech_instalment_grp,
        case
          when max_oth_loan_instalment  =        0 then '00. 0 '
          when max_oth_loan_instalment <=  2000000 then '01. 1-2 Mio'
          when max_oth_loan_instalment <=  4000000 then '02. 2-4 Mio'
          when max_oth_loan_instalment <=  6000000 then '03. 4-6 Mio'
          when max_oth_loan_instalment <=  8000000 then '04. 6-8 Mio'
          when max_oth_loan_instalment <= 10000000 then '05. 8-10 Mio'
          when max_oth_loan_instalment  > 10000000 then '06. >10 Mio'
          else '99. Else' end as oth_loan_instalment_grp,
        case
          when limit_amt  =        0 then '00. 0 '
          when limit_amt <=  2000000 then '01. 1 - 2 Mio'
          when limit_amt <=  5000000 then '02. 2 - 5 Mio'
          when limit_amt <= 10000000 then '03. 5 - 10 Mio'
          when limit_amt <= 15000000 then '04. 10 - 15 Mio'
          when limit_amt <= 20000000 then '05. 15 - 20 Mio'
          when limit_amt <= 25000000 then '06. 20 - 25 Mio'
          when limit_amt <= 30000000 then '07. 25 - 30 Mio'
          else '99. Else' end as SK_limit_amt_grp,
        case
          when max_non_collateral_loan_lim  =        0 then '00. 0 '
          when max_non_collateral_loan_lim <=  2000000 then '01. 1 - 2 Mio'
          when max_non_collateral_loan_lim <=  5000000 then '02. 2 - 5 Mio'
          when max_non_collateral_loan_lim <= 10000000 then '03. 5 - 10 Mio'
          when max_non_collateral_loan_lim <= 15000000 then '04. 10 - 15 Mio'
          when max_non_collateral_loan_lim <= 20000000 then '05. 15 - 20 Mio'
          when max_non_collateral_loan_lim <= 25000000 then '06. 20 - 25 Mio'
          when max_non_collateral_loan_lim <= 30000000 then '07. 25 - 30 Mio'
          else '99. Else' end as non_collateral_limit_amt_grp,
        case
          when clik_bureau_info = 'NoBureau' and 
              (  salary_izi_score is null
              or salary_izi_score = 0) then 'NoBureau-No Alt Score'
          else clik_bureau_info end as clik_bureau_info_2,
        case
          when credit_bureau_score is null then '00. Null'
          when credit_bureau_score  =   0 then '00. 0'
          when credit_bureau_score <= 149 then '00. 1 - 149'
          when credit_bureau_score <= 267 then '01. 150 - 267'
          when credit_bureau_score <= 379 then '02. 268 - 379'
          when credit_bureau_score <= 519 then '03. 380 - 519'
          when credit_bureau_score <= 545 then '04. 520 - 545'
          when credit_bureau_score <= 561 then '05. 546 - 561'
          when credit_bureau_score  > 561 then '06. above 561' --up to 659 based on docs....
          else '99. else' end as clik_bur_score_grp_2,
        case
          when alternative_advance_ai_score is null then '00. null'          
          when alternative_advance_ai_score  =    0 then '00. 0'
          when alternative_advance_ai_score <=  499 then '01. 370 - 499'
          when alternative_advance_ai_score <=  530 then '02. 500 - 530'
          when alternative_advance_ai_score <=  559 then '03. 531 - 559'
          when alternative_advance_ai_score <=  593 then '04. 560 - 593'
          when alternative_advance_ai_score <=  627 then '05. 594 - 627'
          when alternative_advance_ai_score <=  806 then '06. 628 - 806'
          when alternative_advance_ai_score >   806 then '07. above 806'
          else '99. Else' end as bps_score_grp_2,
        case
          when credit_bureau_score is null then '00. Null'
          when credit_bureau_score  =   0 then '00. 0'
          when credit_bureau_score <= 320 then '00. 1 - 320'
          when credit_bureau_score <= 378 then '01. 320 - 378'
          when credit_bureau_score <= 550 then '02. 378 - 550'
          when credit_bureau_score  > 550 then '03. above 550' --up to 659 based on docs....
          else '99. else' end as clik_bur_score_old_grp,
        case
          when alternative_advance_ai_score is null then '00. null'          
          when alternative_advance_ai_score  =    0 then '00. 0'
          when alternative_advance_ai_score <=  500 then '01. 370 - 500'
          when alternative_advance_ai_score <=  538 then '02. 500 - 538'
          when alternative_advance_ai_score <=  575 then '03. 539 - 575'
          when alternative_advance_ai_score >   575 then '04. above 575'
          else '99. Else' end as bps_score_old_grp_1,
        case
          when alternative_advance_ai_score is null then '00. null'          
          when alternative_advance_ai_score  =    0 then '00. 0'
          when alternative_advance_ai_score <=  564 then '01. 370 - 564'
          when alternative_advance_ai_score <=  575 then '02. 564 - 575'
          when alternative_advance_ai_score <=  606 then '03. 575 - 606'
          when alternative_advance_ai_score >   606 then '07. above 606'
          else '99. Else' end as bps_score_old_grp_2,
        case
          when clik_bureau_info in ('Thick')
              and credit_bureau_score < 380 then '01. High Risk'
          when clik_bureau_info in ('Thick')
              and credit_bureau_score < 520 then '02. Medium Risk'
          when clik_bureau_info in ('Thick')
              and credit_bureau_score < 546 then '03. Low2 Risk'
          when clik_bureau_info in ('Thick')
              and credit_bureau_score < 562 then '04. Low1 Risk'
          when clik_bureau_info in ('Thick')
              and credit_bureau_score >= 562 then '05. Very Low Risk'
          
          when clik_bureau_info in ('Thin')
              and credit_bureau_score < 546 
              and alternative_advance_ai_score < 565 then '01. High Risk'
          when clik_bureau_info in ('Thin')
              and credit_bureau_score < 520 
              and alternative_advance_ai_score < 576 then '01. High Risk'
          when clik_bureau_info in ('Thin')
              and credit_bureau_score < 380 
              and alternative_advance_ai_score < 607 then '01. High Risk'
          when clik_bureau_info in ('Thin')
              and credit_bureau_score < 900 
              and alternative_advance_ai_score < 565 then '02. Medium Risk'
          when clik_bureau_info in ('Thin')
              and credit_bureau_score < 562 
              and alternative_advance_ai_score < 576 then '02. Medium Risk'
          when clik_bureau_info in ('Thin')
              and credit_bureau_score < 562 
              and alternative_advance_ai_score < 607 then '02. Medium Risk'
          when clik_bureau_info in ('Thin')
              and credit_bureau_score < 520
              and alternative_advance_ai_score < 1000 then '02. Medium Risk'
          when clik_bureau_info in ('Thin') then '03. Low2 Risk'
          else '99. Non Bureau' end as new_final_risk_level
          
  from combine_2
  where t24_customer_id not in ('101174407','101858872','102072000',
          '102084939','102072802','102096422') -- fraud
),
comb_band_2 as (
  select *,
        case
          when clik_bureau_info in ('Thick')
              and new_final_risk_level = '05. Very Low Risk' then 12
          when clik_bureau_info in ('Thick')
              and new_final_risk_level = '04. Low1 Risk' then 11
          when clik_bureau_info in ('Thick')
              and new_final_risk_level = '03. Low2 Risk' then 10
          when clik_bureau_info in ('Thick')
              and new_final_risk_level = '02. Medium Risk' then 9
          when clik_bureau_info in ('Thin')
              and new_final_risk_level = '03. Low2 Risk' then 10
          when clik_bureau_info in ('Thin')
              and new_final_risk_level = '02. Medium Risk' then 9
          else 0 end as new_limit_multiplier,
        case
          when clik_bureau_info in ('Thick')
              and new_final_risk_level = '05. Very Low Risk' then 70
          when clik_bureau_info in ('Thick')
              and new_final_risk_level = '04. Low1 Risk' then 65
          when clik_bureau_info in ('Thick')
              and new_final_risk_level = '03. Low2 Risk' then 60
          when clik_bureau_info in ('Thick')
              and new_final_risk_level = '02. Medium Risk' then 55
          when clik_bureau_info in ('Thin')
              and new_final_risk_level = '03. Low2 Risk' then 60
          when clik_bureau_info in ('Thin')
              and new_final_risk_level = '02. Medium Risk' then 55
          else 0 end as new_dbr_exposure,
        case
          when clik_bureau_info in ('Thick')
              and new_final_risk_level = '05. Very Low Risk' then 14
          when clik_bureau_info in ('Thick')
              and new_final_risk_level = '04. Low1 Risk' then 12
          when clik_bureau_info in ('Thick')
              and new_final_risk_level = '03. Low2 Risk' then 10
          when clik_bureau_info in ('Thick')
              and new_final_risk_level = '02. Medium Risk' then 8
          when clik_bureau_info in ('Thin')
              and new_final_risk_level = '03. Low2 Risk' then 10
          when clik_bureau_info in ('Thin')
              and new_final_risk_level = '02. Medium Risk' then 8
          else 0 end as new_total_exposure_cap,
        case
          when clik_bureau_info in ('Thick')
              and new_final_risk_level = '05. Very Low Risk' then 7
          when clik_bureau_info in ('Thick')
              and new_final_risk_level = '04. Low1 Risk' then 6
          when clik_bureau_info in ('Thick')
              and new_final_risk_level = '03. Low2 Risk' then 5
          when clik_bureau_info in ('Thick')
              and new_final_risk_level = '02. Medium Risk' then 4
          when clik_bureau_info in ('Thin')
              and new_final_risk_level = '03. Low2 Risk' then 5
          when clik_bureau_info in ('Thin')
              and new_final_risk_level = '02. Medium Risk' then 4
          else 0 end as new_internal_exposure_cap,

        case
          when new_adj_proxy_income_rule in ('01. CC lim > 50 mio rule','02. CC lim > 30 mio rule')
                then max_CC_limit/2
          when new_adj_proxy_income_rule in ('03. CC issuer > 2') then 10000000
          when new_adj_proxy_income_rule in ('04. CC lim 10-30') then max_CC_limit/3
          when new_adj_proxy_income_rule in ('05. Have Collateral Loan rule') then max_collateral_loan_instalment*2.5
          when new_adj_proxy_income_rule in ('06. Have at least 1 CC') then 3000000
          when new_adj_proxy_income_rule in ('07. Have Unsecured Loan') then sum_oth_loan_instalment*1.6
          else 0 end as new_proxy_income_calc,
  from comb_band
),
comb_band_3 as (
  select *,
        case
          when (new_dbr_exposure-final_dbr) < 0 then 0
          else (new_dbr_exposure-final_dbr)*new_proxy_income_calc*new_limit_multiplier end as new_limit_calculation,
        greatest(0,(new_dbr_exposure-final_dbr)*new_proxy_income_calc) as new_available_instalment
  from comb_band_2
),
comb_band_4 as (
  select *,
        least(30000000,
            greatest((new_proxy_income_calc*new_internal_exposure_cap)-internal_exposure,0),
            greatest((new_proxy_income_calc*new_total_exposure_cap)-total_exposure,0),
            greatest((new_available_instalment*new_limit_multiplier)/(1.02*greatest(new_limit_multiplier,1)),0)
            ) as new_limit_calc_tbu

        -- case
        --   when limit_amt = 30000000 then 30000000
        --   when new_limit_calculation > 30000000 then 30000000
        --   when new_limit_calculation > internal_exposure_cap * final_income then least(internal_exposure_cap * final_income,30000000)
        --   when new_limit_calculation > total_exposure_cap * final_income then least(total_exposure_cap * final_income,30000000)
        --   else new_limit_calculation end as new_limit_calc_tbu
  from comb_band_3
),
limit_cust_level as (
  select limit_id,
          new_adj_proxy_income_rule,
          max(limit_amt) as existing_limit,
          max(new_limit_calc_tbu) as new_limit_calc,
          max(final_income) as final_income,
          max(new_proxy_income_calc) as new_proxy_income_calc,
  from comb_band_4
  group by limit_id,new_adj_proxy_income_rule
),
raw_3 as (
  select 
      -- disb_mth,
      flagging,
      DBR_perc,
      final_income_grp,
      proxy_income_rule,
      clik_bureau_info_2,
      exposure_over_cap_grp,
      Coll_type,
      coll_loan_instalment_grp,
      oth_loan_instalment_grp,
      SK_limit_amt_grp,
      non_collateral_limit_amt_grp,
      final_risk_level,
      clik_bur_score_grp_2,
      bps_score_grp_2,
      clik_bur_score_old_grp,
      bps_score_old_grp_1,
      bps_score_old_grp_2,
      new_final_risk_level,
      new_adj_proxy_income_rule,
      coll_loan_tenor_grp,
      status,
      -- sum(disb_amt) as total_disb,
      -- sum(FPD14) as FPD14,
      -- sum(SPD30) as SPD30,
      -- count(distinct t24_customer_id) as count_client,
      sum(limit_amt) as total_limit,
      sum(new_limit_calc_tbu) as total_new_limit_tbu
    
from comb_band_4
group by 
      -- disb_mth,
      flagging,
      DBR_perc,
      final_income_grp,
      proxy_income_rule,
      clik_bureau_info_2,
      exposure_over_cap_grp,
      Coll_type,
      coll_loan_instalment_grp,
      KTA_loan_grp,
      oth_loan_instalment_grp,
      SK_limit_amt_grp,
      non_collateral_limit_amt_grp,
      final_risk_level,
      clik_bur_score_grp_2,
      bps_score_grp_2,
      clik_bur_score_old_grp,
      bps_score_old_grp_1,
      bps_score_old_grp_2,
      new_final_risk_level,
      new_adj_proxy_income_rule,
      coll_loan_tenor_grp,
      status
),
raw_3_bin as (
  select final_dbr,
        -- arrangement_id,
        final_income,
        salary_izi_score,
        credit_bureau_score,
        alternative_advance_ai_score,
        clik_bureau_info_2,
        -- bad_tag
  from comb_band
),
raw_3_check as (
  select *
  from comb_band
  where customer_info_id in (
    select customer_id
    from bur_detail_2
    where collaterals_num > 0
      and provider_code_desc in ('PT Atome Finance Indonesia')
  )
  and proxy_income_rule not in ('01. Have CC with higher limit')
),
cust_limit_grp as (
  select new_adj_proxy_income_rule,
          sum(1) as count_client,
          sum(existing_limit) as sum_Existing_limit,
          sum(new_limit_calc) as sum_new_limit_calc,
          sum(final_income) as sum_final_income_old,
          sum(new_proxy_income_calc) as sum_new_proxy_income_calc
  from limit_cust_level
  group by new_adj_proxy_income_rule
)
select *
from 
-- cust_limit_grp
-- raw_3
comb_band_4
-- limit 100
'''
	New_table_New_Risk_level = client_con.query(sql).to_dataframe()
	return New_table_New_Risk_level

def limit_V2_porto_DPD(client_con):
	
	##DECLARE date1 STRING DEFAULT '2025-07-31';

	sql = '''
With limit_base as (
  select 
      t24_customer_id,
      cast(mis_date as string) as mis_date,
      limit_id,
      cast(limit_amount as numeric) as limit_amt,
      date(datetime(created_at)) as create_dt,
      customer_info_id
  from joey-bi-prod-project.staging.stg_loan__personal_loan_sys_limit_application
  where mis_date=last_day(mis_date)
    and status = "agreementSigned"
),
funded_detail as (
  select t24_customer_id,
        max(ever_funded) as ever_funded
  from joey-bi-ss-risk-fraud-project.credit.funded_info_2025_04
  group by t24_customer_id
),
repay_data as (
  select a.arrangement_id,
          cast(a.mis_date as string) as mis_date,
          sum(a.owed_amount) as sum_owed,
          max(a.dpd) as DPD_acct,
          b.limit_id
  from joey-bi-prod-project.staging.stg_collection__repayment_schedule a
  left join joey-bi-prod-project.staging.stg_collection__limit_loan b
        on a.arrangement_id = b.arrangement_id
        and String(b.mis_date) = date1
  where a.mis_date = last_day(a.mis_date)
  group by a.arrangement_id, a.mis_date, b.limit_id
),
repay_data_cust as (
  select limit_id,
          mis_date,
          sum(sum_owed) as os_client,
          max(DPD_acct) as DPD_client
  from repay_data
  group by limit_id, mis_date
),
bur_detail as (
    SELECT customer_id,
          count_num,
          provider_code_desc,
          contract_phase_desc,
          contract_type,
          contract_type_desc,
          start_date,
          due_date,
          past_due_status_code as latest_kol,
          past_due_status_desc,
          debit_bal as latest_os,
          dpd as latest_dpd,
          max_dpd,
          worst_status,
          worst_status_desc,
          interest_rate,
          credit_limit,
          collaterals_num,
          (cast(substr(due_date,1,4) as int)*12 + cast(substr(due_date,6,2) as int))
          - (cast(substr(start_date,1,4) as int)*12 + cast(substr(start_date,6,2) as int)
          ) as tenor,
          init_credit_lim,
          case
            when interest_rate is null
                or interest_rate = 0 then 0.0001
            else interest_rate end as int_cast

  FROM joey-bi-prod-project.staging.stg_credit_report__clik_credit_granted__hive_merge
  where string(mis_date) = date1
),
bur_detail_client_list as(
  select customer_id,
        max(count_num) as count_num
  from bur_detail
  group by customer_id 
),
bur_detail_client_info as (
  select customer_id,
        count_num,
        max(case
          when contract_type_desc = 'Credit Card' then credit_limit
          else 0 end) as max_CC_limit,
        sum(case
          when contract_type_desc = 'Credit Card' then credit_limit
          else 0 end) as sum_CC_limit,
        sum(case
          when contract_type_desc = 'Credit Card' then latest_os
          else 0 end) as sum_CC_OS,

        max(collaterals_num) as collateral_flag,
        max(case
          when collaterals_num >0 then latest_os
          else 0 end) as max_collateral_loan_OS,
        max(case
          when collaterals_num >0 then 
                  init_credit_lim / (power(1+int_cast/100/12,tenor)-1) * 
                    (int_cast/100/12*power(1+int_cast/100/12,tenor))
          else 0 end) as max_collateral_loan_instalment,
        
        sum(case
          when contract_type_desc in ('Others','Loans provided')
              and collaterals_num = 0 then latest_os
            else 0 end) as sum_kta_loan_OS,
        max(case
          when contract_type_desc in ('Others','Loans provided')
              and collaterals_num = 0 then interest_rate
            else 0 end) as max_kta_int_rate,
        
        sum(case
          when contract_type in ('10','20','P01','P10')
              and collaterals_num = 0 then latest_os
              else 0 end) as sum_fintech_loan_OS,
        max(case
          when contract_type in ('10','20','P01','P10')
              and collaterals_num = 0 then interest_rate
              else 0 end) as max_fintech_int_rate
        
  from bur_detail
  where contract_phase_desc = 'Active'
  group by customer_id, count_num
),
cred_summary as (
  select application_id,
          bureau_type as clik_bureau_info,
          income_verified as izi_income_verification,
          final_income,
          monthly_debt_repayment,
          final_dbr,
          internal_exposure,
          total_exposure,
          internal_exposure_cap,
          total_exposure_cap,
          exposure_dbr
  from joey-bi-prod-project.staging.stg_loan__loan_credit_summary
  where STRING(mis_date)=date1
),
combine as (
  select a.*,
          d.ever_funded,
          e.os_client,
          e.DPD_client,
          case 
            when f.DPD_client is null then 0
            else f.DPD_client end as DPD_prev_mth,
          g.count_num,
          h.clik_bureau_info,
          h.izi_income_verification,
          h.final_income,
          h.monthly_debt_repayment,
          h.final_dbr,
          h.internal_exposure,
          h.total_exposure,
          h.internal_exposure_cap,
          h.total_exposure_cap,
          h.exposure_dbr,
          M.DPD_client as dpd_mob_2
  from limit_base a
  left join funded_detail d on a.t24_customer_id = cast(d.t24_customer_id as string) 
  left join repay_data_cust e on a.limit_id = e.limit_id
                            and a.mis_date = e.mis_date
  left join repay_data_cust f on e.limit_id = f.limit_id
      and cast(substr(e.mis_date,1,4) as int)*12 + cast(substr(e.mis_date,6,2) as int)
          = cast(substr(f.mis_date,1,4) as int)*12 + cast(substr(f.mis_date,6,2) as int) +1
  left join bur_detail_client_list g on a.customer_info_id = g.customer_id 
  left join cred_summary h on a.limit_id = h.application_id 
  left join repay_data_cust M on e.limit_id = M.limit_id
      and cast(substr(e.mis_date,1,4) as int)*12 + cast(substr(e.mis_date,6,2) as int)
          = cast(substr(M.mis_date,1,4) as int)*12 + cast(substr(M.mis_date,6,2) as int) -2
),
          -- DBR x proxy_income x bad rate coba cek, kira2 DBR bisa di fine tune kemana....

combine_2 as (
  select a.*,
        b.max_CC_limit,
        b.sum_CC_limit,
        b.sum_CC_OS,

        b.collateral_flag,
        b.max_collateral_loan_OS,
        b.max_collateral_loan_instalment,
        
        b.sum_kta_loan_OS,
        b.max_kta_int_rate,
        
        b.sum_fintech_loan_OS,
        b.max_fintech_int_rate
  from combine a
  left join bur_detail_client_info b on a.customer_info_id = b.customer_id
          and a.count_num = b.count_num 
),
comb_band as (
  select *,
        case
          when DPD_client is null then '1. Current'
          when DPD_client = 0 then '1. Current'
          when DPD_client <= 30 then '2. DPD 1-30'
          when DPD_client <= 60 then '3. DPD 31-60'
          when DPD_client <= 90 then '4. DPD 61-90'
          when DPD_client <= 120 then '5. DPD 91-120'
          when DPD_client <= 150 then '6. DPD 121-150'
          when DPD_client <= 180 then '7. DPD 151-180'
          when DPD_client > 180 then '8. DPD 181 above'
          else '9. else' end as DPD_band,
        case
          when DPD_prev_mth = 0 then 'Yes'
          when DPD_prev_mth >=  DPD_client then 'No'
          else 'Yes' end as Forward_looking_FR,
        case
          when OS_client/limit_amt > 0.5 then OS_client
          else limit_amt *0.5 end as OS_scenario50,
        case
          when OS_client/limit_amt > 0.7 then OS_client
          else limit_amt *0.7 end as OS_scenario70,
        case
          when max_CC_limit > limit_amt then '01. Have CC with higher limit'
          when sum_kta_loan_OS > limit_amt then '02. Have KTA with higher limit'
          when sum_fintech_loan_OS > limit_amt then '03. Have fintech with higher limit'
          when max_kta_int_rate <= 1.99 then '04. Have KTA with lower interest'
          when max_fintech_int_rate <= 1.99 then '05. Have fintech with lower interest'
          else '99. Other reason' end as Reason_not_used,
        case
          when OS_client/limit_amt <= 0.5 or
               OS_client is null then 'Not Used'
          else 'Used' end as Used_Flag,
        case
          when max_CC_limit > 100000000 then '01. > 100 mio'
          when max_CC_limit >  70000000 then '02. 70 - 100 mio'
          when max_CC_limit >  50000000 then '03. 50 - 70 mio'
          when max_CC_limit >  30000000 then '04. 30 - 50 mio'
          when max_CC_limit >  20000000 then '05. 20 - 30 mio'
          when max_CC_limit >  10000000 then '06. 10 - 20 mio'
          when max_CC_limit >   5000000 then '07.  5 - 10 mio'
          when max_CC_limit >         0 then '08.  1 -  5 mio'
          else '09. Else' end as CC_limit_grp,
        case
          when sum_kta_loan_OS > 100000000 then '01. > 100 mio'
          when sum_kta_loan_OS >  70000000 then '02. 70 - 100 mio'
          when sum_kta_loan_OS >  50000000 then '03. 50 - 70 mio'
          when sum_kta_loan_OS >  30000000 then '04. 30 - 50 mio'
          when sum_kta_loan_OS >  20000000 then '05. 20 - 30 mio'
          when sum_kta_loan_OS >  10000000 then '06. 10 - 20 mio'
          when sum_kta_loan_OS >   5000000 then '07.  5 - 10 mio'
          when sum_kta_loan_OS >         0 then '08.  1 -  5 mio'
          else '09. Else' end as KTA_loan_grp,
        case
          when sum_fintech_loan_OS > 100000000 then '01. > 100 mio'
          when sum_fintech_loan_OS >  70000000 then '02. 70 - 100 mio'
          when sum_fintech_loan_OS >  50000000 then '03. 50 - 70 mio'
          when sum_fintech_loan_OS >  30000000 then '04. 30 - 50 mio'
          when sum_fintech_loan_OS >  20000000 then '05. 20 - 30 mio'
          when sum_fintech_loan_OS >  10000000 then '06. 10 - 20 mio'
          when sum_fintech_loan_OS >   5000000 then '07.  5 - 10 mio'
          when sum_fintech_loan_OS >         0 then '08.  1 -  5 mio'
          else '09. Else' end as fintech_loan_grp,
        case
          when max_kta_int_rate > 1.99 then '01. > 1.99%'
          when max_kta_int_rate > 1.70 then '02. 1.70 - 1.99%'
          when max_kta_int_rate > 1.50 then '03. 1.50 - 1.70%'
          when max_kta_int_rate > 1.30 then '04. 1.30 - 1.50%'
          when max_kta_int_rate > 1.10 then '05. 1.10 - 1.30%'
          when max_kta_int_rate > 0.90 then '06. 0.90 - 1.10%'
          when max_kta_int_rate > 0.70 then '07. 0.70 - 0.90%'
          when max_kta_int_rate > 0.50 then '08. 0.50 - 0.70%'
          when max_kta_int_rate > 0.25 then '09. 0.25 - 0.50%'
          when max_kta_int_rate > 0 then '10. 0.01 - 0.25%'
          when max_kta_int_rate = 0 then '11. 0%'
          else '12. Else' end as kta_int_grp,
        case
          when max_fintech_int_rate > 1.99 then '01. > 1.99%'
          when max_fintech_int_rate > 1.70 then '02. 1.70 - 1.99%'
          when max_fintech_int_rate > 1.50 then '03. 1.50 - 1.70%'
          when max_fintech_int_rate > 1.30 then '04. 1.30 - 1.50%'
          when max_fintech_int_rate > 1.10 then '05. 1.10 - 1.30%'
          when max_fintech_int_rate > 0.90 then '06. 0.90 - 1.10%'
          when max_fintech_int_rate > 0.70 then '07. 0.70 - 0.90%'
          when max_fintech_int_rate > 0.50 then '08. 0.50 - 0.70%'
          when max_fintech_int_rate > 0.25 then '09. 0.25 - 0.50%'
          when max_fintech_int_rate > 0 then '10. 0.01 - 0.25%'
          when max_fintech_int_rate = 0 then '11. 0%'
          else '12. Else' end as fintech_int_grp,
        case 
          when dpd_mob_2 >= 14 then OS_client
          else 0 end as FPD14,
        case 
          when dpd_mob_2 >= 30 then OS_client
          else 0 end as SPD30,
        case
          when final_dbr = 0 then '01. DBR 0%'
          when final_dbr <= 20 then '02. DBR 1 - 20%'
          when final_dbr <= 40 then '03. DBR 21 - 40%'
          when final_dbr <= 60 then '04. DBR 41 - 60%'
          when final_dbr <= 80 then '05. DBR 61 - 80%'
          when final_dbr <= 100 then '06. DBR 81 - 100%'
          when final_dbr > 100 then '07. DBR > 100%'
          else '99. Else' end as DBR_perc
  from combine_2
  where t24_customer_id not in ('101174407','101858872','102072000',
          '102084939','102072802','102096422') -- fraud
),
raw_1 as (
  select 
      mis_date,
      ever_funded,
      DPD_band,
      Forward_looking_FR,
      sum(OS_client) as total_os,
      sum(OS_scenario50) as total_os50,
      sum(OS_scenario70) as total_os70,
      count(distinct t24_customer_id) as count_client,
      sum(limit_amt) as total_limit
      
from comb_band 
group by 
      mis_date,
      ever_funded,
      DPD_band,
      Forward_looking_FR
),
raw_2 as (
  select 
      mis_date,
      ever_funded,
      Reason_not_used,
      Used_Flag,
      clik_bureau_info,
      CC_limit_grp,
      KTA_loan_grp,
      fintech_loan_grp,
      kta_int_grp,
      fintech_int_grp,
      count(distinct t24_customer_id) as count_client
      
from comb_band 
where mis_date = '2025-05-31'
group by 
      mis_date,
      ever_funded,
      Reason_not_used,
      Used_Flag,
      clik_bureau_info,
      CC_limit_grp,
      KTA_loan_grp,
      fintech_loan_grp,
      kta_int_grp,
      fintech_int_grp
),
raw_3 as (
  select 
      mis_date,
      ever_funded,
      -- DPD_band,
      -- Forward_looking_FR,
      DBR_perc,
      sum(OS_client) as total_os,
      sum(FPD14) as FPD14,
      sum(SPD30) as SPD30,
      
from comb_band 
group by 
      mis_date,
      ever_funded,
      -- DPD_band,
      -- Forward_looking_FR,
      DBR_perc
)
select *
from raw_3
'''
	limit_V2_porto_DPD = client_con.query(sql).to_dataframe()
	return limit_V2_porto_DPD

