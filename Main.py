#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import itertools
from itertools import product
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose


def Process(grid_id):

	df = pd.DataFrame({})
	path = "/media/mayur/Softwares/BE Project/ProcessedDataset/November/"
	for i in range(1,8):
		df_new = pd.read_csv(path+'sms-call-internet-mi-2013-11-0{}.csv'.format(i),parse_dates=['activity_date'])
		df = df.append(df_new)
		print("File " + str(i) + " added")


	df['activity_hour'] += 24*(df.activity_date.dt.day-1)


	# ## Series transformation

	df_grid = df[df['square_id']==grid_id]
	#df_grid.set_index('activity_hour', inplace=True) 
	df_grid.drop(['square_id', 'activity_date'], axis=1, inplace=True)
	#df_grid.to_csv('ts-grid-147.csv', index=False, encoding='utf-8')


	# # Split dataset into train and test

	train = df_grid[:125]
	test = df_grid[125:]


	train = train.set_index('activity_hour')    #Run this line once
	test = test.set_index('activity_hour')


	# # Fit Arima model


	'''
	ARIMA model

	parameters_list - list with (p, q, P, Q)
	    p - associated with the auto-regressive aspect of the model
	    d - integration order in ARIMA model (effects the amount of differencing to apply to a time series)
	    D - seasonal integration order 
	    
	'''

	p = d = q = range(0, 2)
	pdq = list(itertools.product(p, d, q))
	seasonal_pdq = [(x[0], x[1], x[2], 24) for x in pdq]


	# AIC Scores
	# Akaike information criterion (AIC) (Akaike, 1974) 
	# is a fined technique based on in-sample fit 
	# to estimate the likelihood of a model to predict/estimate the future values. 
	# A good model is the one that has minimum AIC among all the other models.

	for param in pdq:
		for param_seasonal in seasonal_pdq:
			try:
				mod = sm.tsa.statespace.SARIMAX(train,order=param,
								seasonal_order=param_seasonal,
								enforce_stationarity=False,
								enforce_invertibility=False)

				results = mod.fit()

				print('ARIMA{}x{}24 - AIC:{}'.format(param, param_seasonal, results.aic))

			except:
				continue


	mod = sm.tsa.statespace.SARIMAX(train,
		                        order=(1, 1, 1),
		                        seasonal_order=(0, 1, 1, 24),
		                        enforce_stationarity=False,
		                        enforce_invertibility=False)
	results = mod.fit()

	# setting initial values and some bounds for them
	#ps = qs = range(2, 5)
	ps = qs = Ps = Qs = range(0, 2)
	d = 1
	D = 1
	s = 24 # season length is 24

	# creating list with all the possible combinations of parameters
	parameters = product(ps, qs, Ps, Qs)
	parameters_list = list(parameters)
	len(parameters_list)



	'''
	def optimizeSARIMA(parameters_list, d, D, s):
	    """
		Return df with parameters and corresponding AIC
		
		parameters_list - (p, q, P, Q) tuples
		d - integration order in ARIMA model
		D - seasonal integration order 
		s - length of season
	    """
	    
	    results = []
	    best_aic = float("inf")

	    for param in parameters_list:
		# we need try-except because on some combinations model fails to converge
		try:
		    model=sm.tsa.statespace.SARIMAX(train, order=(param[0], d, param[1]), 
		                                    seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
		except:
		    continue
		aic = model.aic
		# saving best model, AIC and parameters
		if aic < best_aic:
		    best_model = model
		    best_aic = aic
		    best_param = param
		results.append([param, model.aic])

	    result_table = pd.DataFrame(results)
	    result_table.columns = ['parameters', 'aic']
	    # sorting in ascending order, the lower AIC is - the better
	    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)
	    
	    return result_table
	    
	result_table = optimizeSARIMA(parameters_list, d, D, s)
	print(result_table)'''



	best_model = sm.tsa.statespace.SARIMAX(train,
		                        order=(1, 1, 1),
		                        seasonal_order=(0, 1, 1, 24),
		                        enforce_stationarity=False,
		                        enforce_invertibility=False).fit()


	#Predict list

	n = 41
	data = train.copy()
	data['arima_model'] = best_model.fittedvalues

	forecast = pd.DataFrame(best_model.predict(start=data.shape[0], end=data.shape[0]+n))
	
	##forecast.to_csv('forecast1.csv', index='activity_hour', encoding='utf-8',header=False)
	#forecast.to_csv('forecast1.csv', encoding='utf-8',header=False)
	#colnames = ['activity_hour', 'total_activity']
	#forecast = pd.read_csv('forecast1.csv', names=colnames, header=None)    
	#forecast = forecast.set_index('activity_hour')

	#train.to_csv('train.csv', index='activity_hour', encoding='utf-8')
	#test.to_csv('test.csv', index='activity_hour', encoding='utf-8')  

	final_lists = [train, test, forecast]

	return final_lists

if __name__ == "__main__":
	'''
	grid = 325
	out = Process(grid)
	
	train1 = out[0]
	test1 = out[1]
	forecast1 = out[2]
	
	print(train1.head())
	'''
