import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
#from pyramid.arima import auto_arima

df = pd.read_csv('new_pro.csv')

df_grid = df[df.square_id==147]
#df_grid.index = df_grid['activity_hour'] 
df_grid.drop(['square_id', 'activity_date'], axis=1, inplace=True)
#df_grid.to_csv('ts-grid-147.csv', index=False, encoding='utf-8')

#Plot the data
ax = df_grid.plot(x='activity_hour', y='total_activity', label='GRID 147')
plt.xlabel('Hours')
plt.ylabel('Total_activity')
plt.title('')

#Split dataset into train(75) and test(25) 
train = df_grid[:125]
print('Train')
print(train.head(5))

test = df_grid[125:]
print('Test')
print(test.head(5))

#Plot train-test data 
ax = train.plot(x='activity_hour', y='total_activity', label='Train')
test.plot(ax=ax, x='activity_hour', y='total_activity', label='Test')
plt.xlabel('Hours')
plt.ylabel('Total_activity')
plt.show()



'''
ARIMA model

parameters_list - list with (p, q, P, Q)
    p - associated with the auto-regressive aspect of the model
    d - integration order in ARIMA model (effects the amount of differencing to apply to a time series)
    D - seasonal integration order 
    s - length of season

trend, seasonality, and noise

statsmodels.api as sm
'''

#Decompose the series
result = seasonal_decompose(df_grid, model='multiplicative')	#try 'additive'
result.plot()
plt.show()

'''
#Fit Arima model
stepwise_model = auto_arima(df_grid, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
print(stepwise_model.aic())
'''







