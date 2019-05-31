import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from math import sqrt
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import ExponentialSmoothing, Holt
import statsmodels.api as sm

import warnings
warnings.filterwarnings('ignore')

def Process_Holt(grid_id):
    df = pd.DataFrame({})
    path = '/media/mayur/Softwares/BE_Project_Grp30/Processed datasets/'
    for i in range(1,22):
        if i<10:
            df_new = pd.read_csv(path+'sms-call-internet-mi-2013-11-0{}.csv'.format(i), parse_dates=['activity_date'])
        else:
            df_new = pd.read_csv(path+'sms-call-internet-mi-2013-11-{}.csv'.format(i), parse_dates=['activity_date'])
        df = df.append(df_new)
        print("File " + str(i) + " added")
        
    df['activity_hour'] += 24*(df.activity_date.dt.day-1)

    series = df[df['square_id']==147]
    series.set_index('activity_hour', inplace=True) 
    series.drop(['square_id', 'activity_date'], axis=1, inplace=True)

    train = series[:380]
    test = series[380:]


    # Find seasonal components (errors to be removed)
    # sm.tsa.seasonal_decompose(train.total_activity).plot()
    # result = sm.tsa.stattools.adfuller(train.total_activity)
    # plt.show()

    # predict the data
    forecast = test.copy()
    model = ExponentialSmoothing(np.asarray(train['total_activity']) ,seasonal_periods=24 ,trend='add', seasonal='add',).fit()
    forecast['Holt_Winter'] = model.forecast(len(test))

    # plot the data
    plt.plot( train['total_activity'], label='Train')
    plt.plot(test['total_activity'], label='Test')
    plt.plot(forecast['Holt_Winter'], label='Holt_Winter')
    plt.legend(loc='best')
    plt.xlabel('Hours')
    plt.ylabel('Total_activity')
    plt.title('Arima model - Grid {}'.format(grid_id))
    plt.savefig('static/images/Holt_{}.png'.format(grid_id))

    #Calculate errors
    rmse = sqrt(mean_squared_error(test.total_activity, forecast.Holt_Winter))
    mse = abs((forecast.Holt_Winter - test.total_activity))/test.total_activity
    mean_error = np.mean(mse)
    below30 = mse[mse<0.3].count()
    all = mse.count()

    accuracy = 100 - rmse
    # accuracy = (1-mean_error)*100
    # accuracy = (below30/all)*100

    result = [
            {
                'rmse' : rmse,
                'mean_error' : mean_error,
                'accuracy' : accuracy,
                'below30' : below30,
                'all' : all
            }
        ]

    return result

out = Process_Holt(342)
print(out)

