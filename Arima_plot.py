#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt


train = pd.read_csv('train.csv')
train.head()

test = pd.read_csv('test.csv')
test.head()

colnames = ['activity_hour', 'total_activity']
forecast = pd.read_csv('forecast1.csv', names=colnames, header=None)
forecast.head()

ax = train.plot(x='activity_hour', y='total_activity', label='train')
test.plot(ax=ax, x='activity_hour', y='total_activity', label='test')
forecast.plot(ax=ax, x='activity_hour', y='total_activity', label='model')
plt.figure(figsize=(50, 30))
plt.show()


mse = abs((forecast.total_activity - test.total_activity))/test.total_activity
mean_error = np.mean(mse)
print(mean_error)

below10 = mse[mse<0.].count()
all = mse.count()

print(below10, " : ",  all)
print(mse)



