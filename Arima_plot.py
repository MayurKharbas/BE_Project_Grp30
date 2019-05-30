#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

colnames = ['activity_hour', 'total_activity']
forecast = pd.read_csv('forecast.csv', names=colnames, header=None)

# Generate the plot and save as png
ax = train.plot(x='activity_hour', y='total_activity', label='train')
test.plot(ax=ax, x='activity_hour', y='total_activity', label='test')
forecast.plot(ax=ax, x='activity_hour', y='total_activity', label='model')
plt.savefig('plot.png')

# Uncomment below line to show plot
# plt.show()

# mse = abs((forecast.total_activity - test.total_activity))/test.total_activity
# mean_error = np.mean(mse)
# print(mean_error)
# below10 = mse[mse<0.].count()
# all = mse.count()
# print(below10, " : ",  all)
# print(mse)



