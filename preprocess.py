import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.DataFrame({})
for i in range(1,8):
    df_new = pd.read_csv('sms-call-internet-mi-2013-11-0{}.csv'.format(i), parse_dates=['activity_date'])
    df = df.append(df_new)
    print("File " + str(i) + " added")


df['activity_hour'] += 24*(df.activity_date.dt.day-1)


#Write the df to csv file
#df.to_csv('new_pro2.csv', index=False, encoding='utf-8')


#Plot the data
ax = df[df.square_id==147].plot(x='activity_hour', y='total_activity', label='GRID 147')
df[df.square_id==2].plot(ax=ax, x='activity_hour', y='total_activity', label='GRID 2')
plt.xlabel("Hours")
plt.ylabel("Total Activity")
plt.show()


'''
square_id	activity_date	activity_hour	total_activity
1			2013-11-01			0			2.3183586492
1			2013-11-01			1			0.879857571
1			2013-11-01			2			2.0757160953
1			2013-11-01			3			0.8971439431
1			2013-11-01			4			0.5743615238


sklearn - models
rolling_means
'''