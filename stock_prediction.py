import pandas as pd
import quandl, math, datetime
import numpy
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
from pylab import rcParams


#%%
mydata = quandl.get("WIKI/TSLA")
print(mydata.head())

#%%
#get rows required for feature
mydata = mydata[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
#create High-Low Percent / Stock / Dday
mydata['HL_PCT'] = (mydata['Adj. High'] - mydata['Adj. Low']) / mydata['Adj. Low'] * 100.00
#create final percent change total day
mydata['PCT_change'] = (mydata['Adj. Close'] - mydata['Adj. Open']) / mydata['Adj. Open'] * 100.00
#rebuild data-frame with useful information
mydata = mydata[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']] 

print(mydata.tail()) 

#%%
forecast_col = 'Adj. Close'

mydata.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.08*len(mydata)))
print(forecast_out)
mydata['label'] = mydata[forecast_col].shift(-forecast_out) 


#%%
X = numpy.array(mydata.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately=X[-forecast_out:]
X = X[:-forecast_out:]

mydata.dropna(inplace=True)
y = numpy.array(mydata['label'])


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression ()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)
mydata['Forecast'] = numpy.nan

#%%
mydata['Forecast'] = numpy.nan
last_date = mydata.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    mydata.loc[next_date] = [numpy.nan for _ in range(len(mydata.columns)-1)] + [i]

mydata['Adj. Close'].plot()
mydata['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.figure(figsize=(3,4))
rcParams["figure.dpi"] = 300
rcParams['figure.figsize'] = 9, 6
plt.show()







