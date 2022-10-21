---
layout: post
title: WTI Futures Trading With War Sentiment
subtitle: Using war sentiment index and rolling regression analysis to predict WTI futures price
tags: [math, tech]
---

### It can be theorised that WTI price is equal to some function f(USD, inflation, storage). So let's first perform a rolling regression on these factors from in order to choose the optimal parameters:


```python
#Importing all necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import warnings
warnings.filterwarnings('ignore')
```

### WTI 1 month future prices 2010-2019:


```python
data_prices_crude         = pd.read_excel('PET_PRI_FUT_S1_D.xls', sheet_name = 'Data 1', skiprows = 2)
data_prices_crude.columns = ['Date', 'F1', 'F2', 'F3', 'F4']
data_prices_crude         = data_prices_crude[(data_prices_crude['Date'] >= '2009-08-07') & (data_prices_crude['Date'] <= '2021-12-31')]
data_prices_crude         = data_prices_crude.set_index('Date')
data_prices_crude.drop(columns=['F2','F3','F4'],inplace=True)
data_prices_crude.head(100)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>F1</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2009-08-07</th>
      <td>70.93</td>
    </tr>
    <tr>
      <th>2009-08-10</th>
      <td>70.60</td>
    </tr>
    <tr>
      <th>2009-08-11</th>
      <td>69.45</td>
    </tr>
    <tr>
      <th>2009-08-12</th>
      <td>70.16</td>
    </tr>
    <tr>
      <th>2009-08-13</th>
      <td>70.52</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2009-12-22</th>
      <td>74.40</td>
    </tr>
    <tr>
      <th>2009-12-23</th>
      <td>76.67</td>
    </tr>
    <tr>
      <th>2009-12-24</th>
      <td>78.05</td>
    </tr>
    <tr>
      <th>2009-12-28</th>
      <td>78.77</td>
    </tr>
    <tr>
      <th>2009-12-29</th>
      <td>78.87</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 1 columns</p>
</div>



### 10 year breakeven inflation rate 2010-2019:


```python
inflation_rate         = pd.read_excel('T10YIE.xls', skiprows = 10)
inflation_rate.columns = ['Date','Inflation Rate']
inflation_rate         = inflation_rate[(inflation_rate['Date'] >= '2009-08-12') & (inflation_rate['Date'] <= '2021-12-31')]
inflation_rate         = inflation_rate.set_index('Date')
inflation_rate.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Inflation Rate</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2009-08-12</th>
      <td>1.88</td>
    </tr>
    <tr>
      <th>2009-08-13</th>
      <td>1.79</td>
    </tr>
    <tr>
      <th>2009-08-14</th>
      <td>1.70</td>
    </tr>
    <tr>
      <th>2009-08-17</th>
      <td>1.69</td>
    </tr>
    <tr>
      <th>2009-08-18</th>
      <td>1.73</td>
    </tr>
  </tbody>
</table>
</div>



### Nominal USD Index 2010-2019:


```python
exchange_rate         = pd.read_excel('DTWEXBGS.xls', skiprows = 10)
exchange_rate.columns = ['Date','USD Index']
exchange_rate         = exchange_rate[(exchange_rate['Date'] >= '2009-08-07') & (exchange_rate['Date'] <= '2021-12-31')]
exchange_rate         = exchange_rate.set_index('Date')
exchange_rate.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>USD Index</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2009-08-07</th>
      <td>94.4692</td>
    </tr>
    <tr>
      <th>2009-08-10</th>
      <td>94.6087</td>
    </tr>
    <tr>
      <th>2009-08-11</th>
      <td>94.8678</td>
    </tr>
    <tr>
      <th>2009-08-12</th>
      <td>94.5394</td>
    </tr>
    <tr>
      <th>2009-08-13</th>
      <td>94.2392</td>
    </tr>
  </tbody>
</table>
</div>



### Weekly stock data 2010-2019:


```python
data_stocks               = pd.read_excel('PET_STOC_WSTK_DCU_NUS_W.xls', sheet_name = 'Data 1', skiprows = 2)
data_stocks               = data_stocks[['Date', 'Stocks']]
data_stocks               = data_stocks[(data_stocks['Date'] >= '2009-08-07') & (data_stocks['Date'] <= '2021-12-31')]
data_stocks         = data_stocks.set_index('Date')
data_stocks.head(100)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Stocks</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2009-08-07</th>
      <td>1058086</td>
    </tr>
    <tr>
      <th>2009-08-14</th>
      <td>1049688</td>
    </tr>
    <tr>
      <th>2009-08-21</th>
      <td>1049816</td>
    </tr>
    <tr>
      <th>2009-08-28</th>
      <td>1049443</td>
    </tr>
    <tr>
      <th>2009-09-04</th>
      <td>1043637</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2011-06-03</th>
      <td>1073439</td>
    </tr>
    <tr>
      <th>2011-06-10</th>
      <td>1070033</td>
    </tr>
    <tr>
      <th>2011-06-17</th>
      <td>1068311</td>
    </tr>
    <tr>
      <th>2011-06-24</th>
      <td>1063936</td>
    </tr>
    <tr>
      <th>2011-07-01</th>
      <td>1063063</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 1 columns</p>
</div>



### Merging the data together:


```python
df1 = pd.merge(data_prices_crude,inflation_rate,how = 'left',on='Date')
df1 = pd.merge(df1,exchange_rate,how = 'left',on='Date')
df1 = pd.merge(df1,data_stocks,how = 'left',on='Date')
df1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>F1</th>
      <th>Inflation Rate</th>
      <th>USD Index</th>
      <th>Stocks</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2009-08-07</th>
      <td>70.93</td>
      <td>NaN</td>
      <td>94.4692</td>
      <td>1058086.0</td>
    </tr>
    <tr>
      <th>2009-08-10</th>
      <td>70.60</td>
      <td>NaN</td>
      <td>94.6087</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2009-08-11</th>
      <td>69.45</td>
      <td>NaN</td>
      <td>94.8678</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2009-08-12</th>
      <td>70.16</td>
      <td>1.88</td>
      <td>94.5394</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2009-08-13</th>
      <td>70.52</td>
      <td>1.79</td>
      <td>94.2392</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### Stock data is weekly so we must use interpolation:


```python
df1['Stocks'].interpolate(inplace=True)
df1 = df1.loc['2009-08-12':]
```


```python
plt.plot(df1['Stocks'])
```




    [<matplotlib.lines.Line2D at 0x7ff9dd935d30>]




![png](https://github.com/arjunkalsi/arjunkalsi.github.io/tree/master/img/wti/output_14_1.png)



```python
import pandas_datareader as pdr
import seaborn
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

seaborn.set_style("darkgrid")
pd.plotting.register_matplotlib_converters()
%matplotlib inline
```

### Getting optimal coefficients:


```python
w = 60
endog = df1['F1']
exog = sm.add_constant(df1[['Inflation Rate','USD Index','Stocks']])
rols = RollingOLS(endog, exog, window=w)
rres = rols.fit()
params = rres.params.copy()
params.index = np.arange(1, params.shape[0] + 1)
params
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>const</th>
      <th>Inflation Rate</th>
      <th>USD Index</th>
      <th>Stocks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3118</th>
      <td>-228.150461</td>
      <td>20.315538</td>
      <td>-0.470935</td>
      <td>0.000297</td>
    </tr>
    <tr>
      <th>3119</th>
      <td>-187.728962</td>
      <td>21.927039</td>
      <td>-0.507664</td>
      <td>0.000258</td>
    </tr>
    <tr>
      <th>3120</th>
      <td>-150.824993</td>
      <td>23.830340</td>
      <td>-0.550962</td>
      <td>0.000223</td>
    </tr>
    <tr>
      <th>3121</th>
      <td>-117.391030</td>
      <td>26.261025</td>
      <td>-0.606247</td>
      <td>0.000191</td>
    </tr>
    <tr>
      <th>3122</th>
      <td>-279.821665</td>
      <td>3.975534</td>
      <td>-0.088994</td>
      <td>0.000345</td>
    </tr>
  </tbody>
</table>
<p>3122 rows × 4 columns</p>
</div>




```python
params['Date'] = df1.index
params = params.set_index('Date')
params.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>const</th>
      <th>Inflation Rate</th>
      <th>USD Index</th>
      <th>Stocks</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-12-27</th>
      <td>-228.150461</td>
      <td>20.315538</td>
      <td>-0.470935</td>
      <td>0.000297</td>
    </tr>
    <tr>
      <th>2021-12-28</th>
      <td>-187.728962</td>
      <td>21.927039</td>
      <td>-0.507664</td>
      <td>0.000258</td>
    </tr>
    <tr>
      <th>2021-12-29</th>
      <td>-150.824993</td>
      <td>23.830340</td>
      <td>-0.550962</td>
      <td>0.000223</td>
    </tr>
    <tr>
      <th>2021-12-30</th>
      <td>-117.391030</td>
      <td>26.261025</td>
      <td>-0.606247</td>
      <td>0.000191</td>
    </tr>
    <tr>
      <th>2021-12-31</th>
      <td>-279.821665</td>
      <td>3.975534</td>
      <td>-0.088994</td>
      <td>0.000345</td>
    </tr>
  </tbody>
</table>
</div>



### Merging with WTI dataframe:


```python
df1_reg = pd.merge(df1,params,how = 'left',on='Date')
```


```python
df1_regrange = df1_reg.loc['2010-01-04':]
df1_regrange
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>F1</th>
      <th>Inflation Rate_x</th>
      <th>USD Index_x</th>
      <th>Stocks_x</th>
      <th>const</th>
      <th>Inflation Rate_y</th>
      <th>USD Index_y</th>
      <th>Stocks_y</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010-01-04</th>
      <td>81.51</td>
      <td>2.38</td>
      <td>92.3566</td>
      <td>1.036539e+06</td>
      <td>-223.685957</td>
      <td>1.230996</td>
      <td>-0.002131</td>
      <td>0.000286</td>
    </tr>
    <tr>
      <th>2010-01-05</th>
      <td>81.77</td>
      <td>2.34</td>
      <td>92.2236</td>
      <td>1.036640e+06</td>
      <td>-164.148619</td>
      <td>0.685651</td>
      <td>0.008764</td>
      <td>0.000229</td>
    </tr>
    <tr>
      <th>2010-01-06</th>
      <td>83.18</td>
      <td>2.37</td>
      <td>92.0941</td>
      <td>1.036740e+06</td>
      <td>-109.632478</td>
      <td>0.285432</td>
      <td>0.017643</td>
      <td>0.000177</td>
    </tr>
    <tr>
      <th>2010-01-07</th>
      <td>82.66</td>
      <td>2.41</td>
      <td>92.3684</td>
      <td>1.036841e+06</td>
      <td>-49.771309</td>
      <td>-0.418137</td>
      <td>0.016535</td>
      <td>0.000121</td>
    </tr>
    <tr>
      <th>2010-01-08</th>
      <td>82.75</td>
      <td>2.42</td>
      <td>92.1485</td>
      <td>1.036941e+06</td>
      <td>-20.414187</td>
      <td>-0.484475</td>
      <td>0.019674</td>
      <td>0.000093</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2021-12-27</th>
      <td>75.57</td>
      <td>2.50</td>
      <td>115.4964</td>
      <td>1.015275e+06</td>
      <td>-228.150461</td>
      <td>20.315538</td>
      <td>-0.470935</td>
      <td>0.000297</td>
    </tr>
    <tr>
      <th>2021-12-28</th>
      <td>75.98</td>
      <td>2.50</td>
      <td>115.4497</td>
      <td>1.014339e+06</td>
      <td>-187.728962</td>
      <td>21.927039</td>
      <td>-0.507664</td>
      <td>0.000258</td>
    </tr>
    <tr>
      <th>2021-12-29</th>
      <td>76.56</td>
      <td>2.53</td>
      <td>115.3964</td>
      <td>1.013404e+06</td>
      <td>-150.824993</td>
      <td>23.830340</td>
      <td>-0.550962</td>
      <td>0.000223</td>
    </tr>
    <tr>
      <th>2021-12-30</th>
      <td>76.99</td>
      <td>2.58</td>
      <td>115.3163</td>
      <td>1.012468e+06</td>
      <td>-117.391030</td>
      <td>26.261025</td>
      <td>-0.606247</td>
      <td>0.000191</td>
    </tr>
    <tr>
      <th>2021-12-31</th>
      <td>75.21</td>
      <td>2.56</td>
      <td>0.0000</td>
      <td>1.011533e+06</td>
      <td>-279.821665</td>
      <td>3.975534</td>
      <td>-0.088994</td>
      <td>0.000345</td>
    </tr>
  </tbody>
</table>
<p>3023 rows × 8 columns</p>
</div>




```python
df1_regrange['Predicted F1'] = df1_regrange['const'] + (df1_regrange['Inflation Rate_y']*df1_regrange['Inflation Rate_x']) + (df1_regrange['USD Index_y']*df1_regrange['USD Index_x']) + (df1_regrange['Stocks_y']*df1_regrange['Stocks_x'])
df1_regrange

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>F1</th>
      <th>Inflation Rate_x</th>
      <th>USD Index_x</th>
      <th>Stocks_x</th>
      <th>const</th>
      <th>Inflation Rate_y</th>
      <th>USD Index_y</th>
      <th>Stocks_y</th>
      <th>Predicted F1</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010-01-04</th>
      <td>81.51</td>
      <td>2.38</td>
      <td>92.3566</td>
      <td>1.036539e+06</td>
      <td>-223.685957</td>
      <td>1.230996</td>
      <td>-0.002131</td>
      <td>0.000286</td>
      <td>75.292904</td>
    </tr>
    <tr>
      <th>2010-01-05</th>
      <td>81.77</td>
      <td>2.34</td>
      <td>92.2236</td>
      <td>1.036640e+06</td>
      <td>-164.148619</td>
      <td>0.685651</td>
      <td>0.008764</td>
      <td>0.000229</td>
      <td>75.712491</td>
    </tr>
    <tr>
      <th>2010-01-06</th>
      <td>83.18</td>
      <td>2.37</td>
      <td>92.0941</td>
      <td>1.036740e+06</td>
      <td>-109.632478</td>
      <td>0.285432</td>
      <td>0.017643</td>
      <td>0.000177</td>
      <td>76.195043</td>
    </tr>
    <tr>
      <th>2010-01-07</th>
      <td>82.66</td>
      <td>2.41</td>
      <td>92.3684</td>
      <td>1.036841e+06</td>
      <td>-49.771309</td>
      <td>-0.418137</td>
      <td>0.016535</td>
      <td>0.000121</td>
      <td>76.468911</td>
    </tr>
    <tr>
      <th>2010-01-08</th>
      <td>82.75</td>
      <td>2.42</td>
      <td>92.1485</td>
      <td>1.036941e+06</td>
      <td>-20.414187</td>
      <td>-0.484475</td>
      <td>0.019674</td>
      <td>0.000093</td>
      <td>76.778149</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2021-12-27</th>
      <td>75.57</td>
      <td>2.50</td>
      <td>115.4964</td>
      <td>1.015275e+06</td>
      <td>-228.150461</td>
      <td>20.315538</td>
      <td>-0.470935</td>
      <td>0.000297</td>
      <td>69.904710</td>
    </tr>
    <tr>
      <th>2021-12-28</th>
      <td>75.98</td>
      <td>2.50</td>
      <td>115.4497</td>
      <td>1.014339e+06</td>
      <td>-187.728962</td>
      <td>21.927039</td>
      <td>-0.507664</td>
      <td>0.000258</td>
      <td>70.527938</td>
    </tr>
    <tr>
      <th>2021-12-29</th>
      <td>76.56</td>
      <td>2.53</td>
      <td>115.3964</td>
      <td>1.013404e+06</td>
      <td>-150.824993</td>
      <td>23.830340</td>
      <td>-0.550962</td>
      <td>0.000223</td>
      <td>71.812860</td>
    </tr>
    <tr>
      <th>2021-12-30</th>
      <td>76.99</td>
      <td>2.58</td>
      <td>115.3163</td>
      <td>1.012468e+06</td>
      <td>-117.391030</td>
      <td>26.261025</td>
      <td>-0.606247</td>
      <td>0.000191</td>
      <td>73.678898</td>
    </tr>
    <tr>
      <th>2021-12-31</th>
      <td>75.21</td>
      <td>2.56</td>
      <td>0.0000</td>
      <td>1.011533e+06</td>
      <td>-279.821665</td>
      <td>3.975534</td>
      <td>-0.088994</td>
      <td>0.000345</td>
      <td>79.247339</td>
    </tr>
  </tbody>
</table>
<p>3023 rows × 9 columns</p>
</div>



### Plotting F1 against Predicted F1:


```python
plt.figure(figsize=(12,8))
plt.plot(df1_regrange['F1'],label='F1')
plt.plot(df1_regrange['Predicted F1'],label='Predicted F1')
plt.legend()
plt.savefig('F1 versus Predicted F1 (window = 60).png')
plt.show()
```


![png](https://github.com/arjunkalsi/arjunkalsi.github.io/tree/master/img/wti/output_24_0.png)



```python
df1_regrange['F1 - Predicted'] = df1_regrange['F1'] - df1_regrange['Predicted F1']
df1_regrange
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>F1</th>
      <th>Inflation Rate_x</th>
      <th>USD Index_x</th>
      <th>Stocks_x</th>
      <th>const</th>
      <th>Inflation Rate_y</th>
      <th>USD Index_y</th>
      <th>Stocks_y</th>
      <th>Predicted F1</th>
      <th>F1 - Predicted</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010-01-04</th>
      <td>81.51</td>
      <td>2.38</td>
      <td>92.3566</td>
      <td>1.036539e+06</td>
      <td>-223.685957</td>
      <td>1.230996</td>
      <td>-0.002131</td>
      <td>0.000286</td>
      <td>75.292904</td>
      <td>6.217096</td>
    </tr>
    <tr>
      <th>2010-01-05</th>
      <td>81.77</td>
      <td>2.34</td>
      <td>92.2236</td>
      <td>1.036640e+06</td>
      <td>-164.148619</td>
      <td>0.685651</td>
      <td>0.008764</td>
      <td>0.000229</td>
      <td>75.712491</td>
      <td>6.057509</td>
    </tr>
    <tr>
      <th>2010-01-06</th>
      <td>83.18</td>
      <td>2.37</td>
      <td>92.0941</td>
      <td>1.036740e+06</td>
      <td>-109.632478</td>
      <td>0.285432</td>
      <td>0.017643</td>
      <td>0.000177</td>
      <td>76.195043</td>
      <td>6.984957</td>
    </tr>
    <tr>
      <th>2010-01-07</th>
      <td>82.66</td>
      <td>2.41</td>
      <td>92.3684</td>
      <td>1.036841e+06</td>
      <td>-49.771309</td>
      <td>-0.418137</td>
      <td>0.016535</td>
      <td>0.000121</td>
      <td>76.468911</td>
      <td>6.191089</td>
    </tr>
    <tr>
      <th>2010-01-08</th>
      <td>82.75</td>
      <td>2.42</td>
      <td>92.1485</td>
      <td>1.036941e+06</td>
      <td>-20.414187</td>
      <td>-0.484475</td>
      <td>0.019674</td>
      <td>0.000093</td>
      <td>76.778149</td>
      <td>5.971851</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2021-12-27</th>
      <td>75.57</td>
      <td>2.50</td>
      <td>115.4964</td>
      <td>1.015275e+06</td>
      <td>-228.150461</td>
      <td>20.315538</td>
      <td>-0.470935</td>
      <td>0.000297</td>
      <td>69.904710</td>
      <td>5.665290</td>
    </tr>
    <tr>
      <th>2021-12-28</th>
      <td>75.98</td>
      <td>2.50</td>
      <td>115.4497</td>
      <td>1.014339e+06</td>
      <td>-187.728962</td>
      <td>21.927039</td>
      <td>-0.507664</td>
      <td>0.000258</td>
      <td>70.527938</td>
      <td>5.452062</td>
    </tr>
    <tr>
      <th>2021-12-29</th>
      <td>76.56</td>
      <td>2.53</td>
      <td>115.3964</td>
      <td>1.013404e+06</td>
      <td>-150.824993</td>
      <td>23.830340</td>
      <td>-0.550962</td>
      <td>0.000223</td>
      <td>71.812860</td>
      <td>4.747140</td>
    </tr>
    <tr>
      <th>2021-12-30</th>
      <td>76.99</td>
      <td>2.58</td>
      <td>115.3163</td>
      <td>1.012468e+06</td>
      <td>-117.391030</td>
      <td>26.261025</td>
      <td>-0.606247</td>
      <td>0.000191</td>
      <td>73.678898</td>
      <td>3.311102</td>
    </tr>
    <tr>
      <th>2021-12-31</th>
      <td>75.21</td>
      <td>2.56</td>
      <td>0.0000</td>
      <td>1.011533e+06</td>
      <td>-279.821665</td>
      <td>3.975534</td>
      <td>-0.088994</td>
      <td>0.000345</td>
      <td>79.247339</td>
      <td>-4.037339</td>
    </tr>
  </tbody>
</table>
<p>3023 rows × 10 columns</p>
</div>



### Now let's read in the WTI data with the roll implemented from the first assignment:


```python
#Reading in the files
df_wti           = pd.read_excel('HW3input.xlsx', sheet_name = 'WTI')

#transaction cost
t = 0.01
```


```python
df_wti = df_wti.set_index('Date')
df_wti = df_wti.loc['2010-01-04':]
df_wti['F1 - Predicted'] = df1_regrange['F1 - Predicted']
df_wti
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>F1</th>
      <th>F2</th>
      <th>F3</th>
      <th>F4</th>
      <th>Execution_day</th>
      <th>Holding_type</th>
      <th>P/L_F1</th>
      <th>P/L_F2</th>
      <th>Cumulative_P/L_barrel</th>
      <th>Cumulative_P/L_strat</th>
      <th>F1 - Predicted</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010-01-04</th>
      <td>81.51</td>
      <td>82.12</td>
      <td>82.65</td>
      <td>83.12</td>
      <td>0</td>
      <td>0</td>
      <td>2.15</td>
      <td>2.10</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>6.217096</td>
    </tr>
    <tr>
      <th>2010-01-05</th>
      <td>81.77</td>
      <td>82.41</td>
      <td>82.99</td>
      <td>83.52</td>
      <td>0</td>
      <td>0</td>
      <td>0.26</td>
      <td>0.29</td>
      <td>0.26</td>
      <td>26000.0</td>
      <td>6.057509</td>
    </tr>
    <tr>
      <th>2010-01-06</th>
      <td>83.18</td>
      <td>83.75</td>
      <td>84.31</td>
      <td>84.86</td>
      <td>0</td>
      <td>0</td>
      <td>1.41</td>
      <td>1.34</td>
      <td>1.67</td>
      <td>167000.0</td>
      <td>6.984957</td>
    </tr>
    <tr>
      <th>2010-01-07</th>
      <td>82.66</td>
      <td>83.19</td>
      <td>83.75</td>
      <td>84.29</td>
      <td>0</td>
      <td>0</td>
      <td>-0.52</td>
      <td>-0.56</td>
      <td>1.15</td>
      <td>115000.0</td>
      <td>6.191089</td>
    </tr>
    <tr>
      <th>2010-01-08</th>
      <td>82.75</td>
      <td>83.30</td>
      <td>83.87</td>
      <td>84.47</td>
      <td>0</td>
      <td>0</td>
      <td>0.09</td>
      <td>0.11</td>
      <td>1.24</td>
      <td>124000.0</td>
      <td>5.971851</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2021-12-27</th>
      <td>75.57</td>
      <td>75.18</td>
      <td>74.67</td>
      <td>74.12</td>
      <td>0</td>
      <td>0</td>
      <td>1.78</td>
      <td>1.76</td>
      <td>-63.52</td>
      <td>-6352000.0</td>
      <td>5.665290</td>
    </tr>
    <tr>
      <th>2021-12-28</th>
      <td>75.98</td>
      <td>75.60</td>
      <td>75.13</td>
      <td>74.59</td>
      <td>0</td>
      <td>0</td>
      <td>0.41</td>
      <td>0.42</td>
      <td>-63.11</td>
      <td>-6311000.0</td>
      <td>5.452062</td>
    </tr>
    <tr>
      <th>2021-12-29</th>
      <td>76.56</td>
      <td>76.18</td>
      <td>75.71</td>
      <td>75.18</td>
      <td>0</td>
      <td>0</td>
      <td>0.58</td>
      <td>0.58</td>
      <td>-62.53</td>
      <td>-6253000.0</td>
      <td>4.747140</td>
    </tr>
    <tr>
      <th>2021-12-30</th>
      <td>76.99</td>
      <td>76.61</td>
      <td>76.13</td>
      <td>75.58</td>
      <td>0</td>
      <td>0</td>
      <td>0.43</td>
      <td>0.43</td>
      <td>-62.10</td>
      <td>-6210000.0</td>
      <td>3.311102</td>
    </tr>
    <tr>
      <th>2021-12-31</th>
      <td>75.21</td>
      <td>74.88</td>
      <td>74.45</td>
      <td>73.94</td>
      <td>0</td>
      <td>0</td>
      <td>-1.78</td>
      <td>-1.73</td>
      <td>-63.88</td>
      <td>-6388000.0</td>
      <td>-4.037339</td>
    </tr>
  </tbody>
</table>
<p>3023 rows × 11 columns</p>
</div>



### Implementing our strategy and calculating all the necessary values:


```python
df_wti['C(0)']  = 0.0

epsilon = 0

for i in range(len(df_wti)):
    if df_wti['F1 - Predicted'][i] > epsilon:
        df_wti['C(0)'][i] = -1.0
    elif df_wti['F1 - Predicted'][i] < epsilon:
        df_wti['C(0)'][i] = 1.0
```


```python
df_wti['P/L'] = 0.0

for i in range(1,len(df_wti)):
    df_wti['P/L'][i] = df_wti['Cumulative_P/L_barrel'][i]-df_wti['Cumulative_P/L_barrel'][i-1]
```


```python
df_wti['P/L_short'] = -df_wti['P/L']

# have to reverse the transaction costs for the roll
for i in range(1,len(df_wti)):
    if df_wti['Holding_type'][i] == 1 and df_wti['Holding_type'][i-1] == 0:
        df_wti['P/L_short'][i-1] = df_wti['P/L_short'][i-1] - 0.04
```


```python
df_wti.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>F1</th>
      <th>F2</th>
      <th>F3</th>
      <th>F4</th>
      <th>Execution_day</th>
      <th>Holding_type</th>
      <th>P/L_F1</th>
      <th>P/L_F2</th>
      <th>Cumulative_P/L_barrel</th>
      <th>Cumulative_P/L_strat</th>
      <th>F1 - Predicted</th>
      <th>C(0)</th>
      <th>P/L</th>
      <th>P/L_short</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-12-27</th>
      <td>75.57</td>
      <td>75.18</td>
      <td>74.67</td>
      <td>74.12</td>
      <td>0</td>
      <td>0</td>
      <td>1.78</td>
      <td>1.76</td>
      <td>-63.52</td>
      <td>-6352000.0</td>
      <td>5.665290</td>
      <td>-1.0</td>
      <td>1.78</td>
      <td>-1.78</td>
    </tr>
    <tr>
      <th>2021-12-28</th>
      <td>75.98</td>
      <td>75.60</td>
      <td>75.13</td>
      <td>74.59</td>
      <td>0</td>
      <td>0</td>
      <td>0.41</td>
      <td>0.42</td>
      <td>-63.11</td>
      <td>-6311000.0</td>
      <td>5.452062</td>
      <td>-1.0</td>
      <td>0.41</td>
      <td>-0.41</td>
    </tr>
    <tr>
      <th>2021-12-29</th>
      <td>76.56</td>
      <td>76.18</td>
      <td>75.71</td>
      <td>75.18</td>
      <td>0</td>
      <td>0</td>
      <td>0.58</td>
      <td>0.58</td>
      <td>-62.53</td>
      <td>-6253000.0</td>
      <td>4.747140</td>
      <td>-1.0</td>
      <td>0.58</td>
      <td>-0.58</td>
    </tr>
    <tr>
      <th>2021-12-30</th>
      <td>76.99</td>
      <td>76.61</td>
      <td>76.13</td>
      <td>75.58</td>
      <td>0</td>
      <td>0</td>
      <td>0.43</td>
      <td>0.43</td>
      <td>-62.10</td>
      <td>-6210000.0</td>
      <td>3.311102</td>
      <td>-1.0</td>
      <td>0.43</td>
      <td>-0.43</td>
    </tr>
    <tr>
      <th>2021-12-31</th>
      <td>75.21</td>
      <td>74.88</td>
      <td>74.45</td>
      <td>73.94</td>
      <td>0</td>
      <td>0</td>
      <td>-1.78</td>
      <td>-1.73</td>
      <td>-63.88</td>
      <td>-6388000.0</td>
      <td>-4.037339</td>
      <td>1.0</td>
      <td>-1.78</td>
      <td>1.78</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_wti['P/L_C(0)'] = 0.0
df_wti['Cumulative_P/L_C(0)'] = 0.0

# deciding whether your P&L is from the long or short position
for i in range(1,len(df_wti)):
    if (df_wti['C(0)'][i-1] == 1):
        df_wti['P/L_C(0)'][i] = df_wti['P/L'][i]

    if (df_wti['C(0)'][i-1] == -1):
        df_wti['P/L_C(0)'][i] = df_wti['P/L_short'][i]


# if it is time to switch positions you have to add the transaction costs
for i in range(1,len(df_wti)):
    if df_wti['C(0)'][i] != df_wti['C(0)'][i-1]:
        df_wti['P/L_C(0)'][i] -= 0.02

# calculating cumulative P&L
for i in range(1,len(df_wti)):
    df_wti['Cumulative_P/L_C(0)'][i] = df_wti['P/L_C(0)'][i] + df_wti['Cumulative_P/L_C(0)'][i-1]

#just to see the behaviour when the c(0) sign flips. You can see we account for the transaction costs here
df_wti.tail(25)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>F1</th>
      <th>F2</th>
      <th>F3</th>
      <th>F4</th>
      <th>Execution_day</th>
      <th>Holding_type</th>
      <th>P/L_F1</th>
      <th>P/L_F2</th>
      <th>Cumulative_P/L_barrel</th>
      <th>Cumulative_P/L_strat</th>
      <th>F1 - Predicted</th>
      <th>C(0)</th>
      <th>P/L</th>
      <th>P/L_short</th>
      <th>P/L_C(0)</th>
      <th>Cumulative_P/L_C(0)</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-11-24</th>
      <td>78.39</td>
      <td>77.81</td>
      <td>77.15</td>
      <td>76.51</td>
      <td>0</td>
      <td>0</td>
      <td>-0.11</td>
      <td>-0.13</td>
      <td>-60.89</td>
      <td>-6089000.0</td>
      <td>0.794351</td>
      <td>-1.0</td>
      <td>-0.11</td>
      <td>0.11</td>
      <td>0.11</td>
      <td>47.73</td>
    </tr>
    <tr>
      <th>2021-11-29</th>
      <td>69.95</td>
      <td>69.62</td>
      <td>69.28</td>
      <td>68.92</td>
      <td>0</td>
      <td>0</td>
      <td>-8.44</td>
      <td>-8.19</td>
      <td>-69.33</td>
      <td>-6933000.0</td>
      <td>-5.015165</td>
      <td>1.0</td>
      <td>-8.44</td>
      <td>8.44</td>
      <td>8.42</td>
      <td>56.15</td>
    </tr>
    <tr>
      <th>2021-11-30</th>
      <td>66.18</td>
      <td>65.85</td>
      <td>65.49</td>
      <td>65.15</td>
      <td>0</td>
      <td>0</td>
      <td>-3.77</td>
      <td>-3.77</td>
      <td>-73.10</td>
      <td>-7310000.0</td>
      <td>-7.213953</td>
      <td>1.0</td>
      <td>-3.77</td>
      <td>3.77</td>
      <td>-3.77</td>
      <td>52.38</td>
    </tr>
    <tr>
      <th>2021-12-01</th>
      <td>65.57</td>
      <td>65.37</td>
      <td>65.11</td>
      <td>64.83</td>
      <td>0</td>
      <td>0</td>
      <td>-0.61</td>
      <td>-0.48</td>
      <td>-73.71</td>
      <td>-7371000.0</td>
      <td>-5.902779</td>
      <td>1.0</td>
      <td>-0.61</td>
      <td>0.61</td>
      <td>-0.61</td>
      <td>51.77</td>
    </tr>
    <tr>
      <th>2021-12-02</th>
      <td>66.50</td>
      <td>66.27</td>
      <td>66.03</td>
      <td>65.77</td>
      <td>0</td>
      <td>0</td>
      <td>0.93</td>
      <td>0.90</td>
      <td>-72.78</td>
      <td>-7278000.0</td>
      <td>-4.597826</td>
      <td>1.0</td>
      <td>0.93</td>
      <td>-0.93</td>
      <td>0.93</td>
      <td>52.70</td>
    </tr>
    <tr>
      <th>2021-12-03</th>
      <td>66.26</td>
      <td>66.10</td>
      <td>65.93</td>
      <td>65.71</td>
      <td>0</td>
      <td>0</td>
      <td>-0.24</td>
      <td>-0.17</td>
      <td>-73.02</td>
      <td>-7302000.0</td>
      <td>-3.430409</td>
      <td>1.0</td>
      <td>-0.24</td>
      <td>0.24</td>
      <td>-0.24</td>
      <td>52.46</td>
    </tr>
    <tr>
      <th>2021-12-06</th>
      <td>69.49</td>
      <td>69.30</td>
      <td>69.07</td>
      <td>68.80</td>
      <td>0</td>
      <td>0</td>
      <td>3.23</td>
      <td>3.20</td>
      <td>-69.79</td>
      <td>-6979000.0</td>
      <td>0.670935</td>
      <td>-1.0</td>
      <td>3.23</td>
      <td>-3.23</td>
      <td>3.21</td>
      <td>55.67</td>
    </tr>
    <tr>
      <th>2021-12-07</th>
      <td>72.05</td>
      <td>71.84</td>
      <td>71.56</td>
      <td>71.26</td>
      <td>0</td>
      <td>0</td>
      <td>2.56</td>
      <td>2.54</td>
      <td>-67.23</td>
      <td>-6723000.0</td>
      <td>3.435389</td>
      <td>-1.0</td>
      <td>2.56</td>
      <td>-2.56</td>
      <td>-2.56</td>
      <td>53.11</td>
    </tr>
    <tr>
      <th>2021-12-08</th>
      <td>72.36</td>
      <td>72.18</td>
      <td>71.90</td>
      <td>71.64</td>
      <td>0</td>
      <td>0</td>
      <td>0.31</td>
      <td>0.34</td>
      <td>-66.92</td>
      <td>-6692000.0</td>
      <td>3.479082</td>
      <td>-1.0</td>
      <td>0.31</td>
      <td>-0.31</td>
      <td>-0.31</td>
      <td>52.80</td>
    </tr>
    <tr>
      <th>2021-12-09</th>
      <td>70.94</td>
      <td>70.79</td>
      <td>70.54</td>
      <td>70.29</td>
      <td>0</td>
      <td>0</td>
      <td>-1.42</td>
      <td>-1.39</td>
      <td>-68.34</td>
      <td>-6834000.0</td>
      <td>3.377357</td>
      <td>-1.0</td>
      <td>-1.42</td>
      <td>1.42</td>
      <td>1.42</td>
      <td>54.22</td>
    </tr>
    <tr>
      <th>2021-12-10</th>
      <td>71.67</td>
      <td>71.48</td>
      <td>71.22</td>
      <td>70.95</td>
      <td>0</td>
      <td>0</td>
      <td>0.73</td>
      <td>0.69</td>
      <td>-67.61</td>
      <td>-6761000.0</td>
      <td>4.774274</td>
      <td>-1.0</td>
      <td>0.73</td>
      <td>-0.73</td>
      <td>-0.73</td>
      <td>53.49</td>
    </tr>
    <tr>
      <th>2021-12-13</th>
      <td>71.29</td>
      <td>71.06</td>
      <td>70.80</td>
      <td>70.53</td>
      <td>0</td>
      <td>0</td>
      <td>-0.38</td>
      <td>-0.42</td>
      <td>-67.99</td>
      <td>-6799000.0</td>
      <td>5.331989</td>
      <td>-1.0</td>
      <td>-0.38</td>
      <td>0.38</td>
      <td>0.38</td>
      <td>53.87</td>
    </tr>
    <tr>
      <th>2021-12-14</th>
      <td>70.73</td>
      <td>70.52</td>
      <td>70.26</td>
      <td>69.98</td>
      <td>0</td>
      <td>0</td>
      <td>-0.56</td>
      <td>-0.54</td>
      <td>-68.57</td>
      <td>-6857000.0</td>
      <td>5.479031</td>
      <td>-1.0</td>
      <td>-0.58</td>
      <td>0.54</td>
      <td>0.54</td>
      <td>54.41</td>
    </tr>
    <tr>
      <th>2021-12-15</th>
      <td>70.87</td>
      <td>70.66</td>
      <td>70.36</td>
      <td>70.05</td>
      <td>0</td>
      <td>1</td>
      <td>0.14</td>
      <td>0.14</td>
      <td>-68.43</td>
      <td>-6843000.0</td>
      <td>5.293991</td>
      <td>-1.0</td>
      <td>0.14</td>
      <td>-0.14</td>
      <td>-0.14</td>
      <td>54.27</td>
    </tr>
    <tr>
      <th>2021-12-16</th>
      <td>72.38</td>
      <td>72.15</td>
      <td>71.75</td>
      <td>71.35</td>
      <td>0</td>
      <td>1</td>
      <td>1.51</td>
      <td>1.49</td>
      <td>-66.94</td>
      <td>-6694000.0</td>
      <td>5.796050</td>
      <td>-1.0</td>
      <td>1.49</td>
      <td>-1.49</td>
      <td>-1.49</td>
      <td>52.78</td>
    </tr>
    <tr>
      <th>2021-12-17</th>
      <td>70.86</td>
      <td>70.72</td>
      <td>70.35</td>
      <td>69.95</td>
      <td>0</td>
      <td>1</td>
      <td>-1.52</td>
      <td>-1.43</td>
      <td>-68.37</td>
      <td>-6837000.0</td>
      <td>4.801267</td>
      <td>-1.0</td>
      <td>-1.43</td>
      <td>1.43</td>
      <td>1.43</td>
      <td>54.21</td>
    </tr>
    <tr>
      <th>2021-12-20</th>
      <td>68.23</td>
      <td>68.61</td>
      <td>68.36</td>
      <td>68.03</td>
      <td>0</td>
      <td>1</td>
      <td>-2.63</td>
      <td>-2.11</td>
      <td>-70.48</td>
      <td>-7048000.0</td>
      <td>2.162640</td>
      <td>-1.0</td>
      <td>-2.11</td>
      <td>2.11</td>
      <td>2.11</td>
      <td>56.32</td>
    </tr>
    <tr>
      <th>2021-12-21</th>
      <td>71.12</td>
      <td>70.82</td>
      <td>70.44</td>
      <td>70.05</td>
      <td>1</td>
      <td>1</td>
      <td>2.89</td>
      <td>2.21</td>
      <td>-68.27</td>
      <td>-6827000.0</td>
      <td>4.050283</td>
      <td>-1.0</td>
      <td>2.21</td>
      <td>-2.21</td>
      <td>-2.21</td>
      <td>54.11</td>
    </tr>
    <tr>
      <th>2021-12-22</th>
      <td>72.76</td>
      <td>72.33</td>
      <td>71.83</td>
      <td>71.32</td>
      <td>0</td>
      <td>0</td>
      <td>1.64</td>
      <td>1.51</td>
      <td>-66.33</td>
      <td>-6633000.0</td>
      <td>4.650285</td>
      <td>-1.0</td>
      <td>1.94</td>
      <td>-1.94</td>
      <td>-1.94</td>
      <td>52.17</td>
    </tr>
    <tr>
      <th>2021-12-23</th>
      <td>73.79</td>
      <td>73.42</td>
      <td>72.93</td>
      <td>72.40</td>
      <td>0</td>
      <td>0</td>
      <td>1.03</td>
      <td>1.09</td>
      <td>-65.30</td>
      <td>-6530000.0</td>
      <td>5.114491</td>
      <td>-1.0</td>
      <td>1.03</td>
      <td>-1.03</td>
      <td>-1.03</td>
      <td>51.14</td>
    </tr>
    <tr>
      <th>2021-12-27</th>
      <td>75.57</td>
      <td>75.18</td>
      <td>74.67</td>
      <td>74.12</td>
      <td>0</td>
      <td>0</td>
      <td>1.78</td>
      <td>1.76</td>
      <td>-63.52</td>
      <td>-6352000.0</td>
      <td>5.665290</td>
      <td>-1.0</td>
      <td>1.78</td>
      <td>-1.78</td>
      <td>-1.78</td>
      <td>49.36</td>
    </tr>
    <tr>
      <th>2021-12-28</th>
      <td>75.98</td>
      <td>75.60</td>
      <td>75.13</td>
      <td>74.59</td>
      <td>0</td>
      <td>0</td>
      <td>0.41</td>
      <td>0.42</td>
      <td>-63.11</td>
      <td>-6311000.0</td>
      <td>5.452062</td>
      <td>-1.0</td>
      <td>0.41</td>
      <td>-0.41</td>
      <td>-0.41</td>
      <td>48.95</td>
    </tr>
    <tr>
      <th>2021-12-29</th>
      <td>76.56</td>
      <td>76.18</td>
      <td>75.71</td>
      <td>75.18</td>
      <td>0</td>
      <td>0</td>
      <td>0.58</td>
      <td>0.58</td>
      <td>-62.53</td>
      <td>-6253000.0</td>
      <td>4.747140</td>
      <td>-1.0</td>
      <td>0.58</td>
      <td>-0.58</td>
      <td>-0.58</td>
      <td>48.37</td>
    </tr>
    <tr>
      <th>2021-12-30</th>
      <td>76.99</td>
      <td>76.61</td>
      <td>76.13</td>
      <td>75.58</td>
      <td>0</td>
      <td>0</td>
      <td>0.43</td>
      <td>0.43</td>
      <td>-62.10</td>
      <td>-6210000.0</td>
      <td>3.311102</td>
      <td>-1.0</td>
      <td>0.43</td>
      <td>-0.43</td>
      <td>-0.43</td>
      <td>47.94</td>
    </tr>
    <tr>
      <th>2021-12-31</th>
      <td>75.21</td>
      <td>74.88</td>
      <td>74.45</td>
      <td>73.94</td>
      <td>0</td>
      <td>0</td>
      <td>-1.78</td>
      <td>-1.73</td>
      <td>-63.88</td>
      <td>-6388000.0</td>
      <td>-4.037339</td>
      <td>1.0</td>
      <td>-1.78</td>
      <td>1.78</td>
      <td>1.76</td>
      <td>49.70</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_wti['P(t)_C(0)'] = 0.0

for i in range(1,len(df_wti)):
    df_wti['P(t)_C(0)'][i] = df_wti['P(t)_C(0)'][i-1] +(df_wti['P/L_C(0)'][i]*100000)

df_wti.tail(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>F1</th>
      <th>F2</th>
      <th>F3</th>
      <th>F4</th>
      <th>Execution_day</th>
      <th>Holding_type</th>
      <th>P/L_F1</th>
      <th>P/L_F2</th>
      <th>Cumulative_P/L_barrel</th>
      <th>Cumulative_P/L_strat</th>
      <th>F1 - Predicted</th>
      <th>C(0)</th>
      <th>P/L</th>
      <th>P/L_short</th>
      <th>P/L_C(0)</th>
      <th>Cumulative_P/L_C(0)</th>
      <th>P(t)_C(0)</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-12-30</th>
      <td>76.99</td>
      <td>76.61</td>
      <td>76.13</td>
      <td>75.58</td>
      <td>0</td>
      <td>0</td>
      <td>0.43</td>
      <td>0.43</td>
      <td>-62.10</td>
      <td>-6210000.0</td>
      <td>3.311102</td>
      <td>-1.0</td>
      <td>0.43</td>
      <td>-0.43</td>
      <td>-0.43</td>
      <td>47.94</td>
      <td>4794000.0</td>
    </tr>
    <tr>
      <th>2021-12-31</th>
      <td>75.21</td>
      <td>74.88</td>
      <td>74.45</td>
      <td>73.94</td>
      <td>0</td>
      <td>0</td>
      <td>-1.78</td>
      <td>-1.73</td>
      <td>-63.88</td>
      <td>-6388000.0</td>
      <td>-4.037339</td>
      <td>1.0</td>
      <td>-1.78</td>
      <td>1.78</td>
      <td>1.76</td>
      <td>49.70</td>
      <td>4970000.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_wti['dF(t)_C(0)'] = df_wti['P/L_C(0)']*100000
```

### Calculating performance metrics:


```python
N = 12

ann_PL_C = df_wti['P(t)_C(0)'][-1]/N

print(f"C(0) Average Annual P&L: {ann_PL_C}")
```

    C(0) Average Annual P&L: 414166.66666666535



```python
ann_SR_C = ann_PL_C/(np.std(df_wti['dF(t)_C(0)'])*np.sqrt(250))

print(f"C(0) Annualised Sharpe Ratio: {ann_SR_C}")
```

    C(0) Annualised Sharpe Ratio: 0.1914764968595926



```python
df_wti['HWM_C(0)'] = 0.0
df_wti['DD_C(0)'] = 0.0

for i in range(len(df_wti)):
    df_wti['HWM_C(0)'][i] = np.max(df_wti['P(t)_C(0)'][0:i+1])

    df_wti['DD_C(0)'][i] = df_wti['HWM_C(0)'][i] - df_wti['P(t)_C(0)'][i]

```


```python
mdd_C = np.max(df_wti['DD_C(0)'])
mdd_Cdate = df_wti['DD_C(0)'].idxmax()

print(f"C(0) Maximum Drawdown: {mdd_C} at {mdd_Cdate}")
```

    C(0) Maximum Drawdown: 6535999.999999996 at 2015-08-18 00:00:00


### Graphing the equity line:


```python
import seaborn as sns
sns.set()

plt.figure(figsize=(10, 6), dpi=80)
plt.ticklabel_format(style='plain')
plt.plot(df_wti.index, df_wti['P(t)_C(0)'], label = 'C(0) Equity Line')
plt.plot(df_wti.index, -df_wti['DD_C(0)'], label = 'C(0) Drawdown',alpha=0.5)
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.savefig('equity line and drawdown not optpitmized')
plt.show()
```


![png](https://github.com/arjunkalsi/arjunkalsi.github.io/tree/master/img/wti/output_43_0.png)


### Now let's try to find the optimal parameters for the strategy. First let's find the optimal value for w, which is the rolling window value for the rolling regression calculation:


```python
ws = np.arange(11,70,1)

SRoptimal = 0.0
Wopt = 0.0
epsilon = 0

for w in ws:
    #print(w,':')
    #print('')
    endog                          = df1['F1']
    exog                           = sm.add_constant(df1[['Inflation Rate','USD Index','Stocks']])
    rols                           = RollingOLS(endog, exog, window=w,min_nobs=10)
    rres                           = rols.fit()
    params                         = rres.params.copy()
    params.index                   = np.arange(1, params.shape[0] + 1)

    params['Date']                 = df1.index
    params                         = params.set_index('Date')

    df1_reg                        = pd.merge(df1,params,how = 'left',on='Date')

    df1_regrange                   = df1_reg.loc['2010-01-04':]
    df1_regrange['Predicted F1']   = df1_regrange['const'] + (df1_regrange['Inflation Rate_y']*df1_regrange['Inflation Rate_x']) + (df1_regrange['USD Index_y']*df1_regrange['USD Index_x']) + (df1_regrange['Stocks_y']*df1_regrange['Stocks_x'])

    df1_regrange['F1 - Predicted'] = df1_regrange['F1'] - df1_regrange['Predicted F1']
    df_wti['F1 - Predicted']       = df1_regrange['F1 - Predicted']

    df_wti['C(0)']  = 0.0
    for i in range(len(df_wti)):
        if df_wti['F1 - Predicted'][i] > epsilon:
            df_wti['C(0)'][i] = -1.0
        elif df_wti['F1 - Predicted'][i] < epsilon:
            df_wti['C(0)'][i] = 1.0

    df_wti['P/L_C(0)'] = 0.0
    df_wti['Cumulative_P/L_C(0)'] = 0.0

    # deciding whether your P&L is from the long or short position
    for i in range(1,len(df_wti)):
        if (df_wti['C(0)'][i-1] == 1):
            df_wti['P/L_C(0)'][i] = df_wti['P/L'][i]

        if (df_wti['C(0)'][i-1] == -1):
            df_wti['P/L_C(0)'][i] = df_wti['P/L_short'][i]


    # if it is time to switch positions you have to add the transaction costs
    for i in range(1,len(df_wti)):
        if df_wti['C(0)'][i] != df_wti['C(0)'][i-1]:
            df_wti['P/L_C(0)'][i] -= 0.02

    # calculating cumulative P&L
    for i in range(1,len(df_wti)):
        df_wti['Cumulative_P/L_C(0)'][i] = df_wti['P/L_C(0)'][i] + df_wti['Cumulative_P/L_C(0)'][i-1]

    df_wti['P(t)_C(0)'] = 0.0

    for i in range(1,len(df_wti)):
        df_wti['P(t)_C(0)'][i] = df_wti['P(t)_C(0)'][i-1] +(df_wti['P/L_C(0)'][i]*100000)

    df_wti['dF(t)_C(0)'] = df_wti['P/L_C(0)']*100000

    ann_PL_C = df_wti['P(t)_C(0)'][-1]/N
    #print(f"C(0) Average Annual P&L: {ann_PL_C}")

    ann_SR_C = ann_PL_C/(np.std(df_wti['dF(t)_C(0)'])*np.sqrt(250))
    #print(f"C(0) Annualised Sharpe Ratio: {ann_SR_C}")

    df_wti['HWM_C(0)'] = 0.0
    df_wti['DD_C(0)'] = 0.0

    for i in range(len(df_wti)):
        df_wti['HWM_C(0)'][i] = np.max(df_wti['P(t)_C(0)'][0:i+1])
        df_wti['DD_C(0)'][i] = df_wti['HWM_C(0)'][i] - df_wti['P(t)_C(0)'][i]

    mdd_C = np.max(df_wti['DD_C(0)'])
    mdd_Cdate = df_wti['DD_C(0)'].idxmax()

    #print(f"C(0) Maximum Drawdown: {mdd_C} at {mdd_Cdate}")

    if ann_SR_C > SRoptimal:
        SRoptimal = ann_SR_C
        Wopt = w
```


```python
print(SRoptimal)
print(Wopt)
```

    0.42060292596024096
    50


### Thus the optimal rolling regression parameter is w=50, giving a Sharpe Ratio of 0.42 (epsilon=0)

### Run these cells again:


```python
w = 50
endog = df1['F1']
exog = sm.add_constant(df1[['Inflation Rate','USD Index','Stocks']])
rols = RollingOLS(endog, exog, window=w)
rres = rols.fit()
params = rres.params.copy()
params.index = np.arange(1, params.shape[0] + 1)
```


```python
params['Date'] = df1.index
params = params.set_index('Date')
```


```python
df1_reg = pd.merge(df1,params,how = 'left',on='Date')
```


```python
df1_regrange = df1_reg.loc['2010-01-04':]
```


```python
df1_regrange['Predicted F1'] = df1_regrange['const'] + (df1_regrange['Inflation Rate_y']*df1_regrange['Inflation Rate_x']) + (df1_regrange['USD Index_y']*df1_regrange['USD Index_x']) + (df1_regrange['Stocks_y']*df1_regrange['Stocks_x'])
```


```python
plt.figure(figsize=(12,8))
plt.plot(df1_regrange['F1'],label='F1')
plt.plot(df1_regrange['Predicted F1'],label='Predicted F1')
plt.legend()
plt.savefig('F1 versus Predicted F1 (opt window=50)')
plt.show()
```


![png](https://github.com/arjunkalsi/arjunkalsi.github.io/tree/master/img/wti/output_53_0.png)



```python
df1_regrange['F1 - Predicted'] = df1_regrange['F1'] - df1_regrange['Predicted F1']
```


```python
#Reading in the files
df_wti           = pd.read_excel('HW3input.xlsx', sheet_name = 'WTI')

#transaction cost
t = 0.01
```


```python
df_wti = df_wti.set_index('Date')
df_wti = df_wti.loc['2010-01-04':]
df_wti['F1 - Predicted'] = df1_regrange['F1 - Predicted']
```


```python
df_wti['P/L'] = 0.0

for i in range(1,len(df_wti)):
    df_wti['P/L'][i] = df_wti['Cumulative_P/L_barrel'][i]-df_wti['Cumulative_P/L_barrel'][i-1]
```


```python
df_wti['P/L_short'] = -df_wti['P/L']

# have to reverse the transaction costs for the roll
for i in range(1,len(df_wti)):
    if df_wti['Holding_type'][i] == 1 and df_wti['Holding_type'][i-1] == 0:
        df_wti['P/L_short'][i-1] = df_wti['P/L_short'][i-1] - 0.04
```

### Now let's optimise epsilon, which is the parameter used in our strategy function:


```python
epsilons = np.arange(0.0,1.01,0.01)
SRoptimal = 0.0
Eopt = 0.0

for epsilon in epsilons:
    #print(epsilon,":")
    #print("")

    df_wti['C(0)']  = 0.0
    for i in range(len(df_wti)):
        if df_wti['F1 - Predicted'][i] > epsilon:
            df_wti['C(0)'][i] = -1.0
        elif df_wti['F1 - Predicted'][i] < epsilon:
            df_wti['C(0)'][i] = 1.0

    df_wti['P/L_C(0)'] = 0.0
    df_wti['Cumulative_P/L_C(0)'] = 0.0

    # deciding whether your P&L is from the long or short position
    for i in range(1,len(df_wti)):
        if (df_wti['C(0)'][i-1] == 1):
            df_wti['P/L_C(0)'][i] = df_wti['P/L'][i]

        if (df_wti['C(0)'][i-1] == -1):
            df_wti['P/L_C(0)'][i] = df_wti['P/L_short'][i]


    # if it is time to switch positions you have to add the transaction costs
    for i in range(1,len(df_wti)):
        if df_wti['C(0)'][i] != df_wti['C(0)'][i-1]:
            df_wti['P/L_C(0)'][i] -= 0.02

    # calculating cumulative P&L
    for i in range(1,len(df_wti)):
        df_wti['Cumulative_P/L_C(0)'][i] = df_wti['P/L_C(0)'][i] + df_wti['Cumulative_P/L_C(0)'][i-1]

    df_wti['P(t)_C(0)'] = 0.0

    for i in range(1,len(df_wti)):
        df_wti['P(t)_C(0)'][i] = df_wti['P(t)_C(0)'][i-1] +(df_wti['P/L_C(0)'][i]*100000)

    df_wti['dF(t)_C(0)'] = df_wti['P/L_C(0)']*100000

    ann_PL_C = df_wti['P(t)_C(0)'][-1]/N
    #print(f"C({epsilon}) Average Annual P&L: {ann_PL_C}")

    ann_SR_C = ann_PL_C/(np.std(df_wti['dF(t)_C(0)'])*np.sqrt(250))
    #print(f"C({epsilon}) Annualised Sharpe Ratio: {ann_SR_C}")

    df_wti['HWM_C(0)'] = 0.0
    df_wti['DD_C(0)'] = 0.0

    for i in range(len(df_wti)):
        df_wti['HWM_C(0)'][i] = np.max(df_wti['P(t)_C(0)'][0:i+1])
        df_wti['DD_C(0)'][i] = df_wti['HWM_C(0)'][i] - df_wti['P(t)_C(0)'][i]

    mdd_C = np.max(df_wti['DD_C(0)'])
    mdd_Cdate = df_wti['DD_C(0)'].idxmax()

    #print(f"C({epsilon}) Maximum Drawdown: {mdd_C} at {mdd_Cdate}")
    #print("")

    if ann_SR_C > SRoptimal:
        SRoptimal = ann_SR_C
        Eopt = epsilon

```


```python
print(f'Best Sharpe Ratio: {SRoptimal}, Best epsilon: {Eopt}')
```

    Best Sharpe Ratio: 0.42060292596024096, Best epsilon: 0.0


### Thus the optimal Sharpe Ratio is 0.42 when $\epsilon = 0$. Let's experiment with higher values of $\epsilon$:


```python
epsilons = [10,9,8,7,6,5,4,3,2,0]
SRoptimal = 0.0
Eopt = 0.0

for epsilon in epsilons:
    #print(epsilon,":")
    #print("")

    df_wti['C(0)']  = 0.0
    for i in range(len(df_wti)):
        if df_wti['F1 - Predicted'][i] > epsilon:
            df_wti['C(0)'][i] = -1.0
        elif df_wti['F1 - Predicted'][i] < epsilon:
            df_wti['C(0)'][i] = 1.0

    df_wti['P/L_C(0)'] = 0.0
    df_wti['Cumulative_P/L_C(0)'] = 0.0

    # deciding whether your P&L is from the long or short position
    for i in range(1,len(df_wti)):
        if (df_wti['C(0)'][i-1] == 1):
            df_wti['P/L_C(0)'][i] = df_wti['P/L'][i]

        if (df_wti['C(0)'][i-1] == -1):
            df_wti['P/L_C(0)'][i] = df_wti['P/L_short'][i]


    # if it is time to switch positions you have to add the transaction costs
    for i in range(1,len(df_wti)):
        if df_wti['C(0)'][i] != df_wti['C(0)'][i-1]:
            df_wti['P/L_C(0)'][i] -= 0.02

    # calculating cumulative P&L
    for i in range(1,len(df_wti)):
        df_wti['Cumulative_P/L_C(0)'][i] = df_wti['P/L_C(0)'][i] + df_wti['Cumulative_P/L_C(0)'][i-1]

    df_wti['P(t)_C(0)'] = 0.0

    for i in range(1,len(df_wti)):
        df_wti['P(t)_C(0)'][i] = df_wti['P(t)_C(0)'][i-1] +(df_wti['P/L_C(0)'][i]*100000)

    df_wti['dF(t)_C(0)'] = df_wti['P/L_C(0)']*100000

    ann_PL_C = df_wti['P(t)_C(0)'][-1]/N
    #print(f"C({epsilon}) Average Annual P&L: {ann_PL_C}")

    ann_SR_C = ann_PL_C/(np.std(df_wti['dF(t)_C(0)'])*np.sqrt(250))
    #print(f"C({epsilon}) Annualised Sharpe Ratio: {ann_SR_C}")

    df_wti['HWM_C(0)'] = 0.0
    df_wti['DD_C(0)'] = 0.0

    for i in range(len(df_wti)):
        df_wti['HWM_C(0)'][i] = np.max(df_wti['P(t)_C(0)'][0:i+1])
        df_wti['DD_C(0)'][i] = df_wti['HWM_C(0)'][i] - df_wti['P(t)_C(0)'][i]

    mdd_C = np.max(df_wti['DD_C(0)'])
    mdd_Cdate = df_wti['DD_C(0)'].idxmax()

    #print(f"C({epsilon}) Maximum Drawdown: {mdd_C} at {mdd_Cdate}")
    #print("")

    if ann_SR_C > SRoptimal:
        SRoptimal = ann_SR_C
        Eopt = epsilon
```

### So $\epsilon = 0$, $w = 50$ is the optimal pairing here. Our final results are:

- C(0) Average Annual P&L: 909333.3333333308
- C(0) Annualised Sharpe Ratio: 0.42060292596024096
- C(0) Maximum Drawdown: 5717000.000000048 at 2021-11-01 00:00:00

### Let's finally see the tail end of the dataframe and also graph the equity line for these optimal parameters:


```python
df_wti.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>F1</th>
      <th>F2</th>
      <th>F3</th>
      <th>F4</th>
      <th>Execution_day</th>
      <th>Holding_type</th>
      <th>P/L_F1</th>
      <th>P/L_F2</th>
      <th>Cumulative_P/L_barrel</th>
      <th>Cumulative_P/L_strat</th>
      <th>F1 - Predicted</th>
      <th>P/L</th>
      <th>P/L_short</th>
      <th>C(0)</th>
      <th>P/L_C(0)</th>
      <th>Cumulative_P/L_C(0)</th>
      <th>P(t)_C(0)</th>
      <th>dF(t)_C(0)</th>
      <th>HWM_C(0)</th>
      <th>DD_C(0)</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-12-27</th>
      <td>75.57</td>
      <td>75.18</td>
      <td>74.67</td>
      <td>74.12</td>
      <td>0</td>
      <td>0</td>
      <td>1.78</td>
      <td>1.76</td>
      <td>-63.52</td>
      <td>-6352000.0</td>
      <td>5.077895</td>
      <td>1.78</td>
      <td>-1.78</td>
      <td>-1.0</td>
      <td>-1.78</td>
      <td>108.78</td>
      <td>10878000.0</td>
      <td>-178000.0</td>
      <td>15367000.0</td>
      <td>4489000.0</td>
    </tr>
    <tr>
      <th>2021-12-28</th>
      <td>75.98</td>
      <td>75.60</td>
      <td>75.13</td>
      <td>74.59</td>
      <td>0</td>
      <td>0</td>
      <td>0.41</td>
      <td>0.42</td>
      <td>-63.11</td>
      <td>-6311000.0</td>
      <td>4.739100</td>
      <td>0.41</td>
      <td>-0.41</td>
      <td>-1.0</td>
      <td>-0.41</td>
      <td>108.37</td>
      <td>10837000.0</td>
      <td>-41000.0</td>
      <td>15367000.0</td>
      <td>4530000.0</td>
    </tr>
    <tr>
      <th>2021-12-29</th>
      <td>76.56</td>
      <td>76.18</td>
      <td>75.71</td>
      <td>75.18</td>
      <td>0</td>
      <td>0</td>
      <td>0.58</td>
      <td>0.58</td>
      <td>-62.53</td>
      <td>-6253000.0</td>
      <td>3.866517</td>
      <td>0.58</td>
      <td>-0.58</td>
      <td>-1.0</td>
      <td>-0.58</td>
      <td>107.79</td>
      <td>10779000.0</td>
      <td>-58000.0</td>
      <td>15367000.0</td>
      <td>4588000.0</td>
    </tr>
    <tr>
      <th>2021-12-30</th>
      <td>76.99</td>
      <td>76.61</td>
      <td>76.13</td>
      <td>75.58</td>
      <td>0</td>
      <td>0</td>
      <td>0.43</td>
      <td>0.43</td>
      <td>-62.10</td>
      <td>-6210000.0</td>
      <td>2.297269</td>
      <td>0.43</td>
      <td>-0.43</td>
      <td>-1.0</td>
      <td>-0.43</td>
      <td>107.36</td>
      <td>10736000.0</td>
      <td>-43000.0</td>
      <td>15367000.0</td>
      <td>4631000.0</td>
    </tr>
    <tr>
      <th>2021-12-31</th>
      <td>75.21</td>
      <td>74.88</td>
      <td>74.45</td>
      <td>73.94</td>
      <td>0</td>
      <td>0</td>
      <td>-1.78</td>
      <td>-1.73</td>
      <td>-63.88</td>
      <td>-6388000.0</td>
      <td>-4.069233</td>
      <td>-1.78</td>
      <td>1.78</td>
      <td>1.0</td>
      <td>1.76</td>
      <td>109.12</td>
      <td>10912000.0</td>
      <td>176000.0</td>
      <td>15367000.0</td>
      <td>4455000.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(10, 6), dpi=80)
plt.ticklabel_format(style='plain')
plt.plot(df_wti.index, df_wti['P(t)_C(0)'], label = 'C(0) Equity Line')
plt.plot(df_wti.index, -df_wti['DD_C(0)'], label = 'C(0) Drawdown',alpha=0.5)
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.savefig('equity line and drawdown optimized')
plt.show()
```


![png](https://github.com/arjunkalsi/arjunkalsi.github.io/tree/master/img/wti/output_66_0.png)


### Let's download the data in an excel file


```python
# determining the name of the file
file_name = 'model_strategy_Arthur_Arjun.xlsx'

# saving the excel
df_wti.to_excel(file_name)
print('DataFrame is written to Excel File successfully.')
```

    DataFrame is written to Excel File successfully.


### On top of this, and in light of the current political landscape, we also thought it would be interesting to add war sentiment into the strategy somehow. So let's first find and import some data to hopefully derive some conclusions on a more complex strategy. The data we will be using is deaths from conflict and terrorism per 100,000 in the Middle East & North Africa:


```python
data_war         = pd.read_csv('deaths-conflict-terrorism-per-100000.csv')
data_war         = data_war[data_war['Entity'] == 'Middle East & North Africa']
data_war         = data_war[['Year','Deaths - Conflict and terrorism - Sex: Both - Age: All Ages (Rate)']]
data_war.rename(columns={"Year": "Date", "Deaths - Conflict and terrorism - Sex: Both - Age: All Ages (Rate)": "Deaths"},inplace=True)
data_war         = data_war.set_index('Date')
data_war.index   = pd.to_datetime(data_war.index,format='%Y')
data_war         = data_war[(data_war.index >= '2010-01-01') & (data_war.index <= '2021-01-01')]
data_war.head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Deaths</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010-01-01</th>
      <td>2.241973</td>
    </tr>
    <tr>
      <th>2011-01-01</th>
      <td>7.131705</td>
    </tr>
    <tr>
      <th>2012-01-01</th>
      <td>17.841588</td>
    </tr>
    <tr>
      <th>2013-01-01</th>
      <td>16.821756</td>
    </tr>
    <tr>
      <th>2014-01-01</th>
      <td>28.672619</td>
    </tr>
    <tr>
      <th>2015-01-01</th>
      <td>23.710162</td>
    </tr>
    <tr>
      <th>2016-01-01</th>
      <td>24.224164</td>
    </tr>
    <tr>
      <th>2017-01-01</th>
      <td>17.838126</td>
    </tr>
    <tr>
      <th>2018-01-01</th>
      <td>10.961725</td>
    </tr>
    <tr>
      <th>2019-01-01</th>
      <td>5.222731</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_wti2       = df_wti
wti_war = df_wti2.merge(data_war,how='outer',left_index=True,right_index=True)
wti_war
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>F1</th>
      <th>F2</th>
      <th>F3</th>
      <th>F4</th>
      <th>Execution_day</th>
      <th>Holding_type</th>
      <th>P/L_F1</th>
      <th>P/L_F2</th>
      <th>Cumulative_P/L_barrel</th>
      <th>Cumulative_P/L_strat</th>
      <th>...</th>
      <th>P/L</th>
      <th>P/L_short</th>
      <th>C(0)</th>
      <th>P/L_C(0)</th>
      <th>Cumulative_P/L_C(0)</th>
      <th>P(t)_C(0)</th>
      <th>dF(t)_C(0)</th>
      <th>HWM_C(0)</th>
      <th>DD_C(0)</th>
      <th>Deaths</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010-01-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.241973</td>
    </tr>
    <tr>
      <th>2010-01-04</th>
      <td>81.51</td>
      <td>82.12</td>
      <td>82.65</td>
      <td>83.12</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.15</td>
      <td>2.10</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>-1.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2010-01-05</th>
      <td>81.77</td>
      <td>82.41</td>
      <td>82.99</td>
      <td>83.52</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.26</td>
      <td>0.29</td>
      <td>0.26</td>
      <td>26000.0</td>
      <td>...</td>
      <td>0.26</td>
      <td>-0.26</td>
      <td>-1.0</td>
      <td>-0.26</td>
      <td>-0.26</td>
      <td>-26000.0</td>
      <td>-26000.0</td>
      <td>0.0</td>
      <td>26000.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2010-01-06</th>
      <td>83.18</td>
      <td>83.75</td>
      <td>84.31</td>
      <td>84.86</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.41</td>
      <td>1.34</td>
      <td>1.67</td>
      <td>167000.0</td>
      <td>...</td>
      <td>1.41</td>
      <td>-1.41</td>
      <td>-1.0</td>
      <td>-1.41</td>
      <td>-1.67</td>
      <td>-167000.0</td>
      <td>-141000.0</td>
      <td>0.0</td>
      <td>167000.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2010-01-07</th>
      <td>82.66</td>
      <td>83.19</td>
      <td>83.75</td>
      <td>84.29</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.52</td>
      <td>-0.56</td>
      <td>1.15</td>
      <td>115000.0</td>
      <td>...</td>
      <td>-0.52</td>
      <td>0.52</td>
      <td>-1.0</td>
      <td>0.52</td>
      <td>-1.15</td>
      <td>-115000.0</td>
      <td>52000.0</td>
      <td>0.0</td>
      <td>115000.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2021-12-27</th>
      <td>75.57</td>
      <td>75.18</td>
      <td>74.67</td>
      <td>74.12</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.78</td>
      <td>1.76</td>
      <td>-63.52</td>
      <td>-6352000.0</td>
      <td>...</td>
      <td>1.78</td>
      <td>-1.78</td>
      <td>-1.0</td>
      <td>-1.78</td>
      <td>108.78</td>
      <td>10878000.0</td>
      <td>-178000.0</td>
      <td>15367000.0</td>
      <td>4489000.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2021-12-28</th>
      <td>75.98</td>
      <td>75.60</td>
      <td>75.13</td>
      <td>74.59</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.41</td>
      <td>0.42</td>
      <td>-63.11</td>
      <td>-6311000.0</td>
      <td>...</td>
      <td>0.41</td>
      <td>-0.41</td>
      <td>-1.0</td>
      <td>-0.41</td>
      <td>108.37</td>
      <td>10837000.0</td>
      <td>-41000.0</td>
      <td>15367000.0</td>
      <td>4530000.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2021-12-29</th>
      <td>76.56</td>
      <td>76.18</td>
      <td>75.71</td>
      <td>75.18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.58</td>
      <td>0.58</td>
      <td>-62.53</td>
      <td>-6253000.0</td>
      <td>...</td>
      <td>0.58</td>
      <td>-0.58</td>
      <td>-1.0</td>
      <td>-0.58</td>
      <td>107.79</td>
      <td>10779000.0</td>
      <td>-58000.0</td>
      <td>15367000.0</td>
      <td>4588000.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2021-12-30</th>
      <td>76.99</td>
      <td>76.61</td>
      <td>76.13</td>
      <td>75.58</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.43</td>
      <td>0.43</td>
      <td>-62.10</td>
      <td>-6210000.0</td>
      <td>...</td>
      <td>0.43</td>
      <td>-0.43</td>
      <td>-1.0</td>
      <td>-0.43</td>
      <td>107.36</td>
      <td>10736000.0</td>
      <td>-43000.0</td>
      <td>15367000.0</td>
      <td>4631000.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2021-12-31</th>
      <td>75.21</td>
      <td>74.88</td>
      <td>74.45</td>
      <td>73.94</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.78</td>
      <td>-1.73</td>
      <td>-63.88</td>
      <td>-6388000.0</td>
      <td>...</td>
      <td>-1.78</td>
      <td>1.78</td>
      <td>1.0</td>
      <td>1.76</td>
      <td>109.12</td>
      <td>10912000.0</td>
      <td>176000.0</td>
      <td>15367000.0</td>
      <td>4455000.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>3032 rows × 21 columns</p>
</div>



### Interpolate the death count:


```python
wti_war['Deaths'].interpolate(inplace=True)
wti_war = wti_war.loc['2010-01-04':'2019-12-31']
wti_war
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>F1</th>
      <th>F2</th>
      <th>F3</th>
      <th>F4</th>
      <th>Execution_day</th>
      <th>Holding_type</th>
      <th>P/L_F1</th>
      <th>P/L_F2</th>
      <th>Cumulative_P/L_barrel</th>
      <th>Cumulative_P/L_strat</th>
      <th>...</th>
      <th>P/L</th>
      <th>P/L_short</th>
      <th>C(0)</th>
      <th>P/L_C(0)</th>
      <th>Cumulative_P/L_C(0)</th>
      <th>P(t)_C(0)</th>
      <th>dF(t)_C(0)</th>
      <th>HWM_C(0)</th>
      <th>DD_C(0)</th>
      <th>Deaths</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010-01-04</th>
      <td>81.51</td>
      <td>82.12</td>
      <td>82.65</td>
      <td>83.12</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.15</td>
      <td>2.10</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>-1.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.261300</td>
    </tr>
    <tr>
      <th>2010-01-05</th>
      <td>81.77</td>
      <td>82.41</td>
      <td>82.99</td>
      <td>83.52</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.26</td>
      <td>0.29</td>
      <td>0.26</td>
      <td>26000.0</td>
      <td>...</td>
      <td>0.26</td>
      <td>-0.26</td>
      <td>-1.0</td>
      <td>-0.26</td>
      <td>-0.26</td>
      <td>-26000.0</td>
      <td>-26000.0</td>
      <td>0.0</td>
      <td>26000.0</td>
      <td>2.280627</td>
    </tr>
    <tr>
      <th>2010-01-06</th>
      <td>83.18</td>
      <td>83.75</td>
      <td>84.31</td>
      <td>84.86</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.41</td>
      <td>1.34</td>
      <td>1.67</td>
      <td>167000.0</td>
      <td>...</td>
      <td>1.41</td>
      <td>-1.41</td>
      <td>-1.0</td>
      <td>-1.41</td>
      <td>-1.67</td>
      <td>-167000.0</td>
      <td>-141000.0</td>
      <td>0.0</td>
      <td>167000.0</td>
      <td>2.299954</td>
    </tr>
    <tr>
      <th>2010-01-07</th>
      <td>82.66</td>
      <td>83.19</td>
      <td>83.75</td>
      <td>84.29</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.52</td>
      <td>-0.56</td>
      <td>1.15</td>
      <td>115000.0</td>
      <td>...</td>
      <td>-0.52</td>
      <td>0.52</td>
      <td>-1.0</td>
      <td>0.52</td>
      <td>-1.15</td>
      <td>-115000.0</td>
      <td>52000.0</td>
      <td>0.0</td>
      <td>115000.0</td>
      <td>2.319281</td>
    </tr>
    <tr>
      <th>2010-01-08</th>
      <td>82.75</td>
      <td>83.30</td>
      <td>83.87</td>
      <td>84.47</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.09</td>
      <td>0.11</td>
      <td>1.24</td>
      <td>124000.0</td>
      <td>...</td>
      <td>0.09</td>
      <td>-0.09</td>
      <td>-1.0</td>
      <td>-0.09</td>
      <td>-1.24</td>
      <td>-124000.0</td>
      <td>-9000.0</td>
      <td>0.0</td>
      <td>124000.0</td>
      <td>2.338608</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2019-12-24</th>
      <td>61.11</td>
      <td>60.94</td>
      <td>60.65</td>
      <td>60.26</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.59</td>
      <td>0.59</td>
      <td>-70.37</td>
      <td>-7037000.0</td>
      <td>...</td>
      <td>0.59</td>
      <td>-0.59</td>
      <td>1.0</td>
      <td>-0.61</td>
      <td>104.19</td>
      <td>10419000.0</td>
      <td>-61000.0</td>
      <td>10703000.0</td>
      <td>284000.0</td>
      <td>5.222731</td>
    </tr>
    <tr>
      <th>2019-12-26</th>
      <td>61.68</td>
      <td>61.48</td>
      <td>61.16</td>
      <td>60.73</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.57</td>
      <td>0.54</td>
      <td>-69.80</td>
      <td>-6980000.0</td>
      <td>...</td>
      <td>0.57</td>
      <td>-0.57</td>
      <td>-1.0</td>
      <td>0.55</td>
      <td>104.74</td>
      <td>10474000.0</td>
      <td>55000.0</td>
      <td>10703000.0</td>
      <td>229000.0</td>
      <td>5.222731</td>
    </tr>
    <tr>
      <th>2019-12-27</th>
      <td>61.72</td>
      <td>61.53</td>
      <td>61.21</td>
      <td>60.79</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.04</td>
      <td>0.05</td>
      <td>-69.76</td>
      <td>-6976000.0</td>
      <td>...</td>
      <td>0.04</td>
      <td>-0.04</td>
      <td>-1.0</td>
      <td>-0.04</td>
      <td>104.70</td>
      <td>10470000.0</td>
      <td>-4000.0</td>
      <td>10703000.0</td>
      <td>233000.0</td>
      <td>5.222731</td>
    </tr>
    <tr>
      <th>2019-12-30</th>
      <td>61.68</td>
      <td>61.44</td>
      <td>61.10</td>
      <td>60.66</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.04</td>
      <td>-0.09</td>
      <td>-69.80</td>
      <td>-6980000.0</td>
      <td>...</td>
      <td>-0.04</td>
      <td>0.04</td>
      <td>-1.0</td>
      <td>0.04</td>
      <td>104.74</td>
      <td>10474000.0</td>
      <td>4000.0</td>
      <td>10703000.0</td>
      <td>229000.0</td>
      <td>5.222731</td>
    </tr>
    <tr>
      <th>2019-12-31</th>
      <td>61.06</td>
      <td>60.77</td>
      <td>60.41</td>
      <td>59.97</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.62</td>
      <td>-0.67</td>
      <td>-70.42</td>
      <td>-7042000.0</td>
      <td>...</td>
      <td>-0.62</td>
      <td>0.62</td>
      <td>-1.0</td>
      <td>0.62</td>
      <td>105.36</td>
      <td>10536000.0</td>
      <td>62000.0</td>
      <td>10703000.0</td>
      <td>167000.0</td>
      <td>5.222731</td>
    </tr>
  </tbody>
</table>
<p>2528 rows × 21 columns</p>
</div>




```python
wti_war['Deaths'].dtypes
```




    dtype('float64')




```python
plt.plot(wti_war['Deaths'])
```




    [<matplotlib.lines.Line2D at 0x7ff9c30664e0>]




![png](https://github.com/arjunkalsi/arjunkalsi.github.io/tree/master/img/wti/output_75_1.png)


### The strategy is to hold nothing when the deaths are above the threshold. We initially did this with no time lag  we decided to add a time lag of approximately 1 year as acts of war and terror usually occur in succession:


```python
wti_war.dropna(inplace=True)
thresholds = [25.0,24.0,23.0,22.0,21.0]
SRoptimal = 0.0
Topt = 0.0
epsilon = 0

for t in thresholds:
    #print(t,":")
    #print("")

    wti_war['C(0)']  = None
    for i in range(len(wti_war)):

        if wti_war['C(0)'][i] == 0.0:
            continue
        if wti_war['Deaths'][i] > t:
            wti_war['C(0)'][i+250] = 0.0

        if wti_war['F1 - Predicted'][i] > epsilon:
                wti_war['C(0)'][i] = -1.0
        elif wti_war['F1 - Predicted'][i] < epsilon:
                wti_war['C(0)'][i] = 1.0

    wti_war['P/L_C(0)'] = 0.0
    wti_war['Cumulative_P/L_C(0)'] = 0.0

    # deciding whether your P&L is from the long or short position
    for i in range(1,len(wti_war)):
        if (wti_war['C(0)'][i-1] == 0.0):
            wti_war['P/L_C(0)'][i] = 0.0

        elif (wti_war['C(0)'][i-1] == 1.0):
            wti_war['P/L_C(0)'][i] = wti_war['P/L'][i]

        elif (wti_war['C(0)'][i-1] == -1.0):
            wti_war['P/L_C(0)'][i] = wti_war['P/L_short'][i]


    # if it is time to switch positions you have to add the transaction costs
    for i in range(1,len(wti_war)):
        if wti_war['C(0)'][i] != wti_war['C(0)'][i-1]:
            wti_war['P/L_C(0)'][i] -= 0.02

    # calculating cumulative P&L
    for i in range(1,len(wti_war)):
        wti_war['Cumulative_P/L_C(0)'][i] = wti_war['P/L_C(0)'][i] + wti_war['Cumulative_P/L_C(0)'][i-1]

    wti_war['P(t)_C(0)'] = 0.0

    for i in range(1,len(wti_war)):
        wti_war['P(t)_C(0)'][i] = wti_war['P(t)_C(0)'][i-1] +(wti_war['P/L_C(0)'][i]*100000)

    wti_war['dF(t)_C(0)'] = wti_war['P/L_C(0)']*100000

    ann_PL_C = wti_war['P(t)_C(0)'][-1]/N
    #print(f"C(0) Average Annual P&L: {ann_PL_C}")

    ann_SR_C = ann_PL_C/(np.std(wti_war['dF(t)_C(0)'])*np.sqrt(250))
    #print(f"C(0) Annualised Sharpe Ratio: {ann_SR_C}")

    wti_war['HWM_C(0)'] = 0.0
    wti_war['DD_C(0)'] = 0.0

    for i in range(len(wti_war)):
        wti_war['HWM_C(0)'][i] = np.max(wti_war['P(t)_C(0)'][0:i+1])
        wti_war['DD_C(0)'][i] = wti_war['HWM_C(0)'][i] - wti_war['P(t)_C(0)'][i]

    mdd_C = np.max(wti_war['DD_C(0)'])
    mdd_Cdate = wti_war['DD_C(0)'].idxmax()

    #print(f"C(0) Maximum Drawdown: {mdd_C} at {mdd_Cdate}")
    #print("")

    if ann_SR_C > SRoptimal:
        SRoptimal = ann_SR_C
        Topt = t
```

### Now our optimal results are with a threshold of 21:

- C(0) Average Annual P&L: 1164000.0000000016
- C(0) Annualised Sharpe Ratio: 0.6115587914972023
- C(0) Maximum Drawdown: 3742999.999999998 at 2011-08-09 00:00:00

### Let's also graph the equity line below:


```python
plt.figure(figsize=(10, 6), dpi=80)
plt.ticklabel_format(style='plain')
plt.plot(wti_war.index, wti_war['P(t)_C(0)'], label = 'C(0.01) Equity Line')
plt.plot(wti_war.index, -wti_war['DD_C(0)'], label = 'C(0.01) Drawdown',alpha=0.5)
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.savefig('EL + drawdown and geopolitical risk reduction')
plt.show()
```


![png](https://github.com/arjunkalsi/arjunkalsi.github.io/tree/master/img/wti/output_79_0.png)


### Although the war-based strategy is obviously not the most robust and may be completely random, we can see that by assigning a threshold to terminate trading we have considerably reduced maximum drawdown and thus reduced risk and improved P&L. We have also improved the Sharpe ratio by so this experiment was surprisingly successful. By sampling over a larger range we may be able to see more reliable patterns, but we may not have spent enough time in a globalising economy to notice patterns/time-lags like these.

### There is clearly some lookahead bias here, but the project makes for a fun experiment and with better access to war data we may be able to predict fluctuations in price to a larger extent.


```python

```
