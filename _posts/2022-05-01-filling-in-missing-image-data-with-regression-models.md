---
layout: post
title: Filling In Missing Image Data With Regression Models
subtitle: We test 3 different models for missing data filling, and see which performs best.
tags: [math, tech]
---

### Today we're looking at 3 different methods for filling missing image data - a Random Forest Regressor, a Gradient Boosting Regressor, and a 2-D Gaussian kernel detailed below. Let's see which one performs best at filling missing data! This is the picture we are filling in:


![png](https://raw.githubusercontent.com/arjunkalsi/arjunkalsi.github.io/master/img/imgdata/output_1_0.png)

There are many missing values. We first Construct $y_{train}$, $X_{train}$ consisting of all available non-N/A pixel values and their positions. Let $X_{test}$ be the positions of the missing pixels whose values you are trying to reconstruct.

Next, using each of the following methods:

• RandomForestRegressor

• GradientBoostingRegressor

• A 2−D Gaussian kernel feature map regression, where each of your feature maps is φσ(x, x0) = exp(−|x−x0|^2/2σ^2) and you have rescaled your image pixel locations so that x, x0 ∈ [0, 1]^2


```python
y_train = []
y_test  = []
X_train = []
X_test  = []
y_test_actual = []


for i in range(image_ss.shape[0]):
    for j in range(image_ss.shape[1]):
        if not np.isnan(image_ss[i][j]):
            y_train.append(image_ss[i][j])
            X_train.append([i,j])
        else:
            X_test.append([i,j])
            y_test_actual.append(image[i][j])


def extractDigits(lst):
    return [[el] for el in lst]


y_test_actual = np.array(extractDigits(y_test_actual))
```

Now that we have our data sets sorted out, we can test each technique:

### 1. RandomForestRegressor


```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

model = RandomForestRegressor()

gs = GridSearchCV(model,
                  param_grid = {'max_depth': range(1, 11),
                                'min_samples_split': range(10, 60, 10)},
                  cv=5,
                  n_jobs=1,
                  scoring='neg_mean_squared_error')

gs.fit(X_train,y_train)

print(gs.best_params_)
print(-gs.best_score_)
```

    {'max_depth': 10, 'min_samples_split': 20}
    4418.159874196007


```python
rfModel = RandomForestRegressor(max_depth=10,min_samples_split=20)
rfModel.fit(X_train,y_train)
```



    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=10,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=20,
               min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,
               oob_score=False, random_state=None, verbose=0, warm_start=False)



```python
for i in range(image_ss.shape[0]):
    for j in range(image_ss.shape[1]):
        if np.isnan(image_ss[i][j]):
            y = rfModel.predict([[i,j]])
            X_test.append([i,j])
            y_test.append(y)

MSErf = mean_squared_error(y_test,y_test_actual)
MSErf
```


    674.9260644662936

This is the MSE of our RFR method.

```python
image_ss_complete = np.zeros([254,240])

for i in range(254):
    for j in range(240):
        if [i,j] in X_train:
            pos = X_train.index([i,j])
            image_ss_complete[i][j] = y_train[pos]

        elif [i,j] in X_test:
            pos2 = X_test.index([i,j])
            image_ss_complete[i][j] = y_test[pos2]

plt.imshow(image_ss_complete, cmap='gray', vmin=0, vmax=255)
plt.show()
```


![png](https://raw.githubusercontent.com/arjunkalsi/arjunkalsi.github.io/master/img/imgdata/output_16_0.png)


### 2. GradientBoostingRegressor


```python
y_test2 = []

model2 = GradientBoostingRegressor()

gs2 = GridSearchCV(model2,
                  param_grid = {'max_depth': [3,5,10],
                                'learning_rate':[0.1,0.3,0.5],
                                'n_estimators':[50,75,100]},
                  cv=5,
                  n_jobs=1,
                  scoring='neg_mean_squared_error')

gs2.fit(X_train,y_train)

print(gs2.best_params_)
print(-gs2.best_score_)
```

    {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}
    4154.170911663076


```python
gbModel = GradientBoostingRegressor(learning_rate=0.1,max_depth=5,n_estimators=100)
gbModel.fit(X_train,y_train)
```



    GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.1, loss='ls', max_depth=5, max_features=None,
                 max_leaf_nodes=None, min_impurity_decrease=0.0,
                 min_impurity_split=None, min_samples_leaf=1,
                 min_samples_split=2, min_weight_fraction_leaf=0.0,
                 n_estimators=100, n_iter_no_change=None, presort='auto',
                 random_state=None, subsample=1.0, tol=0.0001,
                 validation_fraction=0.1, verbose=0, warm_start=False)



```python
X_test2 = []

for i in range(image_ss.shape[0]):
    for j in range(image_ss.shape[1]):
        if np.isnan(image_ss[i][j]):
            y = gbModel.predict([[i,j]])
            X_test2.append([i,j])
            y_test2.append(y)

MSEgb = mean_squared_error(y_test2,y_test_actual)
MSEgb
```


    807.2361077570746

This is the MSE of our GBR method.

```python
image_ss_complete2 = np.zeros([254,240])

for i in range(254):
    for j in range(240):
        if [i,j] in X_train:
            pos = X_train.index([i,j])
            image_ss_complete2[i][j] = y_train[pos]

        elif [i,j] in X_test2:
            pos2 = X_test2.index([i,j])
            image_ss_complete2[i][j] = y_test2[pos2]

plt.imshow(image_ss_complete2, cmap='gray', vmin=0, vmax=255)
plt.show()
```


![png](https://raw.githubusercontent.com/arjunkalsi/arjunkalsi.github.io/master/img/imgdata/output_23_0.png)


### 3. 2-D Gaussian kernel FM Regressor


```python
from sklearn.model_selection import cross_val_score
from numpy.linalg import norm
from sklearn.linear_model import LinearRegression

X_train_np = np.array(X_train)
X_test_np  = np.array(X_test)
y_train_np = np.array(extractDigits(y_train))

def gauss(x,x_0,sig):
    return np.exp((-norm(x-x_0)**2) / sig**2)


x_0  = np.array([125,125]) # roughly in the middle
sigs = [0.25,0.5,0.75,1,2,3,4,5,6,7,8,9,10,15,30,50]
lowestMSE = np.inf

for sig in sigs:
    gauss_X_train = np.array([gauss(x,x_0,sig) for x in X_train_np])
    gauss_X_train = gauss_X_train.reshape((48658,1))
    scores = cross_val_score(LinearRegression(),gauss_X_train,y_train_np,cv=5,scoring='neg_mean_squared_error')
    print(f'sigma = {sig}')
    print(f'MSE = {-scores.mean()}')
    if (-scores.mean() < lowestMSE):
        lowestMSE = -scores.mean()

```

    sigma = 0.25
    MSE = 9504.965373549849
    sigma = 0.5
    MSE = 9504.96278395924
    sigma = 0.75
    MSE = 9504.935933001057
    sigma = 1
    MSE = inf
    sigma = 2
    MSE = 4.988033674055313e+125
    sigma = 3
    MSE = 3.163547226148574e+56
    sigma = 4
    MSE = 2.820246266433333e+32
    sigma = 5
    MSE = 2.8633232032913005e+21
    sigma = 6
    MSE = 3872029897845332.0
    sigma = 7
    MSE = 1353413061999.9172
    sigma = 8
    MSE = 8911156102.413242
    sigma = 9
    MSE = 312207516.31875813
    sigma = 10
    MSE = 29928598.96471374
    sigma = 15
    MSE = 140776.89363281996
    sigma = 30
    MSE = 12702.314638027401
    sigma = 50
    MSE = 7643.011517213762



The lowest MSE for this method is 7643.01 (not very low) when sigma = 50.
Let's use this on the test set:



```python
lin = LinearRegression()

sig = 50

gauss_X_train = np.array([gauss(x,x_0,sig) for x in X_train_np])
gauss_X_train = gauss_X_train.reshape((X_train_np.shape[0],1))

lin.fit(gauss_X_train,y_train_np)

gauss_X_test = np.array([gauss(x,x_0,sig) for x in X_test_np])
gauss_X_test = gauss_X_test.reshape((X_test_np.shape[0],1))

gauss_y_test = lin.predict(gauss_X_test)
MSE = mean_squared_error(gauss_y_test,y_test_actual)
MSE
```



    5798.050017820173

The MSE for this method is still pretty high. Let's see what it looks like:

```python
image_ss_gauss = image_ss.copy()

index = 0
for i in range(0,image_ss_gauss.shape[0]):
    for j in range(0,image_ss_gauss.shape[1]):
        if np.isnan(image_ss_gauss[i][j]):
            image_ss_gauss[i][j] = gauss_y_test[index]
            index += 1

plt.imshow(image_ss_gauss, cmap='gray', vmin=0, vmax=255)
plt.show()
```


![png](https://raw.githubusercontent.com/arjunkalsi/arjunkalsi.github.io/master/img/imgdata/output_37_0.png)


### We can see that the Random Forest Regressor looks like it's performing the best here as it has the lowest MSE. However, Gradient Boosting looks better and  I think this may be due to the inspection of more optimal parameters in the cross-validation stage, allowing the predict function to predict better parameters.

Seperately, I investigated the effect of adding an L1 − L2 elastic net regularization to the 2−D Gaussian kernel feature map regression method and found the optimal parameters of my model using cross-validation, but I omitted this here.

Another way to improve our model is to add new features as follows. Instead of $y = f(i, j)$, we could also consider the model $y = f(i, j, y_{up}, y_{dn}, y_{l}, y_{r})$ where the $y$ values are the pixel values, if available, directly above, below, to the left and to the right of the target pixel at i, j. Use −999 to denote a missing value pixel:

```python
X_train_yu = []
X_train_yd = []
X_train_yl = []
X_train_yr = []

X_test_yu = []
X_test_yd = []
X_test_yl = []
X_test_yr = []

X_traini = []
X_trainj = []
X_testi  = []
X_testj  = []

for i in range(0,image_ss.shape[0]):
    for j in range(0,image_ss.shape[1]):

        if np.isnan(image_ss[i][j]):

            X_testi.append(i)
            X_testj.append(j)

            # must have a try here otherwise sometimes it returns errors at the boundaries
            # now if it's out of bounds it will just add -999 anyway
            # up
            try:
                if i==0:
                    X_test_yu.append(-999)

                elif np.isnan(image_ss[i-1][j]):
                        X_test_yu.append(-999)

                else:
                    X_test_yu.append(image_ss[i-1][j])

            except IndexError as e:
                X_test_yu.append(-999)

            # down
            try:
                if i==image_ss.shape[0]:
                    X_test_yd.append(-999)

                elif np.isnan(image_ss[i+1][j]):
                        X_test_yd.append(-999)

                else:
                    X_test_yd.append(image_ss[i+1][j])

            except IndexError as e:
                X_test_yd.append(-999)

            # left
            try:
                if j==0:
                    X_test_yl.append(-999)

                elif np.isnan(image_ss[i][j-1]):
                        X_test_yl.append(-999)

                else:
                    X_test_yl.append(image_ss[i][j-1])

            except IndexError as e:
                X_test_yl.append(-999)

            # right
            try:
                if j==image_ss.shape[1]:
                    X_test_yr.append(-999)

                elif np.isnan(image_ss[i][j+1]):
                        X_test_yr.append(-999)

                else:
                    X_test_yr.append(image_ss[i][j+1])

            except IndexError as e:
                X_test_yr.append(-999)

        else:

            X_traini.append(i)
            X_trainj.append(j)

            # up
            try:
                if i==0:
                    X_train_yu.append(-999)

                elif np.isnan(image_ss[i-1][j]):
                        X_train_yu.append(-999)

                else:
                    X_train_yu.append(image_ss[i-1][j])

            except IndexError as e:
                X_train_yu.append(-999)

            # down
            try:
                if i==image_ss.shape[0]:
                    X_train_yd.append(-999)

                elif np.isnan(image_ss[i+1][j]):
                        X_train_yd.append(-999)

                else:
                    X_train_yd.append(image_ss[i+1][j])

            except IndexError as e:
                X_train_yd.append(-999)

            # left
            try:
                if j==0:
                    X_train_yl.append(-999)

                elif np.isnan(image_ss[i][j-1]):
                        X_train_yl.append(-999)

                else:
                    X_train_yl.append(image_ss[i][j-1])

            except IndexError as e:
                X_train_yl.append(-999)

            # right
            try:
                if j==image_ss.shape[1]:
                    X_train_yr.append(-999)

                elif np.isnan(image_ss[i][j+1]):
                        X_train_yr.append(-999)

                else:
                    X_train_yr.append(image_ss[i][j+1])

            except IndexError as e:
                X_train_yr.append(-999)

```

Here's what the dataframe looks like:

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
      <th>i</th>
      <th>j</th>
      <th>up</th>
      <th>down</th>
      <th>left</th>
      <th>right</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>-999.0</td>
      <td>103.0</td>
      <td>-999.0</td>
      <td>89.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>-999.0</td>
      <td>90.0</td>
      <td>84.0</td>
      <td>98.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>89.0</td>
      <td>87.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3</td>
      <td>-999.0</td>
      <td>87.0</td>
      <td>98.0</td>
      <td>-999.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>5</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>82.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>48653</th>
      <td>253</td>
      <td>233</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>56.0</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>48654</th>
      <td>253</td>
      <td>234</td>
      <td>31.0</td>
      <td>-999.0</td>
      <td>44.0</td>
      <td>-999.0</td>
    </tr>
    <tr>
      <th>48655</th>
      <td>253</td>
      <td>236</td>
      <td>32.0</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>48656</th>
      <td>253</td>
      <td>237</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>25.0</td>
      <td>-999.0</td>
    </tr>
    <tr>
      <th>48657</th>
      <td>253</td>
      <td>239</td>
      <td>44.0</td>
      <td>-999.0</td>
      <td>-999.0</td>
      <td>-999.0</td>
    </tr>
  </tbody>
</table>
<p>48658 rows × 6 columns</p>
</div>

Now let's try each of our methods again.

### RandomForestRegressor with ys:


```python
from sklearn.model_selection import GridSearchCV

model = RandomForestRegressor()

gs = GridSearchCV(model,
                  param_grid = {'max_depth': range(1, 11),
                                'min_samples_split': range(10, 60, 10)},
                  cv=5,
                  n_jobs=1,
                  scoring='neg_mean_squared_error')

gs.fit(df1,y_train_np)

rfModel = RandomForestRegressor(max_depth=10,min_samples_split=20)
rfModel.fit(df1,y_train_np)

X_test2 = []
y_test2 = []

y_test2 = rfModel.predict(df2)
MSE = mean_squared_error(y_test_actual,y_test2)
MSE
```



    187.91561207966419

(much lower MSE)

```python
image_ss_complete3 = image_ss.copy()

index = 0
for i in range(image_ss_complete3.shape[0]):
    for j in range(image_ss_complete3.shape[1]):
        if np.isnan(image_ss_complete3[i][j]):
            image_ss_complete3[i][j] = y_test2[index]
            index += 1

plt.imshow(image_ss_complete3, cmap='gray', vmin=0, vmax=255)
plt.show()
```


![png](https://raw.githubusercontent.com/arjunkalsi/arjunkalsi.github.io/master/img/imgdata/output_65_0.png)


### GradientBoostingRegressor with ys


```python
from sklearn.model_selection import GridSearchCV

model2 = GradientBoostingRegressor()

gs = GridSearchCV(model2,
                  param_grid = {'max_depth': [3,5,10],
                                'learning_rate':[0.1,0.3,0.5],
                                'n_estimators':[50,75,100]},
                  cv=5,
                  n_jobs=1,
                  scoring='neg_mean_squared_error')

gs.fit(df1,y_train_np)

gbModel = GradientBoostingRegressor(max_depth=5,learning_rate=0.1,n_estimators=100)
gbModel.fit(df1,y_train_np)

X_test3 = []
y_test3 = []

y_test3 = gbModel.predict(df2)
MSE = mean_squared_error(y_test_actual,y_test3)
MSE
```



    161.93799834274037



```python
image_ss_complete4 = image_ss.copy()

index = 0
for i in range(image_ss_complete4.shape[0]):
    for j in range(image_ss_complete4.shape[1]):
        if np.isnan(image_ss_complete4[i][j]):
            image_ss_complete4[i][j] = y_test3[index]
            index += 1

plt.imshow(image_ss_complete4, cmap='gray', vmin=0, vmax=255)
plt.show()
```


![png](https://raw.githubusercontent.com/arjunkalsi/arjunkalsi.github.io/master/img/imgdata/output_72_0.png)


### We can see the GBR performs really well with this method. We could go even further and add diagonal adjacent values to improve our modelling method, or even pixels that are 2 away from the pixel in focus etc. Either way a GBR may be preferable to a RFR method based on our results here, however you probably know that a deep learning method is likely to perform even better than both, especially in light of the current wave of visual work innovation that is occurring at the moment. Either way thanks for reading and happy image filling! 
