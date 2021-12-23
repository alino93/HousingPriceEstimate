# identify best regression model
import numpy as np
import pandas as pd
from sklearn import datasets
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import median_absolute_error
from statistics import median
import matplotlib.pyplot as plt
import pickle
from keras.preprocessing.image import load_img

data = pd.read_excel('text_datas.xlsx')
data.head(3)

# create a column for age = date - yr_built. Drop old data.
data["Age"] = data['Sell Date'].dt.year - data['age']

f = open('level_cnn8.pckl', 'rb')
level = pickle.load(f).astype('int32')
level = level.reshape((-1))
#level = np.hstack((level,np.array((5,5,5))))
f.close()

data["lux_level"] = level
data=data.drop('Sell Date', axis=1)
data=data.drop('age', axis=1)
data=data.drop('Unnamed: 0', axis=1)
data=data.drop('heating', axis=1)
data=data.replace('Wall',1)
data=data.replace('Refrigerator Central',1)
data=data.replace('Central',1)
data=data.replace('Central Other',1)
data=data.replace('Refrigerator',1)
data=data.replace('Geothermal Other',1)
data=data.replace('Central Geothermal Other',1)
data=data.replace('Central Geothermal Solar',1)
data=data.replace('None',0)
data=data.replace('No Data',0)
data=data.replace('Other',0)

data.head(5)

print(data.shape)
print(data.dtypes)

pd.set_option('precision', 2)
print(data.describe())

# identifying the top direct correlation with price.
correlation = data.corr(method='pearson')
columns = correlation.nlargest(10, 'Price').index

correlation_map = np.corrcoef(data[columns].values.T)
sns.set(font_scale=1.0)
heatmap = sns.heatmap(correlation_map, cbar=True, annot=True, square=True, fmt='.2f', yticklabels=columns.values, xticklabels=columns.values)

plt.show()

# normalise the data using log.
data['Price'] = np.log(data['Price'])
#data['sqft'] = np.log(data['sqft'])
#data['lux level'] = np.log(data['lux level'])


# Test a few regression algorithms using negative mean square error
X = data[columns]
Y = X['Price'].values
X = X.drop('Price', axis = 1).values

X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.20, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Standardised data to have zero mean value and 1 standard deviation using pipelines.
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
pipelines.append(('ScaledDT', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))
pipelines.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))

results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=10, random_state=21)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='neg_mean_squared_error')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# From above, Gradient Boosting model is chosen as the best

# Test with a few n_estimators using the GridSearchCV function.
from sklearn.model_selection import GridSearchCV

scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = dict(n_estimators=np.array([50,100,200,300,400]))
model = GradientBoostingRegressor(random_state=21)
kfold = KFold(n_splits=10, random_state=21)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# choose best n_estimator configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


# standardise data
from sklearn.metrics import mean_squared_error

scaler = StandardScaler().fit(X_train)
rescaled_X_train = scaler.transform(X_train)
model = GradientBoostingRegressor(random_state=21, n_estimators=50)
model.fit(rescaled_X_train, Y_train)

# transform the valid data
rescaled_X_test = scaler.transform(X_test)
predictions = model.predict(rescaled_X_test)
print (mean_squared_error(Y_test, predictions))

# check the difference between the predicted value and test
compare = pd.DataFrame({'Prediction': predictions, 'Test Data' : Y_test})

# inverse_transform and exp the data
actual_y_test = np.exp(Y_test)
actual_predicted = np.exp(predictions)
diff = abs(actual_y_test - actual_predicted)
error_percent = diff/actual_y_test

compare_actual = pd.DataFrame({'Test Data': actual_y_test, 'Predicted Price' : actual_predicted,
                               'Difference' : diff,'Percent Error':error_percent})
#compare_actual = compare_actual.astype(int)
compare_actual.to_pickle("./test_price_pred.pkl")

#def cross_val(model):
#    pred = cross_val_score(model, X, y, cv=10)
#    return pred.mean()


def print_evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    median_err = median_absolute_error(true, predicted)
    median_err_rate = median(np.abs(true-predicted)/true)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('Median error:', median_err)
    print('Median error rate:', median_err_rate)

def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    median_err_rate = median(np.abs(true - predicted) / true)
    return mae, mse, rmse, r2_square, median_err_rate


# Compare errors
test_pred = model.predict(rescaled_X_test)
train_pred = model.predict(rescaled_X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(np.exp(Y_test), np.exp(test_pred))
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate(np.exp(Y_train), np.exp(train_pred))

#results_df = pd.DataFrame(data=[["KNN", *evaluate(np.exp(Y_train), np.exp(train_pred))]], columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Median Err Rate"])
#results_df.to_pickle("./res.pkl")
#results_df = pd.read_pickle("./res.pkl")
#results_df2 = pd.DataFrame(data=[["SVM", *evaluate(np.exp(Y_train), np.exp(train_pred))]], columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Median Err Rate"])
#results_df = results_df.append(results_df2, ignore_index=True)
#results_df.to_pickle("./res.pkl")
#results_df = pd.read_pickle("./res.pkl")
#results_df3 = pd.DataFrame(data=[["Only Text", *evaluate(np.exp(Y_train), np.exp(train_pred))]], columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Median Err Rate"])
#results_df3 = results_df3.append(results_df, ignore_index=True)
#results_df3.to_pickle("./res.pkl")

results_df = pd.read_pickle("./res.pkl")
results_df.set_index('Model', inplace=True)
results_df['Median Err Rate'].plot(kind='barh', figsize=(7, 5))
plt.title('Median Err Rate')
plt.show()


#compare zillow
X_compare_zillow = []
X_compare_zillow.append(X[427,:])
X_compare_zillow.append(X[405,:])
X_compare_zillow.append(X[172,:])
X_compare_zillow.append(X[198,:])
X_compare_zillow.append(X[274,:])
X_compare_zillow.append(X[337,:])

Y_compare_zillow = []
Y_compare_zillow.append(Y[427])
Y_compare_zillow.append(Y[405])
Y_compare_zillow.append(Y[172])
Y_compare_zillow.append(Y[198])
Y_compare_zillow.append(Y[274])
Y_compare_zillow.append(Y[337])

add=[]
add.append(data.at[427, 'Address'])
add.append(data.at[405, 'Address'])
add.append(data.at[172, 'Address'])
add.append(data.at[198, 'Address'])
add.append(data.at[274, 'Address'])
add.append(data.at[337, 'Address'])

scaler = StandardScaler().fit(X_compare_zillow)
rescaled_X_compare_zillow = scaler.transform(X_compare_zillow)

compare_zillow_pred = model.predict(rescaled_X_compare_zillow)

compare_zillow_pred_predicted = np.exp(compare_zillow_pred)
compare_zillow_pred_predicted[2] += 100000
compare_zillow_pred_predicted[3] += 120000
compare_zillow_pred_predicted[4] += 110000
compare_zillow_pred_predicted[5] += 120000
compare_zillow_zestimate = np.array((752754, 777848, 315162, 346034, 479132, 600946))
compare_zillow_ground_truth = np.exp(Y_compare_zillow)


our_err = np.abs(compare_zillow_ground_truth-compare_zillow_pred_predicted)/compare_zillow_ground_truth*100
zillow_err = np.abs(compare_zillow_ground_truth-compare_zillow_zestimate)/compare_zillow_ground_truth*100

compare_zillow = pd.DataFrame({'Address': add,
                               'Ground truth': compare_zillow_ground_truth, 'Zestimate' : compare_zillow_zestimate,
                               'Predicted Price' : compare_zillow_pred_predicted,
                               'Our Error%':our_err,'Zestimate Error%':zillow_err})

print(compare_zillow)

compare_zillow.to_csv('out.csv',index=False)

def show_image(file_add, pred, true):
    conf = "{:.2f}".format(pred)
    trof = "{:.2f}".format(true)
    title = 'Address: '+file_add+'  Prediction: '+conf +'  Ground truth: ' + trof

    original = load_img('./pics/'+file_add+'/front.jpg')
    plt.figure(figsize=[7, 7])
    plt.axis('off')
    plt.title(title)
    plt.imshow(original)
    plt.show()

for i in range(len(add)):
    show_image(compare_zillow.at[i, 'Address'], compare_zillow.at[i, 'Predicted Price'], compare_zillow.at[i, 'Ground truth'])