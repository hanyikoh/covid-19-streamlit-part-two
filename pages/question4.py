import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
# Import train_test_split functionn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, median_absolute_error, mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

r_naught_df = pd.read_csv("dataset/r-naught-value - All.csv")
malaysia_case_df = pd.read_csv('dataset/cases_malaysia.csv')
malaysia_death_df = pd.read_csv('dataset/deaths_malaysia.csv')

r_naught_df = r_naught_df._convert(numeric=True)
r_naught_df = r_naught_df.replace(np.nan, 0)
r_naught_df.set_index('date', inplace=True)
X = r_naught_df.drop(['Malaysia'], axis=1)  # predict newcases
y = r_naught_df['Malaysia']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 2)

def model_evaluation(model, X, y):
    kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
    results = model_selection.cross_val_score(
        model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
    # print("MAE: %.3f (%.3f)" % (results.mean(), results.std()))
    mae = results.mean()
    results = model_selection.cross_val_score(
        model, X, y, cv=kfold, scoring='neg_mean_squared_error')
    # print("MSE: %.3f (%.3f)" % (results.mean(), results.std()))
    mse = results.mean()
    results = model_selection.cross_val_score(
        model, X, y, cv=kfold, scoring='r2')
    # print("R^2: %.3f (%.3f)" % (results.mean(), results.std()))
    r2 = results.mean()
    return mae, mse, r2

def create_LSTM_model(X_train):
  regressor = Sequential()
  regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
  regressor.add(Dropout(rate = 0.2))
  regressor.add(LSTM(units = 150, return_sequences = False, input_shape = (X_train.shape[1], 1)))
  regressor.add(Dropout(rate = 0.2))
  regressor.add(Dense(1))

  return regressor

def Run_LSTM(training_set, df, is_cases):
  sc = MinMaxScaler(feature_range = (0, 1))
  #fit: get min/max of train data
  training_set_scaled = sc.fit_transform(training_set)
  # training_set_scaled = training_set

  ## 30 timesteps and 1 output
  X_train = []
  y_train = []
  for i in range(30, len(training_set_scaled)):
      X_train.append(training_set_scaled[i-30: i, 0])
      y_train.append(training_set_scaled[i, 0])

  X_train, y_train = np.array(X_train), np.array(y_train)

  X_train = np.reshape(X_train, newshape = (X_train.shape[0], X_train.shape[1], 1))
  regressor = create_LSTM_model(X_train)
  regressor.compile(loss='mean_squared_error', optimizer='adam')
  regressor.fit(X_train, y_train, epochs=100, batch_size=32, verbose=2)
  
  dataset_test = df.iloc[-30:, :]
  dataset_train = df.iloc[:-30, :]
  if is_cases:
    dataset_total = pd.concat((dataset_train['cases_new'], dataset_test['cases_new']),axis = 0)
  else:
    dataset_total = pd.concat((dataset_train['deaths_new'], dataset_test['deaths_new']),axis = 0)
  inputs = dataset_total[len(dataset_total) - len(dataset_test) - 30:].values
  inputs = inputs.reshape(-1, 1)
  inputs = inputs.astype(float)
  inputs = sc.transform(inputs)
  # your codes
  ## 60 timesteps and 1 output
  X_test = []
  y_test = []
  for i in range(30, len(inputs)):
      X_test.append(inputs[i-30: i, 0])
      y_test.append(inputs[i, 0])

  X_test, y_test = np.array(X_test), np.array(y_test)
  X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
  y_test = np.reshape(y_test, (y_test.shape[0], 1))
  predict_value = regressor.predict(X_test)
  #inverse the scaled value
  # your codes
  predicted_values = sc.inverse_transform(predict_value)
  real_values= sc.inverse_transform(y_test)
  return predicted_values,real_values

def app():
    st.markdown('>  Predict the R naught index of Malaysia and states')
    mae_list = []
    mse_list = []
    r2_list = []

    model = LinearRegression()
    mae, mse, r2 = model_evaluation(model, X, y)
    mae_list.append(mae)
    mse_list.append(mse)
    r2_list.append(r2)

    model = DecisionTreeRegressor(max_depth=2)
    mae, mse, r2 = model_evaluation(model, X, y)
    mae_list.append(mae)
    mse_list.append(mse)
    r2_list.append(r2)    
    
    model = Lasso(alpha=1.0)
    mae, mse, r2 = model_evaluation(model, X, y)
    mae_list.append(mae)
    mse_list.append(mse)
    r2_list.append(r2)

    model = SVR()
    mae, mse, r2 = model_evaluation(model, X, y)
    mae_list.append(mae)
    mse_list.append(mse)
    r2_list.append(r2)

    model = RandomForestRegressor(max_depth=2)
    mae, mse, r2 = model_evaluation(model, X, y)
    mae_list.append(mae)
    mse_list.append(mse)
    r2_list.append(r2)
    model_list = ['Linear Regression','Decision Tree Regression', 'Lasso Regression', 'SVR Regression', 'Random Forest Regression']
    d = {'Regression Algorithm':model_list,'MAE': mae_list, 'MSE': mse_list, 'R2':r2_list}
    st.table(pd.DataFrame(data=d))

    ##visualize the prediction and real price
    training_set = malaysia_case_df.iloc[:-30, 1:2].values
    predicted_values, real_values = Run_LSTM(training_set,malaysia_case_df, True)
    plt.plot(real_values, color = 'red', label = 'Real New Case Value')
    plt.plot(predicted_values, color = 'blue', label = 'Predicted New Case Value')

    plt.title('Malaysia New Case Prediction')
    plt.xlabel('Time')
    plt.ylabel('New Case Number')
    plt.legend()
    plt.show()
    st.pyplot()

    training_set = malaysia_death_df.iloc[:-30, 1:2].values
    predicted_values, real_values = Run_LSTM(training_set,malaysia_death_df, False)
    plt.plot(real_values, color = 'red', label = 'Real New Deaths Value')
    plt.plot(predicted_values, color = 'blue', label = 'Predicted New Deaths Value')
    # plt.ylim(np.amin(np.concatenate([predicted_values, real_values])),np.amax(np.concatenate([predicted_values, real_values])))

    plt.title('Malaysia New Deaths Prediction')
    plt.xlabel('Time')
    plt.ylabel('New Deaths Number')
    plt.legend()
    plt.show()
    st.pyplot()