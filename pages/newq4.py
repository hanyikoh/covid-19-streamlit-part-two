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

r_naught_df = pd.read_csv("dataset/r-naught-value - All.csv")
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