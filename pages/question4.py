import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

df_final = pd.read_csv("dataset/cleaned_data.csv")
df_final = df_final[['Unnamed: 0', 'Unnamed: 1','hosp_covid_hospital','rtk-ag','cases_recovered','pcr','Checkins number','cases_new']]
df_final.rename(columns = {'Unnamed: 0': 'date', 'Unnamed: 1': 'state'}, inplace=True)
rslt_df_ph = df_final[df_final['state'] == "Pahang"]
rslt_df_kd = df_final[df_final['state'] == "Kedah"]
rslt_df_jh = df_final[df_final['state'] == "Johor"]
rslt_df_sl = df_final[df_final['state'] == "Selangor"]

def confusion_report(y_test, y_pred):
    # Confusion matrix report

    evaluation_methods = []
    evaluation_scores = []

    confusion_majority=confusion_matrix(y_test, y_pred)

    print('Majority classifier Confusion Matrix\n', confusion_majority)

    print('**********************')
    print('Majority TN= ', confusion_majority[0][0])
    print('Majority FP=', confusion_majority[0][1])
    print('Majority FN= ', confusion_majority[1][0])
    print('Majority TP= ', confusion_majority[1][1])
    print('**********************')

    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    accuracy = accuracy_score(y_test, y_pred)

    evaluation_methods.append('Precision')
    evaluation_methods.append('Recall')
    evaluation_methods.append('F1')
    evaluation_methods.append('Accuracy')
    evaluation_scores.append(precision)
    evaluation_scores.append(recall)
    evaluation_scores.append(f1)
    evaluation_scores.append(accuracy)
    return pd.DataFrame({'Evaluation Method':evaluation_methods, 'Score':evaluation_scores}).set_index('Evaluation Method')

def showMSE(y_test,y_pred):
    from sklearn.metrics import r2_score,median_absolute_error,mean_squared_error,mean_absolute_error
    evaluation_methods = []
    evaluation_scores = []
    
    evaluation_methods.append('Median absolute error')
    evaluation_methods.append('Mean absolute error (MAE)')
    evaluation_methods.append('Mean squared error (MSE)')
    evaluation_methods.append('Root mean square error (RMSE)')
    evaluation_methods.append('R squared (R2)')
    
    evaluation_scores.append(median_absolute_error(y_test, y_pred))
    evaluation_scores.append(mean_absolute_error(y_test, y_pred))
    evaluation_scores.append(mean_squared_error(y_test, y_pred))
    evaluation_scores.append(np.sqrt(mean_squared_error(y_test,y_pred)))
    evaluation_scores.append(r2_score(y_test,y_pred))
    return pd.DataFrame({'Evaluation Method':evaluation_methods, 'Error':evaluation_scores}).set_index('Evaluation Method')

def classify(X,y):
        c1,c2 = st.columns(2)
        c1.markdown("> ## Decision Tree Classifier")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 2)
        clf = DecisionTreeClassifier(criterion="gini", max_depth=4, splitter='random') #pruning the tree by setting the depth
        clf = clf.fit(X_train,y_train)# Train Decision Tree Classifer*
        y_pred = clf.predict(X_test)#Predict the response for test dataset*
        #print(clf)
        df = confusion_report(y_test,y_pred)
        c1.table(df)
        c2.markdown("> ## Gaussian Naive Bayes Classifier")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 2)
        model = GaussianNB()
        model = model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        #print(model)
        c2.table(confusion_report(y_test,y_pred))

def regressor(X,y):
    c1,c2 = st.columns(2)
    c1.markdown("> ## Random Forest Regressor")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 2)
    rfr = RandomForestRegressor()
    rfr.fit(X, y)
    y_pred = rfr.predict(X_test)
    c1.table(showMSE(y_test,y_pred))
    
    c2.markdown("> ## Linear Regressor")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 2)
    model = LinearRegression()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    c2.table(showMSE(y_test,y_pred))
    st.write("> #### Linear Regressor has similar accuracy with Lasso Regressor")
    
def getBinsRange(df):  
        data = df['cases_new'].values
        # First quartile (Q1)
        Q1 = np.percentile(data, 25, interpolation = 'midpoint')
        # Third quartile (Q3)
        Q3 = np.percentile(data, 75, interpolation = 'midpoint')

        return [np.min(data),Q1,Q3,np.inf]


def app():
    st.markdown('> Comparing regression and classification models, what model performs well in predicting the daily cases for Pahang, Kedah, Johor, and Selangor?')
    state_choice = st.selectbox( label = "Choose a State :", options=['Johor','Kedah','Pahang','Selangor','All 4 states'] )
    model_choice = st.selectbox( label = "Choose regressor or classifier :", options=['Regressor','Classifier'] )
    features = {'Features used': ['hosp_covid_hospital','rtk-ag','cases_recovered','pcr','Checkins number','cases_new']}
    
    st.table(pd.DataFrame(features))
    
    if state_choice == "Pahang":
        df = rslt_df_ph
    elif state_choice == "Kedah":
        df = rslt_df_kd
    elif state_choice == "Johor":  
        df = rslt_df_jh
    elif state_choice == "Selangor":
        df = rslt_df_sl
    elif state_choice == "All 4 states":
        df = df_final
        
    print(df)
    
    if model_choice == "Classifier":
        df['cases_new_category'] = (pd.cut(df['cases_new'], bins=getBinsRange(df),labels=['Low', 'Medium', 'High'], include_lowest=True))
        X = df.drop(['cases_new','date','state','cases_new_category'], axis=1)
        y = df.cases_new_category 
        classify(X,y)
        df.drop('cases_new_category',axis=1,inplace=True)
    else:
        X = df.drop(['cases_new','date','state'], axis=1)  #predict newcases
        y = df['cases_new']
        regressor(X,y)
        
    st.markdown('Among the approaches to find the best regression model, we have tried fitting into decision tree regressor, linear regression, lasso regression, and random forest regression. As a result, the decision tree regressor is not suitable as its Mean Squared Error is the highest among the models. __Linear regression__ is having a similar accuracy with Lasso regression. Lastly, __Random forest regression__ works the best and has the highest accuracy to predict the Covid-19 daily new cases. Evaluation metrics used for regression have Median absolute error, Mean absolute error (MAE), Mean squared error (MSE), Root mean square error (RMSE) and R squared (R2). For the classification model, we have trained the data using decision tree classifier, logistic regression classifier, K-Nearest Neighbour (KNN) and Gaussian Naive Bayes model. In predicting the test data, __Gaussian Naive Bayes__ have the best accuracy, followed by __Decision Tree classifier__, Logistic regression classifier and KNN. Evaluation metrics used in classification models are precision and recall score, F1 score and Accuracy (number of correct predictions/ total number of predictions made). By using __Johor and All 4 states datasets__, the model is predicting the higher accuracy of daily new cases than the other datasets (Kedah, Pahang, Selangor dataset). ')

