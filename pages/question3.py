from re import U
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
import plotly.graph_objects as go
import plotly.offline as py
import plotly.express as px
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
# from tqdm import tqdm_notebook, tqdm
#tqdm.pandas(tqdm_notebook)
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.metrics import r2_score,median_absolute_error,mean_squared_error,mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix  
from sklearn.metrics import roc_curve, auc      
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

vax_df = pd.read_csv('dataset/aefi.csv')
r_naught_dir = 'dataset/r-naught-value - All.csv'

start_date = vax_df["date"] >= "2021-04-01"
vax_df = vax_df.loc[start_date]
vax_df.drop(['date'],axis=1,inplace=True)
col = ['daily_total',	'daily_serious_npra',	'daily_nonserious',	'daily_nonserious_npra', 'daily_nonserious_mysj_dose1',	'daily_nonserious_mysj_dose2']
vax_df.drop(col,axis=1,inplace=True)

def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))

def getBinsRange(df):  
        data = df['Malaysia'].values
        # First quartile (Q1)
        Q1 = np.percentile(data, 25, interpolation = 'midpoint')
        # Second quartile (Q2)
        Q2 = np.percentile(data, 50, interpolation = 'midpoint')
        # Third quartile (Q3)
        Q3 = np.percentile(data, 75, interpolation = 'midpoint')

        return [np.min(data),(Q1 + Q2)/2, (Q2 + Q3)/2,np.inf]


def app():
    st.markdown("> Classification")
    
    selected_metrics = st.selectbox(label = "Choose a Classifier :", options=['Classify Vaccine Type', 'Classify R-naught Level'] )
    
    if selected_metrics == "Classify Vaccine Type":
        st.markdown("### Classify Vaccine Type based on Vaccine Side-Effects")
        st.markdown("#### Feature Selection - Recursive Feature Elimination (RFE) ")
        st.markdown("After using different methods, we decide to use Recursive Feature Elimination (RFE) to select the most useful feature. Feature selection can reduce overfitting, increase the model's accuracy and reduce training time. RFE known as wrapper feature selection method, which is easy to implement and good for classification accuracy.")
        st.markdown('Recursive Feature Elimination (RFE) works by searching for a subset of features in the training dataset, starting with all of them and successfully removing them until the desired number remains. This is accomplished by re-fitting the model using the given machine learning algorithm, ranking features by importance, discarding the least important features, and fitting the model again. This procedure is repeated until only a certain number of features are left.')
        X = vax_df.drop(["vaxtype"], 1)
        colnames = X.columns

        # define dataset
        X, y = make_classification(n_samples=1000, n_features=24, n_informative=5, n_redundant=5, random_state=1)
        # define RFE
        rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=15)
        # fit RFE
        rfe.fit(X, y)
        rfe_score = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)
        rfe_score = pd.DataFrame(list(rfe_score.items()), columns=['Features', 'Score'])
        rfe_score = rfe_score.sort_values("Score", ascending = False)
        #fig, ax = plt.subplots(figsize=(5,5)) 
        
        sns.set(style='whitegrid')
        sns_rfe_plot = sns.catplot(x="Score", y="Features", data = rfe_score[:], kind = "bar",  orientation='horizontal',
                    height=10, aspect=.7, palette='Purples_r')
        plt.title("RFE Features Ranking")
        plt.xlabel('FEATURE IMPORTANCE')
        plt.ylabel('FEATURE NAMES')
        st.pyplot()
        
        
        selected_col = ['d1_site_pain', 'd1_site_swelling','d1_joint_pain','d1_weakness', 'd1_fever','d1_vomiting','d1_rash', 'd2_site_pain', 'd2_site_swelling', 'd2_site_redness', 'd2_tiredness', 'd2_headache', 'd2_weakness','d2_fever', 'd2_chills']
        st.markdown("#### Class Balancing - SMOTE")
        
        df = pd.DataFrame(vax_df['vaxtype'].value_counts())
        df = df.reset_index()

        col1,col2 = st.columns(2)
        with col1:
            fig2 = px.bar(df, x='index', y='vaxtype')
            fig2.update_layout(title='Vaccine Types Histogram', xaxis_title='Vaccine Type', yaxis_title='Frequency')
            fig2.show()
            st.plotly_chart(fig2, use_container_width=True)
        X = np.array(vax_df[selected_col])
        y = np.array(vax_df.loc[:, vax_df.columns == 'vaxtype'])
        sm = SMOTE(random_state=2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())
        df2 = pd.DataFrame(pd.value_counts(y_train_res))
        df2 = df2.reset_index()
        with col2:
            fig3 = px.bar(df2, x='index', y=0)
            fig3.update_layout(title='Vaccine Types Histogram', xaxis_title='Vaccine Type', yaxis_title='Frequency')
            fig3.show()
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("#### Clssification Model - Logistic Regression")
        logreg = LogisticRegression() #solver best 'S' curve
        logreg.fit(X_train_res, y_train_res)
        y_pred = logreg.predict(X_test)
        result = pd.DataFrame()
        result = pd.DataFrame(X_test,columns = selected_col)
        result['y_test'] = y_test.ravel()
        result['y_pred'] = y_pred

        table = result.head()
        st.table(table)
        st.markdown('### Evaluation Metrics for Classifier\n')
        cm = confusion_matrix(y_test, y_pred)
        cm = pd.DataFrame(cm,
                     index = ['Pfizer','Sinovac','Astrazeneca'], 
                     columns =  ['Pfizer','Sinovac','Astrazeneca'])
        plt.figure(figsize=(8, 6))
        plt.title('Confusion Matrix (with SMOTE)', size=16)
        sns.heatmap(cm, annot=True, cmap='Blues')

        plt.ylabel('Actual Values')
        plt.xlabel('Predicted Values')
        st.pyplot()
        #fig1 = px.bar(rfe_score[:], x="Score", y="Features", orientation='h', color_continuous_scale='Rainbow')
        #fig1.update_layout(title='RFE Features Ranking')
        #st.plotly_chart(fig1)
        result = pd.DataFrame({'Evaluation Metrics':['Precision','Recall','F1 Score', 'Accuracy'],
                   'Score':[precision_score(y_test, y_pred, average="weighted"),recall_score(y_test, y_pred, average="weighted"),f1_score(y_test, y_pred, average="weighted"),accuracy_score(y_test, y_pred)]})
        result = result.set_index('Evaluation Metrics')
        st.table(result)
        # st.markdown('##### Precision= {:.2f}'.format(precision_score(y_test, y_pred, average="weighted")))
        # st.markdown('##### Recall= {:.2f}'. format(recall_score(y_test, y_pred, average="weighted")))
        # st.markdown('##### F1= {:.2f}'. format(f1_score(y_test, y_pred, average="weighted")))
        # st.markdown('##### Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))

    elif selected_metrics == "Classify R-naught Level":
        start_date = "2021-07-01"
        end_date = "2021-09-30"
        labels = ['Low','Medium','High']

        r_naught_df = pd.read_csv(r_naught_dir)
        r_naught_df['date'] = pd.to_datetime(r_naught_df['date'], format='%d/%m/%Y')
        # r_naught_df.drop(columns=['Date'], inplace=True)
        after_start_date = r_naught_df["date"] >= start_date
        before_end_date = r_naught_df["date"] <= end_date
        between_two_dates = after_start_date & before_end_date
        r_naught_df = r_naught_df.loc[between_two_dates]
        r_naught_df.head()
        
        r_naught_df_copy = r_naught_df.copy()

        r_naught_df_copy['Malaysia_Category'] = (
            pd.cut(
                r_naught_df_copy['Malaysia'].values, bins=getBinsRange(r_naught_df_copy),labels=labels, include_lowest=True)
        )
        r_naught_df_copy = r_naught_df_copy._convert(numeric=True)
        r_naught_df_copy = r_naught_df_copy.replace(np.nan,0)
        r_naught_df_copy.set_index('date', inplace=True)
        X = r_naught_df_copy.drop(['Malaysia','Malaysia_Category'], axis=1)  #predict newcases
        y = r_naught_df_copy['Malaysia_Category']
        
        sm = SMOTE(random_state=42)

        X_sm, y_sm = sm.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.3, random_state=1)
        clf = DecisionTreeClassifier(criterion="entropy", max_depth=5, splitter='best') #pruning the tree by setting the depth
        clf = clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        result = pd.DataFrame()
        result = pd.DataFrame(X_test)
        result['y_test'] = y_test.ravel()
        result['y_pred'] = y_pred
        table = result.head()
        
        st.markdown('### Classification Model - Decision Tree Classifier')
        st.table(table)
        st.markdown('### Evaluation Metrics for Classifier')
        cm = confusion_matrix(y_test, y_pred)
        cm = pd.DataFrame(cm,
                index = ['Low','Medium','High'], 
                columns =  ['Low','Medium','High'])
        plt.figure(figsize=(8, 6))
        plt.title('Confusion Matrix (with SMOTE)', size=16)
        
        sns.heatmap(cm, annot=True, cmap='Purples')
        plt.ylabel('Actual Values')
        plt.xlabel('Predicted Values')
        st.pyplot()
        #fig4 = px.imshow(cm)
        #fig4.update_layout(title = 'Confusion Matrix (with SMOTE)')
        #fig4.show()
        #st.plotly_chart(fig4, use_container_width=True)
        result = pd.DataFrame({'Evaluation Metrics':['Precision','Recall','F1 Score', 'Accuracy'],
            'Score':[precision_score(y_test, y_pred, average="weighted"),recall_score(y_test, y_pred, average="weighted"),f1_score(y_test, y_pred, average="weighted"),accuracy_score(y_test, y_pred)]})
        result = result.set_index('Evaluation Metrics')
        st.table(result)
        # st.markdown('##### Precision= {:.2f}'.format(precision_score(y_test, y_pred, average="weighted")))
        # st.markdown('##### Recall= {:.2f}'. format(recall_score(y_test, y_pred, average="weighted")))
        # st.markdown('##### F1= {:.2f}'. format(f1_score(y_test, y_pred, average="weighted")))
        # st.markdown('##### Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))