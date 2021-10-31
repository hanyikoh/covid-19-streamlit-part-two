import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

malaysia_case_dir = "dataset/cases_malaysia.csv"
state_case_dir = "dataset/cases_state.csv"
clusters_dir = "dataset/clusters.csv"
malaysia_tests_dir = "dataset/tests_malaysia.csv"
states_tests_dir = "dataset/tests_state.csv"
pkrc_dir = "dataset/pkrc.csv"
checkIn_dir = "dataset/checkin_state.csv"
hospital_dir = "dataset/hospital.csv"

def app():
    st.markdown('> Discuss the exploratory data analysis steps you have conducted including detection of outliers and missing values?')
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    chosen = st.radio(
    'Choose a Dataset',
    ["Malaysia Cases", "State Cases", "Clusters", "Malaysia Tests","State Tests","PKRC","Hospital","State CheckIn"])
    st.markdown(f"__{chosen} Dataset:__")
    start_date = "2021-07-01"
    end_date = "2021-08-31"

    if chosen == "State Cases":
        state_case_df = pd.read_csv(state_case_dir)
        after_start_date = state_case_df["date"] >= start_date
        before_end_date = state_case_df["date"] <= end_date
        between_two_dates = after_start_date & before_end_date
        state_case_df = state_case_df.loc[between_two_dates]

        st.text('Daily recorded COVID-19 cases at state level.')
        st.write('First 5 rows of the dataset')
        st.table(state_case_df.head().reset_index(drop=True))
        
        st.write('Statistical Overview')
        st.table(state_case_df.describe())
        
        st.write("Missing Values Detection")
        col1, col2 = st.columns(2)
        null_df=pd.DataFrame({'Column':state_case_df.isna().sum().index, 'Count of Null Values':state_case_df.isna().sum().values})  
        col1.table(null_df)
        
        missing_values = state_case_df.isnull().sum() / len(state_case_df)
        missing_values = missing_values[missing_values > 0]
        missing_values.sort_values(inplace=True)
        missing_values = missing_values.to_frame()
        missing_values.columns = ['Count of Missing Values']
        missing_values.index.names = ['Name']
        missing_values['Column Name'] = state_case_df.columns

        sns.set(style="whitegrid", color_codes=True)
        sns.barplot(x = 'Column Name', y = 'Count of Missing Values', data=missing_values)
        plt.xticks(rotation = 90)
        plt.show()
        col2.pyplot()

        st.write('Outliers detection with Boxplot')
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        # fig.suptitle('Outliers Visualization')
        plt.subplots_adjust(left=None, bottom= 0.1, right=None, top=0.5, wspace=0.2, hspace=0.6)

        sns.boxplot(data=state_case_df,x=state_case_df["cases_import"],ax=axes[0])
        axes[0].set_title('Import Case')
        sns.boxplot(data=state_case_df,x=state_case_df["cases_new"],ax=axes[1])
        axes[1].set_title('New Case')
        sns.boxplot(data=state_case_df,x=state_case_df["cases_recovered"],ax=axes[2])
        axes[2].set_title('Recovered Case')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    elif chosen == "Clusters":
        clusters_df = pd.read_csv(clusters_dir)
        after_start_date = clusters_df["date_announced"] >= start_date
        before_end_date = clusters_df["date_announced"] <= end_date
        between_two_dates = after_start_date & before_end_date
        clusters_df = clusters_df.loc[between_two_dates]
        clusters_df['date'] = clusters_df.date_announced

        st.text('Exhaustive list of announced clusters with relevant epidemiological datapoint.')
        st.write('First 5 rows of the dataset')
        st.table(clusters_df.head().reset_index(drop=True))

        st.write('Statistical Overview')
        st.table(clusters_df.describe())

        st.write("Missing Values Detection")
        col1, col2 = st.columns(2)
        null_df=pd.DataFrame({'Column':clusters_df.isna().sum().index, 'Count of Null Values':clusters_df.isna().sum().values})  
        col1.table(null_df)
        
        missing_values = clusters_df.isnull().sum() / len(clusters_df)
        missing_values = missing_values[missing_values > 0]
        missing_values.sort_values(inplace=True)
        missing_values = missing_values.to_frame()
        missing_values.columns = ['Count of Missing Values']
        missing_values.index.names = ['Name']
        missing_values['Column Name'] = clusters_df.columns

        sns.set(style="whitegrid", color_codes=True)
        sns.barplot(x = 'Column Name', y = 'Count of Missing Values', data=missing_values)
        plt.xticks(rotation = 90)
        plt.show()
        col2.pyplot()

        st.write('Outliers detection with Boxplot')
        fig, axes = plt.subplots(3, 3, figsize=(15, 5), sharey=True)
        plt.subplots_adjust(left=None, bottom= 0.1, right=None, top=2, wspace=0.2, hspace=0.6)
        sns.boxplot(data=clusters_df,x=clusters_df["cases_new"],ax=axes[0][0])
        axes[0][0].set_title('cases_new')
        sns.boxplot(data=clusters_df,x=clusters_df["cases_total"],ax=axes[0][1])
        axes[0][1].set_title('cases_total')
        sns.boxplot(data=clusters_df,x=clusters_df["cases_active"],ax=axes[0][2])
        axes[0][2].set_title('cases_active')
        sns.boxplot(data=clusters_df,x=clusters_df["tests"],ax=axes[1][0])
        axes[1][0].set_title('tests')
        sns.boxplot(data=clusters_df,x=clusters_df["icu"],ax=axes[1][1])
        axes[1][1].set_title('icu')
        sns.boxplot(data=clusters_df,x=clusters_df["deaths"],ax=axes[1][2])
        axes[1][2].set_title('deaths')
        sns.boxplot(data=clusters_df,x=clusters_df["recovered"],ax=axes[2][0])
        axes[2][0].set_title('recovered')
        fig.delaxes(axes[2][1])
        fig.delaxes(axes[2][2])

        sns.boxplot(data=clusters_df,x=clusters_df["recovered"])
        st.pyplot()

    elif chosen == "State Tests":
        states_tests_df = pd.read_csv(states_tests_dir)
        after_start_date = states_tests_df["date"] >= start_date
        before_end_date = states_tests_df["date"] <= end_date
        between_two_dates = after_start_date & before_end_date
        states_tests_df = states_tests_df.loc[between_two_dates]
        
        st.text('Daily tests (note: not necessarily unique individuals) by type at state level.')
        st.write('First 5 rows of the dataset')
        st.table(states_tests_df.head().reset_index(drop=True))

        st.write('Statistical Overview')
        st.table(states_tests_df.describe())

        st.write("Missing Values Detection")
        col1, col2 = st.columns(2)
        null_df=pd.DataFrame({'Column':states_tests_df.isna().sum().index, 'Count of Null Values':states_tests_df.isna().sum().values})  
        col1.table(null_df)
        
        missing_values = states_tests_df.isnull().sum() / len(states_tests_df)
        missing_values = missing_values[missing_values > 0]
        missing_values.sort_values(inplace=True)
        missing_values = missing_values.to_frame()
        missing_values.columns = ['Count of Missing Values']
        missing_values.index.names = ['Name']
        missing_values['Column Name'] = states_tests_df.columns

        sns.set(style="whitegrid", color_codes=True)
        sns.barplot(x = 'Column Name', y = 'Count of Missing Values', data=missing_values)
        plt.xticks(rotation = 90)
        plt.show()
        col2.pyplot()

        st.write('Outliers detection with Boxplot')
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
        # fig.suptitle('Outliers Visualization')
        plt.subplots_adjust(left=None, bottom= 0.1, right=None, top=0.5, wspace=0.2, hspace=0.6)

        sns.boxplot(data=states_tests_df, x = states_tests_df["rtk-ag"],ax=axes[0])
        axes[0].set_title('rtk-ag')
        sns.boxplot(data=states_tests_df,x = states_tests_df["pcr"],ax=axes[1])
        axes[1].set_title('pcr')

        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    elif chosen == "Malaysia Cases":
        malaysia_case_df = pd.read_csv(malaysia_case_dir)
        after_start_date = malaysia_case_df["date"] >= start_date
        before_end_date = malaysia_case_df["date"] <= end_date
        between_two_dates = after_start_date & before_end_date
        malaysia_case_df = malaysia_case_df.loc[between_two_dates]

        st.text('Daily recorded COVID-19 cases at country level.')
        st.write('First 5 rows of the dataset')
        st.table(malaysia_case_df.head().reset_index(drop=True))

        st.write('Statistical Overview')
        st.table(malaysia_case_df.describe())

        st.write("Missing Values Detection")
        col1, col2 = st.columns(2)
        null_df=pd.DataFrame({'Column':malaysia_case_df.isna().sum().index, 'Count of Null Values':malaysia_case_df.isna().sum().values})  
        col1.table(null_df)
        
        missing_values = malaysia_case_df.isnull().sum() / len(malaysia_case_df)
        missing_values = missing_values[missing_values > 0]
        missing_values.sort_values(inplace=True)
        missing_values = missing_values.to_frame()
        missing_values.columns = ['Count of Missing Values']
        missing_values.index.names = ['Name']
        missing_values['Column Name'] = malaysia_case_df.columns

        sns.set(style="whitegrid", color_codes=True)
        sns.barplot(x = 'Column Name', y = 'Count of Missing Values', data=missing_values)
        plt.xticks(rotation = 90)
        plt.show()
        col2.pyplot()

        st.write('Outliers detection with Boxplot')
        fig, axes = plt.subplots(4, 3, figsize=(15, 5), sharey=True)
        # fig.suptitle('Outliers Visualization')
        plt.subplots_adjust(left=None, bottom= 0.1, right=None, top=2, wspace=0.2, hspace=0.6)

        sns.boxplot(data=malaysia_case_df,x=malaysia_case_df["cases_new"],ax=axes[0][0])
        axes[0][0].set_title('New Case')
        sns.boxplot(data=malaysia_case_df,x=malaysia_case_df["cases_import"],ax=axes[0][1])
        axes[0][1].set_title('Case Imprt')
        sns.boxplot(data=malaysia_case_df,x=malaysia_case_df["cases_recovered"],ax=axes[0][2])
        axes[0][2].set_title('Case Recovered')
        sns.boxplot(data=malaysia_case_df,x=malaysia_case_df["cluster_import"],ax=axes[1][0])
        axes[1][0].set_title('cluster_workplace')
        sns.boxplot(data=malaysia_case_df,x=malaysia_case_df["cluster_religious"],ax=axes[1][1])
        axes[1][1].set_title('cluster_religious')
        sns.boxplot(data=malaysia_case_df,x=malaysia_case_df["cluster_community"],ax=axes[1][2])
        axes[1][2].set_title('cluster_community')
        sns.boxplot(data=malaysia_case_df,x=malaysia_case_df["cluster_highRisk"],ax=axes[2][0])
        axes[2][0].set_title('cluster_highRisk')
        sns.boxplot(data=malaysia_case_df,x=malaysia_case_df["cluster_education"],ax=axes[2][1])
        axes[2][1].set_title('cluster_education')
        sns.boxplot(data=malaysia_case_df,x=malaysia_case_df["cluster_detentionCentre"],ax=axes[2][2])
        axes[2][2].set_title('cluster_detentionCentre')
        sns.boxplot(data=malaysia_case_df,x=malaysia_case_df["cluster_workplace"],ax=axes[3][0])
        axes[3][0].set_title('cluster_workplace')
        fig.delaxes(axes[3][1])
        fig.delaxes(axes[3][2])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    elif chosen == "Malaysia Tests":
        malaysia_tests_df = pd.read_csv(malaysia_tests_dir)
        after_start_date = malaysia_tests_df["date"] >= start_date
        before_end_date = malaysia_tests_df["date"] <= end_date
        between_two_dates = after_start_date & before_end_date
        malaysia_tests_df = malaysia_tests_df.loc[between_two_dates]

        st.text('Daily tests (note: not necessarily unique individuals) by type at country level.')
        st.write('First 5 rows of the dataset')
        st.table(malaysia_tests_df.head().reset_index(drop=True))

        st.write('Statistical Overview')
        st.table(malaysia_tests_df.describe())

        st.write("Missing Values Detection")
        col1, col2 = st.columns(2)
        null_df=pd.DataFrame({'Column':malaysia_tests_df.isna().sum().index, 'Count of Null Values':malaysia_tests_df.isna().sum().values})  
        col1.table(null_df)
        
        missing_values = malaysia_tests_df.isnull().sum() / len(malaysia_tests_df)
        missing_values = missing_values[missing_values > 0]
        missing_values.sort_values(inplace=True)
        missing_values = missing_values.to_frame()
        missing_values.columns = ['Count of Missing Values']
        missing_values.index.names = ['Name']
        missing_values['Column Name'] = malaysia_tests_df.columns

        sns.set(style="whitegrid", color_codes=True)
        sns.barplot(x = 'Column Name', y = 'Count of Missing Values', data=missing_values)
        plt.xticks(rotation = 90)
        plt.show()
        col2.pyplot()

        st.write('Outliers detection with Boxplot')
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
        # fig.suptitle('Outliers Visualization')
        plt.subplots_adjust(left=None, bottom= 0.1, right=None, top=0.5, wspace=0.2, hspace=0.6)

        sns.boxplot(data=malaysia_tests_df, x = malaysia_tests_df["rtk-ag"],ax=axes[0])
        axes[0].set_title('rtk-ag')

        sns.boxplot(data=malaysia_tests_df,x = malaysia_tests_df["pcr"],ax=axes[1])
        axes[1].set_title('pcr')
    
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    elif chosen == "PKRC":
        pkrc_df = pd.read_csv(pkrc_dir)
        after_start_date = pkrc_df["date"] >= start_date
        before_end_date = pkrc_df["date"] <= end_date
        between_two_dates = after_start_date & before_end_date
        pkrc_df = pkrc_df.loc[between_two_dates]

        st.text('Flow of patients to/out of Covid-19 Quarantine and Treatment Centres (PKRC), with capacity and utilisation.')
        st.write('First 5 rows of the dataset')
        st.table(pkrc_df.head().reset_index(drop=True))

        st.write('Statistical Overview')
        st.table(pkrc_df.describe())

        st.write("Missing Values Detection")
        col1, col2 = st.columns(2)
        null_df=pd.DataFrame({'Column':pkrc_df.isna().sum().index, 'Count of Null Values':pkrc_df.isna().sum().values})  
        col1.table(null_df)
        
        missing_values = pkrc_df.isnull().sum() / len(pkrc_df)
        missing_values = missing_values[missing_values > 0]
        missing_values.sort_values(inplace=True)
        missing_values = missing_values.to_frame()
        missing_values.columns = ['Count of Missing Values']
        missing_values.index.names = ['Name']
        missing_values['Column Name'] = pkrc_df.columns

        sns.set(style="whitegrid", color_codes=True)
        sns.barplot(x = 'Column Name', y = 'Count of Missing Values', data=missing_values)
        plt.xticks(rotation = 90)
        plt.show()
        col2.pyplot()

        st.write('Outliers detection with Boxplot')
        fig, axes = plt.subplots(4, 3, figsize=(15, 5), sharey=True)
        # fig.suptitle('Outliers Visualization')
        plt.subplots_adjust(left=None, bottom= 0.1, right=None, top=2, wspace=0.2, hspace=0.6)

        sns.boxplot(data=pkrc_df, x = pkrc_df["beds"],ax=axes[0][0])
        axes[0][0].set_title('beds')
        sns.boxplot(data=pkrc_df,x = pkrc_df["admitted_pui"],ax=axes[0][1])
        axes[0][1].set_title('admitted_pui')
        sns.boxplot(data=pkrc_df, x = pkrc_df["admitted_covid"],ax=axes[0][2])
        axes[0][2].set_title("admitted_covid")
        sns.boxplot(data=pkrc_df,x = pkrc_df["admitted_total"],ax=axes[1][0])
        axes[1][0].set_title('admitted_total')
        sns.boxplot(data=pkrc_df, x = pkrc_df["discharge_pui"],ax=axes[1][1])
        axes[1][1].set_title('discharge_pui')
        sns.boxplot(data=pkrc_df,x = pkrc_df["discharge_covid"],ax=axes[1][2])
        axes[1][2].set_title('discharge_covid')
        sns.boxplot(data=pkrc_df, x = pkrc_df["discharge_total"],ax=axes[2][0])
        axes[2][0].set_title('discharge_total')
        sns.boxplot(data=pkrc_df,x = pkrc_df["pkrc_covid"],ax=axes[2][1])
        axes[2][1].set_title('pkrc_covid')
        sns.boxplot(data=pkrc_df, x = pkrc_df["pkrc_pui"],ax=axes[2][2])
        axes[2][2].set_title('pkrc_pui')
        sns.boxplot(data=pkrc_df,x = pkrc_df["pkrc_noncovid"],ax=axes[3][0])
        axes[3][0].set_title('pkrc_noncovid')
        fig.delaxes(axes[3][1])
        fig.delaxes(axes[3][2])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    elif chosen == "State CheckIn":
        checkIn_df = pd.read_csv(checkIn_dir)
        after_start_date = checkIn_df["date"] >= start_date
        before_end_date = checkIn_df["date"] <= end_date
        between_two_dates = after_start_date & before_end_date
        checkIn_df = checkIn_df.loc[between_two_dates]

        st.text('Daily checkins on MySejahtera at state level.')
        st.write('First 5 rows of the dataset')
        st.table(checkIn_df.head().reset_index(drop=True))

        st.write('Statistical Overview')
        st.table(checkIn_df.describe())

        st.write("Missing Values Detection")
        col1, col2 = st.columns(2)
        null_df=pd.DataFrame({'Column':checkIn_df.isna().sum().index, 'Count of Null Values':checkIn_df.isna().sum().values})  
        col1.table(null_df)
        
        missing_values = checkIn_df.isnull().sum() / len(checkIn_df)
        missing_values = missing_values[missing_values > 0]
        missing_values.sort_values(inplace=True)
        missing_values = missing_values.to_frame()
        missing_values.columns = ['Count of Missing Values']
        missing_values.index.names = ['Name']
        missing_values['Column Name'] = checkIn_df.columns

        sns.set(style="whitegrid", color_codes=True)
        sns.barplot(x = 'Column Name', y = 'Count of Missing Values', data=missing_values)
        plt.xticks(rotation = 90)
        plt.show()
        col2.pyplot()

        st.write('Outliers detection with Boxplot')
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        # fig.suptitle('Outliers Visualization')
        plt.subplots_adjust(left=None, bottom= 0.1, right=None, top=0.5, wspace=0.2, hspace=0.6)

        sns.boxplot(data=checkIn_df, x = checkIn_df["checkins"],ax=axes[0])
        axes[0].set_title('checkins')
        sns.boxplot(data=checkIn_df,x = checkIn_df["unique_ind"],ax=axes[1])
        axes[1].set_title('unique_ind')
        sns.boxplot(data=checkIn_df, x = checkIn_df["unique_loc"],ax=axes[2])
        axes[2].set_title('unique_loc')

        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    elif chosen == "Hospital":
        hospital_df = pd.read_csv(hospital_dir)
        after_start_date = hospital_df["date"] >= start_date
        before_end_date = hospital_df["date"] <= end_date
        between_two_dates = after_start_date & before_end_date
        hospital_df = hospital_df.loc[between_two_dates]
        hospital_df.head()
        st.text('Flow of patients to/out of hospitals, with capacity and utilisation.')
        st.write('First 5 rows of the dataset')
        st.table(hospital_df.head().reset_index(drop=True))

        st.write('Statistical Overview')
        st.table(hospital_df.describe())

        st.write("Missing Values Detection")
        col1, col2 = st.columns(2)
        null_df=pd.DataFrame({'Column':hospital_df.isna().sum().index, 'Count of Null Values':hospital_df.isna().sum().values})  
        col1.table(null_df)
        
        missing_values = hospital_df.isnull().sum() / len(hospital_df)
        missing_values = missing_values[missing_values > 0]
        missing_values.sort_values(inplace=True)
        missing_values = missing_values.to_frame()
        missing_values.columns = ['Count of Missing Values']
        missing_values.index.names = ['Name']
        missing_values['Column Name'] = hospital_df.columns

        sns.set(style="whitegrid", color_codes=True)
        sns.barplot(x = 'Column Name', y = 'Count of Missing Values', data=missing_values)
        plt.xticks(rotation = 90)
        plt.show()
        col2.pyplot()

        st.write('Outliers detection with Boxplot')
        fig, axes = plt.subplots(4, 3, figsize=(15, 5), sharey=True)
        plt.subplots_adjust(left=None, bottom= 0.1, right=None, top=2, wspace=0.2, hspace=0.6)
        sns.boxplot(data=hospital_df, x = hospital_df["beds"],ax=axes[0][0])
        axes[0][0].set_title('beds')
        sns.boxplot(data=hospital_df,x = hospital_df["beds_covid"],ax=axes[0][1])
        axes[0][1].set_title('beds_covid')
        sns.boxplot(data=hospital_df, x = hospital_df["beds_noncrit"],ax=axes[0][2])
        axes[0][2].set_title('beds_noncrit')
        sns.boxplot(data=hospital_df, x = hospital_df["admitted_pui"],ax=axes[1][0])
        axes[1][0].set_title('admitted_pui')
        sns.boxplot(data=hospital_df,x = hospital_df["admitted_covid"],ax=axes[1][1])
        axes[1][1].set_title('admitted_covid')
        sns.boxplot(data=hospital_df, x = hospital_df["admitted_total"],ax=axes[1][2])
        axes[1][2].set_title('admitted_total')
        sns.boxplot(data=hospital_df, x = hospital_df["discharged_pui"],ax=axes[2][0])
        axes[2][0].set_title('discharged_pui')
        sns.boxplot(data=hospital_df,x = hospital_df["discharged_covid"],ax=axes[2][1])
        axes[2][1].set_title('discharged_covid')
        sns.boxplot(data=hospital_df, x = hospital_df["discharged_total"],ax=axes[2][2])
        axes[2][2].set_title('discharged_total')
        sns.boxplot(data=hospital_df, x = hospital_df["hosp_covid"],ax=axes[3][0])
        axes[3][0].set_title('hosp_covid')
        sns.boxplot(data=hospital_df,x = hospital_df["hosp_pui"],ax=axes[3][1])
        axes[3][1].set_title('hosp_pui')
        sns.boxplot(data=hospital_df, x = hospital_df["hosp_noncovid"],ax=axes[3][2])
        axes[3][2].set_title('hosp_noncovid')

        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    st.markdown('In this part, we tried to explore the data and generate initial insights. After reading in the datasets and filtering out the months needed, we have used functions such as _head(), info(), describe(), shape()_ to learn the basic information of each dataset and used _isna()_ and isnull() to find any existence of missing data. There are no null values found, hence in this part we did not do any action of filling nulls. Thus, boxplots for every column in the datasets were used to visualize the data distribution in terms of the shape, spreadness, min, mode, median, and outliers. The processes above were repeated for all eight datasets.')