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
deaths_dir = "dataset/deaths_state.csv"
vaccine_dir = "dataset/vax_state.csv"
aefi_dir = "dataset/aefi.csv"

def app():
    st.markdown('> Informative Insights from the Datasets')
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    chosen = st.selectbox(label = "Choose a Topic :", options=["Relationships between Covid-19 Vaccination and Daily NewCases", 
     "Covid-19 Daily New Cases and Daily New Deaths for each State", 
     "The admission and discharge flow in PKRC, hospital, ICUand ventilators usage situation of each state", 
     "The trend for vaccinated and cumulative vaccination reg-istration for each state",
     "The trend of R naught index value for each state",
     "The  interest  in  COVID-19  keywords  of  each  state  fromgoogle trends data"] )

    st.markdown(f"__{chosen} :__")
    start_date = "2021-07-01"
    end_date = "2021-09-30"


    if chosen == "Relationships between Covid-19 Vaccination and Daily NewCases":
        vaccine_df = pd.read_csv(vaccine_dir)
        deaths_df = pd.read_csv(deaths_dir)
        after_start_date = vaccine_df["date"] >= start_date
        before_end_date = vaccine_df["date"] <= end_date
        between_two_dates = after_start_date & before_end_date
        vaccine_df = vaccine_df.loc[between_two_dates]
        
        df = pd.DataFrame()
        df['vaccine'] = vaccine_df.daily.cumsum()
        df =(df-df.min())/(df.max()-df.min())
        df['date'] = vaccine_df.date
        df['state'] = vaccine_df.state
        df.set_index('date',inplace=True)
        state_case_df = pd.read_csv(state_case_dir)
        after_start_date = state_case_df["date"] >= start_date
        before_end_date = state_case_df["date"] <= end_date
        between_two_dates = after_start_date & before_end_date
        state_case_df = state_case_df.loc[between_two_dates]
        
        df2 = pd.DataFrame()
        df2['cases_new'] = state_case_df.cases_new
        df2 =(df2-df2.min())/(df2.max()-df2.min())
        df2['date'] = state_case_df.date

        df2.set_index('date',inplace=True)
        df = pd.concat([df, df2], axis=1)
        df3= df.copy()
        df = df.groupby('date').sum()

        sns.set(rc={'figure.figsize':(8,8)})
        sns.set(style='whitegrid')
        sns.scatterplot(data=df, x="vaccine", y="cases_new")
        plt.title('Effects of Vaccination on Daily New Cases')
        plt.xlabel('Vaccination')
        plt.ylabel('Daily New Cases')
        st.pyplot()
        st.markdown('Overall, vaccination has affected the daily Covid-19 new cases and play its role in Malaysia significantly.')
        st.markdown('Next, we look into each state and see the effetiveness of vaccination.')
        state = ['Johor', 'Kedah', 'Kelantan', 'Melaka', 'Negeri Sembilan', 'Pahang', 'Perak', 'Perlis', 'Pulau Pinang', 'Sabah', 'Sarawak', 'Selangor', 'Terengganu', 'W.P. Kuala Lumpur', 'W.P. Labuan', 'W.P. Putrajaya']
        sns.set(rc={'figure.figsize':(8,8)})
        graph = sns.FacetGrid(df3, row ="state", row_order=state ,hue ="state",height=4, aspect=1)
        # map the above form facetgrid with some attributes
        graph.map(plt.scatter, "vaccine", "cases_new", edgecolor ="w").add_legend()
        # show the object
        st.pyplot()
        st.markdown('In conclusion, the states daily new cases that has been affected a lot by vaccination are Johor, Kedah, Pulau Penang, Sabah, Selangor and W.P. Kuala Lumpur, the states is having a very obvious quick drop after certain point.  The states mentioned are having strong non-linear correlation between vaccination and daily new cases for each state from July until September.  This is probably related to the population density and the r-naught value of the states mentioned above.')
        affective_state = ['Johor', 'Kedah', 'Pulau Pinang', 'Sabah', 'Selangor', 'W.P. Kuala Lumpur']
        sns.set(rc={'figure.figsize':(8,8)})
        graph = sns.FacetGrid(df3, col ="state", col_order=affective_state ,hue ="state",height=4, aspect=1)
        # map the above form facetgrid with some attributes
        graph.map(plt.scatter, "vaccine", "cases_new", edgecolor ="w").add_legend()
        # show the object
        plt.show()
        st.pyplot()
        
    elif chosen == "Covid-19 Daily New Cases and Daily New Deaths for each State":
        deaths_df = pd.read_csv(deaths_dir)
        after_start_date = deaths_df["date"] >= start_date
        before_end_date = deaths_df["date"] <= end_date
        between_two_dates = after_start_date & before_end_date
        deaths_df = deaths_df.loc[between_two_dates]
        
        dc = pd.DataFrame()
        dc['deaths_cases'] = deaths_df.deaths_new
        dc =(dc-dc.min())/(dc.max()-dc.min())
        dc['date'] = deaths_df.date
        dc['state'] = deaths_df.state
        dc.set_index('date',inplace=True)
        
        state_case_df = pd.read_csv(state_case_dir)
        after_start_date = state_case_df["date"] >= start_date
        before_end_date = state_case_df["date"] <= end_date
        between_two_dates = after_start_date & before_end_date
        state_case_df = state_case_df.loc[between_two_dates]
        df2 = pd.DataFrame()
        df2['cases_new'] = state_case_df.cases_new
        df2 =(df2-df2.min())/(df2.max()-df2.min())
        df2['date'] = state_case_df.date

        df2.set_index('date',inplace=True)
        nc_dc = pd.concat([dc, df2], axis=1)
        mean = nc_dc.groupby('state').mean()
        print("Avg Deaths median value: " ,mean.deaths_cases.median())
        print("Avg Daily New Cases median value: " ,mean.cases_new.median())
        
        sns.set(rc={'figure.figsize':(20,8)})
        sns.set(style='whitegrid')
        sns.lineplot(data=mean)
        plt.title('Average Daily New Cases and New Deaths of each state')
        st.pyplot()
        
        state_list = mean.loc[(mean['deaths_cases'] >= 0.019) & (mean['cases_new'] >= 0.09)]
        st.table(state_list)
        
    elif chosen == "The admission and discharge flow in PKRC, hospital, ICUand ventilators usage situation of each state":
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

    elif chosen == "The trend for vaccinated and cumulative vaccination reg-istration for each state":
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

    elif chosen == "The trend of R naught index value for each state":
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

    elif chosen == "The  interest  in  COVID-19  keywords  of  each  state  fromgoogle trends data":
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

    st.markdown('In this part, we are trying to explore and generate informative insights from the Malaysia COVID-19 Cases and Vaccination datasets.')