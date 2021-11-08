import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly import tools
import plotly.express as px
import plotly.subplots as sp
# import geojson as gp
# import folium
# from streamlit_folium import folium_static

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
state_registration_dir = "dataset/vaxreg_state.csv"
state_vaccination_dir = "dataset/vax_state.csv"
icu_dir = "dataset/icu.csv"
malaysia_trends_coronavirus_dir = 'dataset/googletrends_malaysia_coronavirus.csv'
r_naught_dir = "dataset/r-naught-value - All.csv"
malaysia_trends_vaccine_comparison_dir = "dataset/googletrends_malaysia_vaccine_comparison.csv"
states_trends_astrazeneca_dir = "dataset/googletrends_states_astrazeneca.csv"
states_trends_moderna_dir= "dataset/googletrends_states_moderna.csv"
states_trends_pfizer_dir= "dataset/googletrends_states_pfizer.csv"
states_trends_sinovac_dir= "dataset/googletrends_states_sinovac.csv"
states_trends_symptoms_dir= "dataset/googletrends_states_symptoms.csv"
states_trends_vaccine_dir= "dataset/googletrends_states_vaccine.csv"
states_trends_cansino_dir= "dataset/googletrends_states_cansino.csv"
population_dir = "dataset/population.csv"

def app():
    st.markdown('> Informative Insights from the Datasets')
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    chosen = st.selectbox(label = "Choose a Topic :", options=["Relationships between Covid-19 Vaccination and Daily New Cases", 
     "Covid-19 Daily New Cases and Daily New Deaths for each State", 
     "The admission and discharge flow in PKRC, hospital, ICU and ventilators usage situation of each state", 
     "The trend for vaccinated and cumulative vaccination registration for each state",
     "The trend of R naught index value for each state",
     "The interest in COVID-19 keywords of each state from google trends data"] )

    st.markdown(f"__{chosen} :__")
    start_date = "2021-07-01"
    end_date = "2021-09-30"


    if chosen == "Relationships between Covid-19 Vaccination and Daily New Cases":
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

        fig1a = px.scatter(df, x="vaccine", y="cases_new",labels={
                     "vaccine": "Vaccination",
                     "cases_new": "Daily New Cases",
                 },
              title='Effects of Vaccination on Daily New Cases')

        fig1a.show()
        st.plotly_chart(fig1a, use_container_width=True)
        st.markdown('Overall, vaccination has affected the daily Covid-19 new cases and play its role in Malaysia significantly.')
        st.markdown('Next, we look into each state and see the effetiveness of vaccination.')

        fig1b = sp.make_subplots(rows=4, cols=4,
                                subplot_titles=("Johor", "Kedah" ,"Kelantan", "Melaka", 
                                                "Negeri Sembilan", "Pahang", "Perak", "Perlis", 
                                                "Pulau Pinang", "Sabah", "Sarawak", "Selangor", 
                                                "Terengganu", "W.P. Kuala Lumpur", "W.P. Labuan", "W.P. Putrajaya")) 

        dfJohor = pd.DataFrame(df3.loc[df3['state'] == 'Johor'])
        dfKedah = pd.DataFrame(df3.loc[df3['state'] == 'Kedah'])
        dfKelantan = pd.DataFrame(df3.loc[df3['state'] == 'Kelantan'])
        dfMelaka = pd.DataFrame(df3.loc[df3['state'] == 'Melaka'])
        dfNS = pd.DataFrame(df3.loc[df3['state'] == 'Negeri Sembilan'])
        dfPahang = pd.DataFrame(df3.loc[df3['state'] == 'Pahang'])
        dfPerak = pd.DataFrame(df3.loc[df3['state'] == 'Perak'])
        dfPerlis = pd.DataFrame(df3.loc[df3['state'] == 'Perlis'])
        dfPP = pd.DataFrame(df3.loc[df3['state'] == 'Pulau Pinang'])
        dfSabah = pd.DataFrame(df3.loc[df3['state'] == 'Sabah'])
        dfSarawak = pd.DataFrame(df3.loc[df3['state'] == 'Sarawak'])
        dfSelangor = pd.DataFrame(df3.loc[df3['state'] == 'Selangor'])
        dfTerengganu = pd.DataFrame(df3.loc[df3['state'] == 'Terengganu'])
        dfKL = pd.DataFrame(df3.loc[df3['state'] == 'W.P. Kuala Lumpur'])
        dfLabuan = pd.DataFrame(df3.loc[df3['state'] == 'W.P. Labuan'])
        dfPutrajaya = pd.DataFrame(df3.loc[df3['state'] == 'W.P. Putrajaya'])

        figure1 = px.scatter(dfJohor, x="vaccine", y="cases_new",color_discrete_sequence=['red'])
        figure2 = px.scatter(dfKedah, x='vaccine', y='cases_new',color_discrete_sequence=['fuchsia'])
        figure3 = px.scatter(dfKelantan, x="vaccine", y="cases_new",color_discrete_sequence=['lime'])
        figure4 = px.scatter(dfMelaka, x='vaccine', y='cases_new',color_discrete_sequence=['yellow'])
        figure5 = px.scatter(dfNS, x="vaccine", y="cases_new",color_discrete_sequence=['blue'])
        figure6 = px.scatter(dfPahang, x='vaccine', y='cases_new',color_discrete_sequence=['aqua'])
        figure7 = px.scatter(dfPerak, x="vaccine", y="cases_new",color_discrete_sequence=['orange'])
        figure8 = px.scatter(dfPerlis, x='vaccine', y='cases_new',color_discrete_sequence=['burlywood'])
        figure9 = px.scatter(dfPP, x="vaccine", y="cases_new",color_discrete_sequence=['brown'])
        figure10 = px.scatter(dfSabah, x='vaccine', y='cases_new',color_discrete_sequence=['gray'])
        figure11 = px.scatter(dfSarawak, x="vaccine", y="cases_new",color_discrete_sequence=['darkviolet'])
        figure12 = px.scatter(dfSelangor, x='vaccine', y='cases_new',color_discrete_sequence=['palegoldenrod'])
        figure13 = px.scatter(dfTerengganu, x="vaccine", y="cases_new",color_discrete_sequence=['pink'])
        figure14 = px.scatter(dfKL, x='vaccine', y='cases_new',color_discrete_sequence=['seagreen'])
        figure15 = px.scatter(dfLabuan, x="vaccine", y="cases_new",color_discrete_sequence=['skyblue'])
        figure16 = px.scatter(dfPutrajaya, x='vaccine', y='cases_new',color_discrete_sequence=['salmon'])

        figure1_traces = []
        figure2_traces = []
        figure3_traces = []
        figure4_traces = []
        figure5_traces = []
        figure6_traces = []
        figure7_traces = []
        figure8_traces = []
        figure9_traces = []
        figure10_traces = []
        figure11_traces = []
        figure12_traces = []
        figure13_traces = []
        figure14_traces = []
        figure15_traces = []
        figure16_traces = []

        for trace in range(len(figure1["data"])):
            figure1_traces.append(figure1["data"][trace])
        for trace in range(len(figure2["data"])):
            figure2_traces.append(figure2["data"][trace])
        for trace in range(len(figure3["data"])):
            figure3_traces.append(figure3["data"][trace])
        for trace in range(len(figure4["data"])):
            figure4_traces.append(figure4["data"][trace])
        for trace in range(len(figure5["data"])):
            figure5_traces.append(figure5["data"][trace])
        for trace in range(len(figure6["data"])):
            figure6_traces.append(figure6["data"][trace])
        for trace in range(len(figure7["data"])):
            figure7_traces.append(figure7["data"][trace])
        for trace in range(len(figure8["data"])):
            figure8_traces.append(figure8["data"][trace])
        for trace in range(len(figure9["data"])):
            figure9_traces.append(figure9["data"][trace])
        for trace in range(len(figure10["data"])):
            figure10_traces.append(figure10["data"][trace])
        for trace in range(len(figure11["data"])):
            figure11_traces.append(figure11["data"][trace])
        for trace in range(len(figure12["data"])):
            figure12_traces.append(figure12["data"][trace])
        for trace in range(len(figure13["data"])):
            figure13_traces.append(figure13["data"][trace])
        for trace in range(len(figure14["data"])):
            figure14_traces.append(figure14["data"][trace])
        for trace in range(len(figure15["data"])):
            figure15_traces.append(figure15["data"][trace])
        for trace in range(len(figure16["data"])):
            figure16_traces.append(figure16["data"][trace])

        for traces in figure1_traces:
            fig1b.append_trace(traces, row=1, col=1)
        for traces in figure2_traces:
            fig1b.append_trace(traces, row=1, col=2)
        for traces in figure3_traces:
            fig1b.append_trace(traces, row=1, col=3)
        for traces in figure4_traces:
            fig1b.append_trace(traces, row=1, col=4)
        for traces in figure5_traces:
            fig1b.append_trace(traces, row=2, col=1)
        for traces in figure6_traces:
            fig1b.append_trace(traces, row=2, col=2)
        for traces in figure7_traces:
            fig1b.append_trace(traces, row=2, col=3)
        for traces in figure8_traces:
            fig1b.append_trace(traces, row=2, col=4)
        for traces in figure9_traces:
            fig1b.append_trace(traces, row=3, col=1)
        for traces in figure10_traces:
            fig1b.append_trace(traces, row=3, col=2)
        for traces in figure11_traces:
            fig1b.append_trace(traces, row=3, col=3)
        for traces in figure12_traces:
            fig1b.append_trace(traces, row=3, col=4)
        for traces in figure13_traces:
            fig1b.append_trace(traces, row=4, col=1)
        for traces in figure14_traces:
            fig1b.append_trace(traces, row=4, col=2)
        for traces in figure15_traces:
            fig1b.append_trace(traces, row=4, col=3)
        for traces in figure16_traces:
            fig1b.append_trace(traces, row=4, col=4)
        fig1b.update_traces(marker={'size': 4})
        fig1b.show()
        st.plotly_chart(fig1b, use_container_width=True)
        st.markdown('In conclusion, the states daily new cases that has been affected a lot by vaccination are Johor, Kedah, Pulau Penang, Sabah, Selangor and W.P. Kuala Lumpur, the states is having a very obvious quick drop after certain point.  The states mentioned are having strong non-linear correlation between vaccination and daily new cases for each state from July until September.  This is probably related to the population density and the r-naught value of the states mentioned above.')
        
        fig1c = sp.make_subplots(rows=2, cols=3,print_grid=True,
                        vertical_spacing=0.15,
                        horizontal_spacing=0.03,
                        x_title="Vaccination", 
                        y_title="Daily New Cases",
                        subplot_titles=("Johor", "Kedah" ,"Pulau Pinang",
                                        "Sabah", "Selangor", "W.P. Kuala Lumpur")) 

        for trace in range(len(figure1["data"])):
            figure1_traces.append(figure1["data"][trace])
        for trace in range(len(figure2["data"])):
            figure2_traces.append(figure2["data"][trace])
        for trace in range(len(figure9["data"])):
            figure9_traces.append(figure9["data"][trace])
        for trace in range(len(figure10["data"])):
            figure10_traces.append(figure10["data"][trace])
        for trace in range(len(figure12["data"])):
            figure12_traces.append(figure12["data"][trace])
        for trace in range(len(figure14["data"])):
            figure14_traces.append(figure14["data"][trace])

        for traces in figure1_traces:
            fig1c.append_trace(traces, row=1, col=1)
        for traces in figure2_traces:
            fig1c.append_trace(traces, row=1, col=2)
        for traces in figure9_traces:
            fig1c.append_trace(traces, row=1, col=3)
        for traces in figure10_traces:
            fig1c.append_trace(traces, row=2, col=1)
        for traces in figure12_traces:
            fig1c.append_trace(traces, row=2, col=2)
        for traces in figure14_traces:
            fig1c.append_trace(traces, row=2, col=3)
        fig1b.update_traces(marker={'size': 4})
        fig1c.show()
        st.plotly_chart(fig1c, use_container_width=True)
        
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
        
        mean_p=mean.reset_index()
        fig2c = go.Figure()
        fig2c.add_trace(go.Scatter(x=mean_p['state'], y=mean_p['deaths_cases'],
                            mode='lines',
                            name='Deaths Cases'))
        fig2c.add_trace(go.Scatter(x=mean_p['state'], y=mean_p['cases_new'],
                            mode='lines',
                            name='New Cases'))
        fig2c.update_layout(
            title="Average Daily New Cases and New Deaths of each state",
            xaxis_title="State")#, yaxis_title=""
        fig2c.show()
        st.plotly_chart(fig2c, use_container_width=True)
        
        state_list = mean.loc[(mean['deaths_cases'] >= 0.019) & (mean['cases_new'] >= 0.09)]
        st.table(state_list)
        
    elif chosen == "The admission and discharge flow in PKRC, hospital, ICU and ventilators usage situation of each state":
        pkrc_df = pd.read_csv(pkrc_dir)
        after_start_date = pkrc_df["date"] >= start_date
        before_end_date = pkrc_df["date"] <= end_date
        between_two_dates = after_start_date & before_end_date
        pkrc_df = pkrc_df.loc[between_two_dates]

        fig3a = px.line(pkrc_df, x="date", y="admitted_covid", color='state',
              labels={
                     "date": "Date",
                     "admitted_covid": "Number of Individuals Admitted to PKRC",
                     "state": "State"
                 }, 
              title='Daily PKRC Admission Flow')
        fig3a.show()
        st.plotly_chart(fig3a,use_container_width=True)
        st.text('Based on the line graph of daily PKRC admission flow above, we can see that Sabah, Selangor, Johor, and Pahang have higher number of COVID-19 patients admitted to PKRC than other states over the three months. Sabah has the highest number of admitted patients per day. In general, the patients admission rate of each state fluctuated over the three months but has shown a downward trend and lower rate toward September compared to July.')

        fig3b = px.line(pkrc_df, x="date", y="discharge_covid", color='state',
              labels={
                     "date": "Date",
                     "discharge_covid": "Number of Individuals Discharged from PKRC",
                     "state": "State"
                 }, 
              title='Daily PKRC Discharge Flow')
        fig3b.show()
        st.plotly_chart(fig3b, use_container_width=True)
        st.text('Based on the line graph of daily PKRC discharge flow above, we can see that Selangor, Sabah, Johor, and Pahang have higher number of COVID-19 patients discharged from PKRC than other states over the three months. This may be because they already have more patients. Selangor has the highest number of discharged patients per day. In general, the patients admission rate of each state fluctuated over the three months and has similar pattern to the admission rate.')

        fig3c = px.line(pkrc_df, x="date", y="pkrc_covid", color='state',
              labels={
                     "date": "Date",
                     "pkrc_covid": "Number of Individuals in PKRC",
                     "state": "State"
                 }, 
              title='PKRC Total COVID-19 Patients Flow')
        fig3c.show()
        st.plotly_chart(fig3c, use_container_width=True)
        st.text('Based on the line graph of PKRC total COVID-19 patients flow above, we can see that Sarawak, Pahang, Sabah, and Selangor have higher total number of COVID-19 patients PKRC than other states over the three months. Sarawak has the highest total number of COVID-19 patients PKRC per day. In general, the total number of COVID-19 patients of each state fluctuated over the three months except for Perlis, W.P. Labuan, and Kedah.')

        st.title('Hospital')
        hospital_df = pd.read_csv(hospital_dir)
        after_start_date = hospital_df["date"] >= start_date
        before_end_date = hospital_df["date"] <= end_date
        between_two_dates = after_start_date & before_end_date
        hospital_df = hospital_df.loc[between_two_dates]

        fig3d = px.line(hospital_df, x="date", y="admitted_covid", color='state',
              labels={
                     "date": "Date",
                     "admitted_covid": "Number of Individuals Admitted to Hospital",
                     "state": "State"
                 }, 
              title='Daily Hospital Admission Flow')
        fig3d.show()
        st.plotly_chart(fig3d, use_container_width=True)      
        st.text('Based on the line graph of daily hospital admission flow above, we can see that Selangor, Sarawak, and Johor have higher number of COVID-19 patients admitted to hospital than other states over the three months. Selangor has the highest number of admitted patients per day. These 3 states are now declining toward September while other states maintain the steady rate for the daily number of patients admitted.')

        fig3e = px.line(hospital_df, x="date", y="discharged_covid", color='state',
              labels={
                     "date": "Date",
                     "discharged_covid": "Number of Individuals Discharged to Hospital",
                     "state": "State"
                 }, 
              title='Daily Hospital Discharge Flow')
        fig3e.show()
        st.plotly_chart(fig3e, use_container_width=True)
        st.text('Based on the line graph of daily hospital discharge flow above, we can see that Selangor, Sarawak, and Johor have higher number of COVID-19 patients discharged from hospital than other states over the three months. This may be because they already have more patients. Sarawak has the highest number of discharged patients per day. In general, the patients admission rate of each state fluctuated over the three months and has similar pattern to the admission rate.')

        fig3f = px.line(hospital_df, x="date", y="hosp_covid", color='state',
              labels={
                     "date": "Date",
                     "hosp_covid": "Number of Individuals in Hospital",
                     "state": "State"
                 }, 
              title='Hospital Total COVID-19 Patients Flow')
        fig3f.show()
        st.plotly_chart(fig3f, use_container_width=True)
        st.text('Based on the line graph of hospital total COVID-19 patients flow above, we can see that Selangor and Johor have higher total number of COVID-19 patients PKRC than other states over the three months. W.P. Kuala Lumpur has higher total patients until mid of August than other states, and starts to drop visibly toward September. Selangor has the highest total number of COVID-19 patients hospital per day. In general, the total number of COVID-19 patients of each state fluctuated over the three months except for Perlis, W.P. Putrajaya, and W.P. Labuan.')

        st.title('ICU')
        icu_df = pd.read_csv(icu_dir)
        after_start_date = icu_df["date"] >= start_date
        before_end_date = icu_df["date"] <= end_date
        between_two_dates = after_start_date & before_end_date
        icu_df = icu_df.loc[between_two_dates]

        fig3g = px.line(icu_df, x="date", y="icu_covid", color='state',
              labels={
                     "date": "Date",
                     "icu_covid": "Number of Individuals under Intensive Care",
                     "state": "State"
                 }, 
              title='ICU Total COVID-19 Patients Flow')
        fig3g.show()
        st.plotly_chart(fig3g, use_container_width=True)
        st.text('Based on the line graph of daily ICU total COVID-19 patients flow above, we can see that Selangor has a overwhelming higher number of COVID-19 patients admitted to ICU than other states over the three months. Selangor also has the highest number of admitted patients per day. In general, the total number of ICU COVID-19 patients of all states are now dropping or staying steady toward September except Sabah.')

        fig3h = px.line(icu_df, x="date", y="vent_covid", color='state',
              labels={
                     "date": "Date",
                     "vent_covid": "Number of Individuals on Mechanical Ventilation under Intensive Care",
                     "state": "State"
                 }, 
              title='ICU Total COVID-19 Patients on Mechanical Ventilation Flow')
        fig3h.show()
        st.plotly_chart(fig3h, use_container_width=True)
        st.text('Based on the line graph of daily ICU total COVID-19 patient on mechanical ventilation flow above, we can see that Selangor and W.P. Kuala Lumpur have higher number of COVID-19 patients needed the ventilator assistance than other states over the three months. Selangor has the highest number of patients on ventilation per day. In general, the patients admission rate of each state fluctuated over the three months and remained steady whereas Selangor and W.P. Kuala Lumpur are now declining toward September.')
        st.text('In conclusion, according to the graphs above, the states that require more attention are Selangor,Johor, Sabah, W.P. Kuala Lumpus, Sarawak, and Pahang. Although the reasons behind them having more patients may be because they have more population, it is still obvious that they need more attention from government to put in efforts and works to improve the situation.')


    elif chosen == "The trend for vaccinated and cumulative vaccination registration for each state":
        state_registration_df = pd.read_csv(state_registration_dir)
        state_registration_df_copy = state_registration_df.copy()
        state_registration_df_copy.drop(state_registration_df_copy.columns.difference(['date','state','total']), 1, inplace=True)
        state_registration_df_copy['date'] = pd.to_datetime(state_registration_df_copy['date'], format = '%Y-%m-%d')
        
        population_df = pd.read_csv(population_dir)

        population_df.drop(population_df.columns.difference(['state','pop']), 1, inplace=True)
        merged_df =  pd.merge(state_registration_df, population_df, on='state')
        merged_df['rate'] = merged_df['total'] / merged_df['pop']
        merged_df.drop(merged_df.columns.difference(['date','state','total','rate']), 1, inplace=True)
        merged_df['date'] = pd.to_datetime(merged_df['date'], format = '%Y-%m-%d')

        fig4a = px.line(merged_df, x="date", y="total", color="state",
                labels={
                     "date": "Date",
                     "total": "Total Vaccination Registrations",
                 }, 
              title='Total Vaccination Registrations Per State')
        fig4a.show()
        st.plotly_chart(fig4a, use_container_width=True)
        st.markdown('From the chart, it can be seen that the selangor has been the top in the population of registered citizens. It might because the people lived in Selangor are having more accurate information about vaccine and more educated.')

        fig4a1 = px.line(merged_df, x="date", y="rate", color="state",
                labels={
                     "date": "Date",
                     "rate": "Total Vaccination Registration's Rate",
                 }, 
              title='Total Vaccination Registrations Rate Per State')
        fig4a1.show()
        st.plotly_chart(fig4a1, use_container_width=True)
        st.markdown('From the chart, it can be seen that the Kuala Lumpur has been the top in the rate of population of registered citizens. And in contrast, Sabah has been in the bottom, it might be because the citizens in Sabah are having transportation issue to get to the vaccination center and difficult to receive the correct information about vaccination.')



        state_vaccination_df = pd.read_csv(state_vaccination_dir)
        state_vaccination_df_copy = state_vaccination_df.copy()
        state_vaccination_df_copy.drop(state_vaccination_df_copy.columns.difference(['date','state','daily']), 1, inplace=True)
        state_vaccination_df_copy['date'] = pd.to_datetime(state_vaccination_df_copy['date'], format = '%Y-%m-%d')

        fig4b = px.line(state_vaccination_df_copy, x="date", y="daily", color="state",
                labels={
                     "date": "Date",
                     "daily": "Daily Vaccinations",
                 }, 
              title='Daily Vaccinations Per State')
        fig4b.show()
        st.plotly_chart(fig4b, use_container_width=True)
        st.markdown('From the chart, we can see the Selangor had been the top before September in daily new vaccination but there is a sudden drop before entered September. It might be becuase of the government was having several vaccination boost plan in Selangor before September and most of the citizens were fully vaccinated before September. ')

    elif chosen == "The trend of R naught index value for each state":
        r_naught_df = pd.read_csv(r_naught_dir)
        r_naught_df['date'] = pd.to_datetime(r_naught_df['date'], format='%d/%m/%Y')
        after_start_date = r_naught_df["date"] >= start_date
        before_end_date = r_naught_df["date"] <= end_date
        between_two_dates = after_start_date & before_end_date
        r_naught_df = r_naught_df.loc[between_two_dates]
        r_naught_df = r_naught_df._convert(numeric=True)
        r_naught_df = r_naught_df.replace(np.nan, 0)
        r_naught_df['date'] = pd.to_datetime(r_naught_df['date'], format = '%Y-%m-%d')
        #r_naught_df.set_index('date', inplace=True)
        st.markdown('We used an extra dataset which is the R-Naught Index value in our project. There were some r naught indexes not published by MOH. We have filled in the null values with r naught index value of the day before. There were also some string values in the dataset, such as “3 cases” in a single day. For those data, we decided to replace it with zero since the number of cases is so small that we cannot calculate for the R-naught index.')
        fig5a = go.Figure()
        fig5a.add_trace(go.Scatter(x=r_naught_df['date'], y=r_naught_df['Selangor'],
                                    mode='lines',
                                    name='Selangor'))
        fig5a.add_trace(go.Scatter(x=r_naught_df['date'], y=r_naught_df['Sabah'],
                                    mode='lines',
                                    name='Sabah'))
        fig5a.add_trace(go.Scatter(x=r_naught_df['date'], y=r_naught_df['P. Pinang'],
                                    mode='lines',
                                    name='Pulau Pinang'))
        fig5a.add_trace(go.Scatter(x=r_naught_df['date'], y=r_naught_df['Johor'],
                                    mode='lines',
                                    name='Johor'))
        fig5a.add_trace(go.Scatter(x=r_naught_df['date'], y=r_naught_df['Kelantan'],
                                    mode='lines',
                                    name='Kelantan'))
        fig5a.add_trace(go.Scatter(x=r_naught_df['date'], y=r_naught_df['Melaka'],
                                    mode='lines',
                                    name='Melaka'))
        fig5a.add_trace(go.Scatter(x=r_naught_df['date'], y=r_naught_df['WP Kuala Lumpur'],
                                    mode='lines',
                                    name='W.P. Kuala Lumpur'))
        fig5a.add_trace(go.Scatter(x=r_naught_df['date'], y=r_naught_df['WP Labuan'],
                                    mode='lines',
                                    name='W.P. Labuan'))
        fig5a.add_trace(go.Scatter(x=r_naught_df['date'], y=r_naught_df['WP Putrajaya'],
                                    mode='lines',
                                    name='W.P. Putrajaya'))
        fig5a.add_trace(go.Scatter(x=r_naught_df['date'], y=r_naught_df['Negeri Sembilan'],
                                    mode='lines',
                                    name='Negeri Sembilan'))
        fig5a.add_trace(go.Scatter(x=r_naught_df['date'], y=r_naught_df['Kedah'],
                                    mode='lines',
                                    name='Kedah'))
        fig5a.add_trace(go.Scatter(x=r_naught_df['date'], y=r_naught_df['Pahang'],
                                    mode='lines',
                                    name='Pahang'))
        fig5a.add_trace(go.Scatter(x=r_naught_df['date'], y=r_naught_df['Perak'],
                                    mode='lines',
                                    name='Perak'))
        fig5a.add_trace(go.Scatter(x=r_naught_df['date'], y=r_naught_df['Terengganu'],
                                    mode='lines',
                                    name='Terengganu'))
        fig5a.add_trace(go.Scatter(x=r_naught_df['date'], y=r_naught_df['Sarawak'],
                                    mode='lines',
                                    name='Sarawak'))
        fig5a.add_trace(go.Scatter(x=r_naught_df['date'], y=r_naught_df['Perlis'],
                                    mode='lines',
                                    name='Perlis'))
        fig5a.update_layout(
                    title="Daily R-Naught Values of Each State",
                    xaxis_title="State", yaxis_title="R-Naught Value")

        fig5a.show()
        st.plotly_chart(fig5a, use_container_width=True)
        st.markdown('From the chart, we can find that the RNaught Value of two states fluctuates wildly. They are Perlis and Labuan. For Perlis, we suspect the high RNaught Value is because of the high density of population as the area of the state is the smallest in Malaysia. For Labuan, we guess the high registration rate helps prevent the spread of Covid-19 as Labuan is one of the states with the highest vaccination rate in the early covid period.')
        #Create figure object
        # import urllib.request, json 
        # with urllib.request.urlopen("https://gist.githubusercontent.com/heiswayi/81a169ab39dcf749c31a/raw/b2b3685f5205aee7c35f0b543201907660fac55e/malaysia.geojson") as url:
        #     malaysia_geo = geojson.load(url)
        # #     print(data)
        # with open('./msia-states.json') as f:
        #     malaysia_geo = geojson.load(f)
        # Dataset = pd.DataFrame({"City/County":['Johor'],'Population':[3213214]})
        # fig = go.Figure(
        #     go.Choroplethmapbox(
        #         geojson = malaysia_geo, #Assign geojson file
        #         featureidkey = "properties.name_1", #Assign feature key
        #         locations = Dataset["City/County"], #Assign location data
        #         z = Dataset["Population"], #Assign information data
        #         zauto = True,
        #         colorscale = 'viridis',
        #         showscale = True,
        #     )
        # )

        #Update layout
        # fig.update_layout(
        #     mapbox_style = "carto-positron", #Decide a style for the map
        #     mapbox_zoom = 6, #Zoom in scale
        #     mapbox_center = {"lat": 4.2105, "lon": 101.9758}, #Center location of the map
        # )
        # fig.show()
        # st.plotly_chart(fig, use_container_width=True)

        # jsonl = f"masia-states.json"
        # m = folium.Map(location=[4.2105,101.9758], titles = 'Covid 19 Regsitration', name="Light Map", zoom_start=5, attr="My data attribution")
        # df_map = gp.read_file(f"masia-states.json")
        # states = folium.Choropleth(
        #     geo_data = df_map,
        #     data = df_map,
        #     key_on = "feature.properties.name_1",
        #     columns=['name_1', 'Cases'],
        #     fill_color="YlOrRd",
        #     fill_opacity=0.7,
        #     line_opacity=0.5,
        #     legend_name="Cases",
        #     reset=True
        # ).add_to(m)

        # states.geojson.add_child(
        #     folium.features.GeoJsonTooltip(fields=['name_1','Cases'], aliases=['State: ','Cases: '])
        # )

        # folium_static(m)
        # make_map_responsive="""
        # <style>
        # [title~="st.iframe"]{width:100%}
        # </style>
        # """
        # st.markdown(make_map_responsive, unsafe_allow_html=True)


    elif chosen == "The interest in COVID-19 keywords of each state from google trends data":
        st.title('Malaysia Overall Trends')
        malaysia_trends_coronavirus_df = pd.read_csv(malaysia_trends_coronavirus_dir)
        malaysia_trends_coronavirus_df['Date'] = pd.to_datetime(malaysia_trends_coronavirus_df['Date'], format='%d/%m/%Y')

        fig6a = px.line(malaysia_trends_coronavirus_df, x="Date", y="Interest Score", title='Malaysia Search Trend of Coronavirus')
        fig6a.show()
        st.plotly_chart(fig6a, use_container_width=True)
        st.text("Based on the graph above, we can see that the search trend fluctuated over the one year (19th Oct 2020 - 19th Oct 2021), people living in Malaysia searched about 'coronavirus' the most around the May of 2021. It is noticeable that people's interest in coronavirus/COVID-19 will be higher when the confirmed cases are higher because people will like to know the details of the recent situation even more.")

        malaysia_trends_vaccine_comparison_df = pd.read_csv(malaysia_trends_vaccine_comparison_dir)
        malaysia_trends_vaccine_comparison_df['Date'] = pd.to_datetime(malaysia_trends_vaccine_comparison_df['Date'], format='%d/%m/%Y')
        
        melted_df = malaysia_trends_vaccine_comparison_df.melt('Date', var_name='Vaccine Types',  value_name='Interest Score')
        fig6b = px.line(melted_df, x="Date", y="Interest Score", color="Vaccine Types",
              title='Malaysia Search Trend of Different Vaccines')
        fig6b.show()
        st.plotly_chart(fig6b, use_container_width=True)
        st.text('Based on the graph above, we can see that the overall search trend for 5 types of vaccines: Moderna, Sinovac, AstraZeneca, Cansino, and Pfizer in one year span. We noticed that most poeple were interested in AstraZeneca vaccines, followed by Sinovac vaccines and Pfizer vaccines. There were less interest towards Moderna and Cansino vaccines. This may be due to the different quantities of each type of vaccine imported and used by the Malaysia government.')

        st.title('Malaysia States Trends')
        st.subheader('Astraneca')
        states_trends_astrazeneca_df =  pd.read_csv(states_trends_astrazeneca_dir)
        fig6c = px.line(states_trends_astrazeneca_df, x="State", y="Interest Score",markers=True, 
                        color_discrete_sequence=['maroon'], title='Malaysia Search Trend of AstraZeneca')
        fig6c.show()
        st.plotly_chart(fig6c, use_container_width=True)

        st.subheader('Cansino')
        states_trends_cansino_df =  pd.read_csv(states_trends_cansino_dir)
        fig6d = px.line(states_trends_cansino_df, x="State", y="Interest Score", markers=True,
                        color_discrete_sequence=['purple'], title='Malaysia Search Trend of Cansino')
        fig6d.show()
        st.plotly_chart(fig6d, use_container_width=True)

        st.subheader('Moderna')
        states_trends_moderna_df =  pd.read_csv(states_trends_moderna_dir)
        fig6e = px.line(states_trends_moderna_df, x="State", y="Interest Score", markers=True,
                        color_discrete_sequence=['green'],title='Malaysia Search Trend of Moderna')
        fig6e.show()
        st.plotly_chart(fig6e, use_container_width=True)

        st.subheader('Pfizer')
        states_trends_pfizer_df =  pd.read_csv(states_trends_pfizer_dir)
        fig6f = px.line(states_trends_pfizer_df, x="State", y="Interest Score", markers=True,
                        color_discrete_sequence=['olive'],title='Malaysia Search Trend of Pfizer')
        fig6f.show()
        st.plotly_chart(fig6f, use_container_width=True)

        st.subheader('Sinovac')
        states_trends_sinovac_df =  pd.read_csv(states_trends_sinovac_dir)
        fig6g = px.line(states_trends_sinovac_df, x="State", y="Interest Score", markers=True,
                        color_discrete_sequence=['navy'], title='Malaysia Search Trend of Sinovac')
        fig6g.show()
        st.plotly_chart(fig6g, use_container_width=True)

        st.subheader('Covid-19 Symptoms')
        states_trends_symptoms_df =  pd.read_csv(states_trends_symptoms_dir)
        fig6h = px.line(states_trends_symptoms_df, x="State", y="Interest Score", markers=True,
                        color_discrete_sequence=['chocolate'], title='Malaysia Search Trend of Symptoms')
        fig6h.show()
        st.plotly_chart(fig6h, use_container_width=True)
        st.text('The graph above showed that Johor and Perlis has searched the keyword "Covid Symptoms" the most and the least respectively')

        st.subheader('Covid-19 Vaccine')
        states_trends_vaccine_df =  pd.read_csv(states_trends_vaccine_dir)
        fig6i = px.line(states_trends_vaccine_df, x="State", y="Interest Score", markers=True,
                        color_discrete_sequence=['deeppink'], title='Malaysia Search Trend of Vaccine')
        fig6i.show()
        st.plotly_chart(fig6i, use_container_width=True)
        st.text('The graph above showed that Selangor and W.P. Labuan has searched the keyword "COVID-19 Vaccine" the most and the least respectively.')

        st.subheader('Total Interest Score by States')
        df_list = [states_trends_astrazeneca_df,states_trends_cansino_df, 
                states_trends_moderna_df, states_trends_pfizer_df,
                states_trends_sinovac_df, states_trends_symptoms_df, states_trends_vaccine_df]
        new_interest_df = df_list[0]

        for df_ in df_list[1:]:
            new_interest_df = new_interest_df.merge(df_, on='State')

        new_interest_df.set_index('State')
        new_interest_df['Total Interest Score'] = new_interest_df['Total Interest Score'] = new_interest_df.sum(axis=1)
        total_interest_df = new_interest_df[['State','Total Interest Score']]

        st.header('State Cases')
        full_start_date = "2020-10-19"
        full_end_date = "2021-10-19"
        full_state_case_df = pd.read_csv(state_case_dir)
        after_start_date = full_state_case_df["date"] >= full_start_date
        before_end_date = full_state_case_df["date"] <= full_end_date
        between_two_dates = after_start_date & before_end_date
        full_state_case_df = full_state_case_df.loc[between_two_dates]
        full_state_case_df = full_state_case_df[['state','cases_new']]
        total_state_case_df = full_state_case_df.groupby(['state'],as_index=False).agg({'cases_new': 'sum'})
        total_state_case_df = total_state_case_df.rename(columns={'state': 'State', 'cases_new': 'Total Cases'})
        total_case_interest_df = pd.merge(total_state_case_df, total_interest_df, on=['State','State'])
        total_case_interest_df = total_case_interest_df.sort_values(by=['Total Cases'],ascending=False)

        normalized_total_case_interest_df = total_case_interest_df.copy()
        normalized_total_case_interest_df['Total Cases'] = normalized_total_case_interest_df['Total Cases'] /normalized_total_case_interest_df['Total Cases'].abs().max()
        normalized_total_case_interest_df['Total Interest Score'] = normalized_total_case_interest_df['Total Interest Score'] /normalized_total_case_interest_df['Total Interest Score'].abs().max()

        df_melted = normalized_total_case_interest_df.melt("State",var_name="Total Cases & Total Interest Score",value_name="Normalized Data")
        fig6j = px.line(df_melted, x="State", y="Normalized Data", color='Total Cases & Total Interest Score',markers=True,
              title='States: Cases vs Interest Score')
        fig6j.show()
        st.plotly_chart(fig6j, use_container_width=True)

        correlation = normalized_total_case_interest_df['Total Cases'].corr(normalized_total_case_interest_df['Total Interest Score']) 
        print(correlation)
        corr = normalized_total_case_interest_df.corr()
        fig6k = px.imshow(corr)
        fig6k.show()
        st.plotly_chart(fig6k, use_container_width=True)
        st.text('Based on the graph above and correlation coefficient of 0.6285571402052665, the total cases and total interest score of each state have a moderate positive correlation. The higher the confirmed COVID-19 cases, the higher the interest score where people tend to search more about COVID-19.')
        st.text("In short, this section demonstrated basic exploration and analyzation on google trends data in Malaysia. The search trend of 'coronavirus' and other keywords fluctuated over the one year span (19th Oct 2020 - 19th Oct 2021) due to various reasons. It is noticeable that interest score for 'coronavirus' topped around the May of 2021 because of the pandemic situation started to worsen. The correlation between COVID-19 cases and people's interest towards COVID-19 is moderate positive. People in different states have different interest levels for different keywords, whichever is more related to themselves will be searched more. For example, Sabah topped the 'Cansino' interest score while other states showed lower interest, because Cansino vaccines are mostly available and vaccinated in Sabah. This section demonstrated basic exploration and analyzation on google trends data.")

    st.markdown('In this part, we are trying to explore and generate informative insights from the Malaysia COVID-19 Cases and Vaccination datasets.')


    