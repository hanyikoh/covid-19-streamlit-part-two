import numpy as np
import pandas as pd
import streamlit as st
#import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
from kneed import KneeLocator, DataGenerator as dg
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
import plotly.graph_objects as go
import plotly.offline as py
import plotly.express as px

start_date = "2021-07-01"
end_date = "2021-09-30"
malaysia_vaccination_dir = "dataset/vax_malaysia.csv"
malaysia_vaccination_df = pd.read_csv(malaysia_vaccination_dir)
after_start_date = malaysia_vaccination_df["date"] >= start_date
before_end_date = malaysia_vaccination_df["date"] <= end_date
between_two_dates = after_start_date & before_end_date
malaysia_vaccination_df = malaysia_vaccination_df.loc[between_two_dates]

malaysia_case_dir = "dataset/cases_malaysia.csv"
malaysia_case_df = pd.read_csv(malaysia_case_dir)
after_start_date = malaysia_case_df["date"] >= start_date
before_end_date = malaysia_case_df["date"] <= end_date
between_two_dates = after_start_date & before_end_date
malaysia_case_df = malaysia_case_df.loc[between_two_dates]

vaccines_type_df = malaysia_vaccination_df[['date','pfizer1','pfizer2','sinovac1','sinovac2','astra1','astra2','cansino']]
vaccines_type_df['Pfizer'] = vaccines_type_df.loc[:, ('pfizer1', 'pfizer2')].sum(axis=1)
vaccines_type_df['Sinovac'] = vaccines_type_df.loc[:, ('sinovac1','sinovac2')].sum(axis=1)
vaccines_type_df['AstraZeneca'] = vaccines_type_df.loc[:, ('astra1','astra2')].sum(axis=1)
vaccines_type_df['Cansino'] = vaccines_type_df.loc[:, ('cansino')]
vaccines_type_df = vaccines_type_df[['Pfizer','Sinovac','AstraZeneca','Cansino']]
vaccines_type_df.reset_index(drop=True, inplace=True)
vaccines_type_df =(vaccines_type_df-vaccines_type_df.min())/(vaccines_type_df.max()-vaccines_type_df.min())
vaccines_date_df = malaysia_vaccination_df[['date']]
vaccines_date_df['Pfizer'] = vaccines_type_df['Pfizer'].values
vaccines_date_df['Sinovac'] = vaccines_type_df['Sinovac'].values
vaccines_date_df['AstraZeneca'] = vaccines_type_df['AstraZeneca'].values
vaccines_date_df['Cansino'] = vaccines_type_df['Cansino'].values

def app():
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=vaccines_date_df['date'], y=vaccines_date_df['Pfizer'],
                            mode='markers', name='Pfizer'))
    fig1.add_trace(go.Scatter(x=vaccines_date_df['date'], y=vaccines_date_df['Sinovac'],
                            mode='markers', name='Sinovac'))
    fig1.add_trace(go.Scatter(x=vaccines_date_df['date'], y=vaccines_date_df['AstraZeneca'],
                            mode='markers',name='AstraZeneca'))
    fig1.add_trace(go.Scatter(x=vaccines_date_df['date'], y=vaccines_date_df['Cansino'],
                            mode='markers', name='Cansino'))
    fig1.update_layout(xaxis_title="Date")
    fig1.show()
    st.plotly_chart(fig1, use_container_width=True)

    ss = StandardScaler()
    X = ss.fit_transform(vaccines_type_df)

    distortions = []
    K = range(1,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(vaccines_type_df)
        distortions.append(kmeanModel.inertia_)
 
    x, y = dg.convex_decreasing()
    kn = KneeLocator(x, y, curve='convex', direction='decreasing')
    print('Optimal k: ',kn.knee)

    df = pd.DataFrame({'Clusters': K, 'Distortions': distortions})
    fig2 = (px.line(df, x='Clusters', y='Distortions', template='seaborn')).update_traces(mode='lines+markers')
    fig2.add_vline(x=kn.knee, line_width=2, line_dash="dash", line_color="green")
    fig2.update_layout(title="The Elbow Method Showing The Optimal K")
    fig2.show()
    st.plotly_chart(fig2, use_container_width=True)
    
    kmeans = KMeans(n_clusters=2, init='random')
    kmeans = kmeans.fit_predict(vaccines_type_df)
    vaccines_type_df['cluster_group'] = kmeans.labels_
    vaccines_type_df['Date'] = vaccines_date_df['date'].values
    figa = px.scatter(vaccines_type_df, x='Date', y='Pfizer', color='cluster_group')
    figb = px.scatter(vaccines_type_df, x='Date', y='Sinovac', color='cluster_group')
    figc = px.scatter(vaccines_type_df, x='Date', y='AstraZeneca', color='cluster_group')
    figd = px.scatter(vaccines_type_df, x='Date', y='Cansino', color='cluster_group')
    fig3 = go.Figure(data=figa.data + figb.data + figc.data + figd.data)
    fig3.update_layout(xaxis_title="Date")
    fig3.show()
    st.plotly_chart(fig3, use_container_width=True)
      
    cumm_vacc = malaysia_vaccination_df[['date','cumul']]
    daily_case = malaysia_case_df[['date','cases_new']]
    daily_cumvacc_df = pd.merge(cumm_vacc, daily_case, on=['date','date'])
    daily_cumvacc_df = daily_cumvacc_df.drop(columns=['date'])
    daily_cumvacc_df = daily_cumvacc_df.rename(columns={'cumul': 'Cummulative Vaccinated People (Partial & Full)', 'cases_new': 'Daily Cases'})
    fig4 = px.scatter(daily_cumvacc_df,x="Cummulative Vaccinated People (Partial & Full)", y="Daily Cases")
    fig4.show()
    st.plotly_chart(fig4, use_container_width=True)

    af = AffinityPropagation(damping=0.7,preference=-50)
    clustering = af.fit(daily_cumvacc_df)
    cluster_centers_indices = af.cluster_centers_indices_
    n_clusters_ = len(cluster_centers_indices)
    daily_cumvacc_df['cluster_group'] = clustering.labels_
    fig5 = px.scatter(daily_cumvacc_df,x="Cummulative Vaccinated People (Partial & Full)", y="Daily Cases",color='cluster_group')
    fig5.show()
    st.plotly_chart(fig5, use_container_width=True)
    
    st.markdown("By applying affinity propagation algorithm on the cummulative vaccine doses and daily cases in Malaysia over the three months, the estimated number of clusters is 6. With the graph shown above is a concave shape of points plotted, we can induced that the six clusters are: \n1. Very low cummulative vaccine doses & very low daily cases (Purple)\n2. Low cummulative vaccine doses & low daily cases (Blue, light blue)\n3. Moderate cummulative vaccine doses & moderate daily cases (Aqua blue)\n4. High cummulative vaccine doses & very high daily cases (Green)\n5. High cummulative vaccine doses & moderate daily cases (Yellow, orange)\n6. Very high cummulative vaccine doses & low daily cases (Red)\nThe first cluster indicated a pandemic situation where the daily cases remained low and steady although the cummulative vaccine doses is not high. The second, third and fourth cluster indicated the climbing trend of both cummulative vaccine doses and daily cases, which means the vaccines effects have not shown yet and there might be outbreaks of COVID-19 happening at that time. The fifth cluster indicated the start of the daily cases decline trend where the vaccine can be said to be effective and have suppressed the pandemic a bit. The sixth cluster indicated the drastic drop of daily cases from its peak as the cummulative vaccine doses is very high.\nIn short, the vaccines are effective but require sometime to show its affects. Hence, we should continue the vaccinations.")