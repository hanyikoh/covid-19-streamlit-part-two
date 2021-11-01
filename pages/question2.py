import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans
from kneed import KneeLocator, DataGenerator as dg
from sklearn.cluster import AffinityPropagation


state_case_dir = "dataset/cases_state.csv"
start_date = "2021-07-01"
end_date = "2021-09-30"
malaysia_vaccination_dir = "dataset/vax_malaysia.csv"
malaysia_vaccination_df = pd.read_csv(malaysia_vaccination_dir)
after_start_date = malaysia_vaccination_df["date"] >= start_date
before_end_date = malaysia_vaccination_df["date"] <= end_date
between_two_dates = after_start_date & before_end_date
malaysia_vaccination_df = malaysia_vaccination_df.loc[between_two_dates]
vaccines_type_df = malaysia_vaccination_df[['date','pfizer1','pfizer2','sinovac1','sinovac2','astra1','astra2','cansino']]
vaccines_type_df['Pfizer'] = vaccines_type_df.loc[:, ('pfizer1', 'pfizer2')].sum(axis=1)
vaccines_type_df['Sinovac'] = vaccines_type_df.loc[:, ('sinovac1','sinovac2')].sum(axis=1)
vaccines_type_df['AstraZeneca'] = vaccines_type_df.loc[:, ('astra1','astra2')].sum(axis=1)
vaccines_type_df['Cansino'] = vaccines_type_df.loc[:, ('cansino')]
vaccines_type_df = vaccines_type_df[['Pfizer','Sinovac','AstraZeneca','Cansino']]
vaccines_type_df.reset_index(drop=True, inplace=True)
vaccines_type_df =(vaccines_type_df-vaccines_type_df.min())/(vaccines_type_df.max()-vaccines_type_df.min())
malaysia_case_dir = "dataset/cases_malaysia.csv"

def app():
    sns.set(rc={'figure.figsize':(15,10)})
    cols = ['Pfizer', 'Sinovac', 'AstraZeneca', 'Cansino']
    fig, ax = plt.subplots(1,1)
    for col in cols:
        sns.scatterplot(vaccines_type_df.index, vaccines_type_df[col], ax=ax)
    st.pyplot()


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

    plt.figure(figsize=(16,8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    st.pyplot()

 
    kmeans = KMeans(n_clusters=2, init='random')
    label = kmeans.fit_predict(vaccines_type_df)
    u_labels = np.unique(label)
    plt.figure(figsize=(15,10))
    for i in u_labels:
        plt.scatter(vaccines_type_df.iloc[label == i , 0] , vaccines_type_df.iloc[label == i , 1] , label = i)
    plt.title('Daily Total Vaccine Doses in Malaysia')
    plt.legend()
    st.pyplot()
    
    malaysia_case_df = pd.read_csv(malaysia_case_dir)
    after_start_date = malaysia_case_df["date"] >= start_date
    before_end_date = malaysia_case_df["date"] <= end_date
    between_two_dates = after_start_date & before_end_date
    malaysia_case_df = malaysia_case_df.loc[between_two_dates]
    cumm_vacc = malaysia_vaccination_df[['date','cumul']]
    daily_case = malaysia_case_df[['date','cases_new']]
    daily_cumvacc_df = pd.merge(cumm_vacc, daily_case, on=['date','date'])
    daily_cumvacc_df = daily_cumvacc_df.drop(columns=['date'])
    daily_cumvacc_df = daily_cumvacc_df.rename(columns={'cumul': 'Cummulative Vaccinated People (Partial & Full)', 'cases_new': 'Daily Cases'})
    sns.set(rc={'figure.figsize':(15,10)})
    sns.scatterplot(data=daily_cumvacc_df, x="Cummulative Vaccinated People (Partial & Full)", y="Daily Cases")
    st.pyplot()


    af = AffinityPropagation(damping=0.7,preference=-50)
    clustering = af.fit(daily_cumvacc_df)
    cluster_centers_indices = af.cluster_centers_indices_
    n_clusters_ = len(cluster_centers_indices)
    plt.scatter(daily_cumvacc_df.iloc[:,0], daily_cumvacc_df.iloc[:,1], c=clustering.labels_, cmap='rainbow', alpha=0.7, edgecolors='b')
    plt.title(f'Estimated number of clusters = {n_clusters_}')
    plt.xlabel('Cummulative Vaccine Doses (Partial & Full)')
    plt.ylabel('Daily Cases')
    st.pyplot()
    
    st.markdown("By applying affinity propagation algorithm on the cummulative vaccine doses and daily cases in Malaysia over the three months, the estimated number of clusters is 6. With the graph shown above is a concave shape of points plotted, we can induced that the six clusters are: \n1. Very low cummulative vaccine doses & very low daily cases (Purple)\n2. Low cummulative vaccine doses & low daily cases (Blue, light blue)\n3. Moderate cummulative vaccine doses & moderate daily cases (Aqua blue)\n4. High cummulative vaccine doses & very high daily cases (Green)\n5. High cummulative vaccine doses & moderate daily cases (Yellow, orange)\n6. Very high cummulative vaccine doses & low daily cases (Red)\nThe first cluster indicated a pandemic situation where the daily cases remained low and steady although the cummulative vaccine doses is not high. The second, third and fourth cluster indicated the climbing trend of both cummulative vaccine doses and daily cases, which means the vaccines effects have not shown yet and there might be outbreaks of COVID-19 happening at that time. The fifth cluster indicated the start of the daily cases decline trend where the vaccine can be said to be effective and have suppressed the pandemic a bit. The sixth cluster indicated the drastic drop of daily cases from its peak as the cummulative vaccine doses is very high.\nIn short, the vaccines are effective but require sometime to show its affects. Hence, we should continue the vaccinations.")