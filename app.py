import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# from pandas_profiling import ProfileReport
# from streamlit_pandas_profiling import st_profile_report
from multipage import MultiPage
from pages import question1,question2,question3,question4

# Create an instance of the app 
app = MultiPage()
st.set_page_config(layout='wide',page_title='Mooncake\'s Project', page_icon='🌕')
st.set_option('deprecation.showPyplotGlobalUse', False)

# Import Cases and Testing Dataset
malaysia_case_dir = "dataset/cases_malaysia.csv"
state_case_dir = "dataset/cases_state.csv"
clusters_dir = "dataset/clusters.csv"
malaysia_tests_dir = "dataset/tests_malaysia.csv"
states_tests_dir = "dataset/tests_state.csv"
pkrc_dir = "dataset/pkrc.csv"
checkIn_dir = "dataset/checkin_state.csv"

# Sidebar
mooncake = Image.open('./mooncake_logo.png')
st.sidebar.image(mooncake, width=64)
st.sidebar.header('TT3V - Mooncake')
st.sidebar.markdown('''
    Koh Han Yi (1181302907)  
    Lee Min Xuan (1181302793)  
    Tan Jia Qi (1191301879)
    '''
)

question_num = st.sidebar.selectbox(
    'Select Sections:',
('Informative Insights from the Datasets','Clustering','Classification','Regression')
)

st.sidebar.markdown('__Datasets Used__')
st.sidebar.markdown('MOH, CITF, R-Naught, and Google Trends Datasets')

st.sidebar.markdown('__Open data on COVID-19 in Malaysia__')
st.sidebar.markdown('[Ministry of Health (MOH) Malaysia](https://github.com/MoH-Malaysia/covid19-public)')
st.sidebar.markdown('__Open data on Malaysia\'s National Covid-​19 Immunisation Programme__')
st.sidebar.markdown('[COVID-19 Immunisation Task Force (CITF) Malaysia](https://github.com/CITF-Malaysia/citf-public)')
st.sidebar.markdown('__R-Naught Value in Malaysia__')
st.sidebar.markdown('[COVID-19 Malaysia](https://covid-19.moh.gov.my/kajian-dan-penyelidikan/nilai-r-malaysia)')
st.sidebar.markdown('__Google Trends in Malaysia__')
st.sidebar.markdown('[Google Trends](https://trends.google.com/trends/?geo=MY)')

st.sidebar.markdown("""<style>.css-1aumxhk {padding: 10em;}</style>""", unsafe_allow_html=True)
st.sidebar.markdown('''<small>TDS 3301 Data Mining | Group Project </small>''', unsafe_allow_html=True)

# Web App Title
mmu = Image.open('./mmu_logo.png')
st.image(mmu, width=300)
st.title("TDS 3301 Data Mining - Group Project")
st.write("QUESTION 2: Malaysia COVID-19 Cases and Vaccination")
st.header(f"{question_num}")


if question_num == "Informative Insights from the Datasets":
    question1.app()

elif question_num == "Clustering":
    question2.app()

elif question_num == "Classification":
    question3.app()

elif question_num == "Regression":
    question4.app()
