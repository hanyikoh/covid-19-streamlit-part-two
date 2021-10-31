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
st.set_page_config(layout='wide',page_title='Mooncake\'s Assignment', page_icon='🌕')
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
    'Select Question:',
('Question (i)', 'Question (ii)', 'Question (iii)', 'Question (iv)')
)

st.sidebar.markdown('__Questions__')

st.sidebar.markdown('i. Exploratory data analysis steps conducted.')
st.sidebar.markdown('ii. States that exhibit strong correlation with Pahang and Johor.')
st.sidebar.markdown('iii. Strong features/indicators to daily cases for Pahang, Kedah, Johor, and Selangor.')
st.sidebar.markdown('iv. Models (regression/classification) that performs well in predicting the daily cases for Pahang, Kedah, Johor, and Selangor.')

st.sidebar.markdown('__Datasets Used__')
st.sidebar.markdown('Categories: Cases and Testing, Healthcare, Mobility and Contact Tracing')

st.sidebar.markdown('__Open data on COVID-19 in Malaysia__')
st.sidebar.markdown('[Ministry of Health (MOH) Malaysia](https://github.com/MoH-Malaysia/covid19-public)')

st.sidebar.markdown("""<style>.css-1aumxhk {padding: 10em;}</style>""", unsafe_allow_html=True)
st.sidebar.markdown('''<small>TDS 3301 Data Mining | Group Assignment </small>''', unsafe_allow_html=True)

# Web App Title
mmu = Image.open('./mmu_logo.png')
st.image(mmu, width=300)
st.title("TDS 3301 Data Mining - Group Assignment")
st.write("Question 3: COVID-19 in Malaysia")
st.header(f"{question_num}")


if question_num == "Question (i)":
    question1.app()

elif question_num == "Question (ii)":
    question2.app()

elif question_num == "Question (iii)":
    question3.app()

elif question_num == "Question (iv)":
    question4.app()
