#######################
# Import libraries
pip install statsmodels==0.13.5 #for ARIMA
import streamlit as st

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn import datasets
import matplotlib.ticker as ticker
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#For the values
pip install pmdarima==2.0.3
from pmdarima import auto_arima

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, classification_report

import re
import random #testing

import warnings
warnings.filterwarnings('ignore')

#######################
# Page configuration
st.set_page_config(
    page_title="Google Playstore Top 100 Games(2021) data", # Replace this with your Project's Title
    page_icon="assets/icon.png", # You may replace this with a custom icon or emoji related to your project
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################

# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

# Sidebar
with st.sidebar:

    # Sidebar Title (Change this with your project's title)
    st.title('Google Playstore Top Games (2021)')

    # Page Button Navigation
    st.subheader("Pages")

    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'
    
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'

    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"

    if st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        st.session_state.page_selection = "data_cleaning"

    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"

    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "prediction"

    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"

    # Project Members
    st.subheader("Members")
    st.markdown("1. Joanna Hyacinth M. Reyes") #I did all the graphs and datasets so what?

#######################
# Data

# Load data
df = pd.read_csv("data/android-games.csv")

#######################

# Pages

# About Page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")

    st.write("This application explores and analyzes a dataset of Android games available on the Google Play Store. 
    The dataset contains information about various aspects of these games, including their category, installs, ratings, and growth over time.")

    # Your content for the ABOUT page goes here

# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")

    st.write("Google Playstore Dataset")
    st.write("This dataset is composed of top 100 games in Google Play Store.\n") 
    st.markdown('<a href="https://www.kaggle.com/datasets/dhruvildave/top-play-store-games" target="_blank">dataset link</a>', unsafe_allow_html=True)


    # Your content for your DATASET page goes here
    col = st.columns((3,3), gap='medium')

    with col[0]:
        df.dtypes
        st.write("The data shows a series of float, int, object and a bool non-null dtypes.")
    with col[1]:
        df.info()
    

# EDA Page
elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")


    col = st.columns((1.5, 4.5, 3), gap='medium')

    # Your content for the EDA page goes here

    with col[0]:
        st.markdown('#### Graphs Column 1')
        def pon():
          paid_counts = df['paid'].value_counts()

          plt.figure(figsize=(3, 3))
          plt.pie(paid_counts, labels=paid_counts.index, autopct='%1.1f%%', startangle=90)
          plt.title('Proportion of Paid and Free Games')
          plt.show()
        pon()
        
    with col[1]:
    
        st.markdown('#### Installs Distribution per Catgory')
        def ibc():
          plt.figure(figsize=(10, 5))
          sns.violinplot(x='category', y='installs', data=df)
          plt.xticks(rotation=90)
          plt.title('Distribution of Installs per Category')
          plt.xlabel('Category')
          plt.ylabel('Installs')
          plt.tight_layout()
          plt.show()
        ibc()     
        
        st.markdown('#### Average Rating per Catgory')
        plt.figure(figsize=(10, 6))
        sns.barplot(x='category', y='average rating', data=df)
        plt.xticks(rotation=90)
        plt.title('Average Rating per Category')
        plt.xlabel('Category')
        plt.ylabel('Average Rating')
        plt.tight_layout()
        plt.show()
   
    with col[2]:
        st.markdown('#### Games 30/60 Day Growth')

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='category', y='growth (30 days)', data=df)
        plt.xticks(rotation=90)
        plt.title('30-Day Growth per Category Distribution')
        plt.xlabel('Category')
        plt.ylabel('30-Day Growth')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='category', y='growth (60 days)', data=df)
        plt.xticks(rotation=90)
        plt.title('60-Day Growth per Category Distribution')
        plt.xlabel('Category')
        plt.ylabel('60-Day Growth')
        plt.tight_layout()
        plt.show()
  

# Data Cleaning Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")

    st.write("The data contains 1730 rows and 15 columns that doesn'T contain any null values.\n")


    st.write("The installs column is composed of 'milestones' meaning it shows how many times the game was downloaded. It does not show the accurate number of installs of a game, rather it 
    depicts a milestone of how many times the game has been downloaded, thus it will be converted to represent it numerically to improve the models.")

    #converting the object type in order to get the average numerical sense of it
    #installs is aparently a milestone, I had to replace
    label_encoder = LabelEncoder()

    install_ranges = OrdinalEncoder(categories=[['100.0 k', '500.0 k', '1.0 M', '5.0 M', '10.0 M', '50.0 M', '100.0 M', '500.0 M', '1000.0 M']],
                                 handle_unknown='use_encoded_value', unknown_value=-1)
    df['installsNumber'] = label_encoder.fit_transform(df['installs'])
    #######

    

    # Your content for the DATA CLEANING / PREPROCESSING page goes here
    

# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    # Your content for the MACHINE LEARNING page goes here



    

# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")

    # Your content for the PREDICTION page goes here

# Conclusions Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")

    # Your content for the CONCLUSION page goes here
