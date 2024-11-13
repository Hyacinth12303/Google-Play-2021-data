#######################
# Import libraries

import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import altair as alt

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

#from pmdarima import auto_arima
#I decide to cancel this and only show the snippet code of this(no implementation)

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

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
    st.title('Google Playstore Top Games (2021) Data Analysis')

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
    st.markdown("1. Joanna Hyacinth M. Reyes") #I did all the graphs, trained models, predicted stuffs, and datasets so what?

#######################
# Data

# Load data
df = pd.read_csv("data/android-games.csv")

#######################

# Pages

# About Page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")

    st.write("This application explores and analyzes a dataset of Android games available on the Google Play Store. The dataset contains information about various aspects of these games, including their category, installs, ratings, and growth over time.")

    st.markdown("")

    with st.expander('**Data Column Description**', expanded=True):
        st.write('''
                - :blue[Rank]: The game's rank in the list of games.
                - :blue[Title]: The name of the game.
                - :blue[Total Ratings]: The total number of ratings the game has received.
                - :blue[Installs]: The total number of downloads the game has reached.
                - :blue[Average Rating]: The average rating score of the game. (1-5 stars)
                - :blue[Growth (30 days)]: The growth in the total number of downloads and ratings over the last 30 days
                - :blue[Growth (60 days)]: The growth in the total number of downloads and ratings over the last 60 days
                - :blue[Price]: The price of the game.
                - :blue[Category]: The genre or category to which the game belongs (e.g., Action, Adventure, Puzzle).
                - :blue[5 star ratings]: The total number of 5-star ratings the game has received.
                - :blue[4 star ratings]: The total number of 5-star ratings the game has received.
                - :blue[3 star ratings]: The total number of 5-star ratings the game has received.
                - :blue[2 star ratings]: The total number of 5-star ratings the game has received.
                - :blue[1 star ratings]: The total number of 5-star ratings the game has received.
                - :blue[Paid]: A boolean value indicating whether the game is a paid game (True) or free (False).           
                ''')

    #Growth is a metric formed by including increase in total number of installs and total number of ratings and finding the average percentage growth. 
    #It is calculated in comparison the day the metric is updated last 30 days and last 60 days. So it may highly safe to assume that it is over the same period of time.


    # Your content for the ABOUT page goes here

    st.markdown("**Pages**")

    st.write("1. Dataset - Brief description of the Top 100 Google Playstore Games dataset used in this dashboard.")
    st.write("2. EDA - Exploratory Data Analysis of the games dataset. Highlighting the distribution of Iris species and the relationship between the features. It includes graphs such as Pie Chart, Violinplots, Barplots, Boxplots and Scatterplots.")
    st.write("3. Data Cleaning / Pre-processing - Data cleaning and pre-processing steps such as encoding the installs column for training and testing sets.")
    st.write("4. Machine Learning - Training two supervised classification models: Random Forest Regression and Decision Tree. This also includes model details.")
    st.write("5. Prediction - Prediction page where 15 random different games will be displayed and its predicted rank and growth in 60 days")
    st.write("6. Conclusion - Summary of the insights and observations from the EDA and model training.")

# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")

    st.write("**Google Playstore Dataset**")
    st.write("This dataset is composed of top 100 games in Google Play Store for each game category, scraped and provided by Dhruvil Dave in kaggle.\n") 
    st.markdown('<a href="https://www.kaggle.com/datasets/dhruvildave/top-play-store-games" target="_blank">🔗 dataset link</a>', unsafe_allow_html=True)
    st.dataframe(df)

    st.write("**Descriptive Statistics:**")
    st.dataframe(df.describe())
    
    # Your content for your DATASET page goes here
    col = st.columns((1,1,3), gap='medium')
    
    with col[0]:
        st.write("Data Types")
        st.write(df.dtypes)
    with col[1]:
        st.write("Missing Values")
        st.write(df.isnull().sum()) 
    with col[2]:
        st.write("The data contains 1730 rows and 15 columns that doesn't contain any null values.\n")
        
    

# EDA Page
elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")

    st.write("")
    col = st.columns((2, 4.5), gap='small')

    # Your content for the EDA page goes here

    with col[0]:
        st.markdown('**Proportion of Paid and Free Games**')
        def pon():
            paid_counts = df['paid'].value_counts()
            plt.figure(figsize=(3, 3))
            plt.pie(paid_counts, labels=paid_counts.index, autopct='%1.1f%%', startangle=90)
            st.pyplot(plt)
        pon()
        
        paid_games = df['paid'].sum()
        total_games = df['title'].count()
        st.write(f"- Out of {total_games} games, there are only {paid_games} paid games in the top 100.")
        
    with col[1]:
        
        def par():
            plt.figure(figsize=(10, 6))
            sns.barplot(x='category', y='average rating', data=df)
            plt.xticks(rotation=90)
            plt.title('Average Rating per Category')
            plt.xlabel('Category')
            plt.ylabel('Average Rating')
            plt.tight_layout()
            st.pyplot(plt)
        par()
    st.markdown("---")

    
    col = st.columns((4.5,2.5), gap='small')
    with col[0]:    
        def ibc():
                plt.figure(figsize=(10, 5))
                sns.violinplot(x='category', y='installs', data=df)
                plt.xticks(rotation=90)
                plt.title('Installs Distribution per Category')
                plt.xlabel('Category')
                plt.ylabel('Installs')
                st.pyplot(plt)
        ibc()   
        num_10M_titles = len(df[df['installs'] == '10.0 M'])
    with col[1]: 
        st.write('**Percentage of Installs in Android Games**')
        def insc():
            install_counts = df['installs'].value_counts() 
            plt.figure(figsize=(8, 8)) 
            
            wedges, texts, autotexts = plt.pie(install_counts, autopct='%1.1f%%', startangle=90, 
                                              textprops=dict(color="w")) 
            plt.legend(wedges, 
                       [f"{install_count} ({percentage:.1f}%)" 
                        for install_count, percentage in zip(install_counts.index, install_counts / install_counts.sum() * 100)],
                       title="Installs", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1)) 
            plt.show()
            plt.tight_layout()
            st.pyplot(plt)
        insc()
        st.write(f"Basing from this graph, the games with the most downloads have reached 10M installs. There are a total of {num_10M_titles} games that has reached over 10M downloads.")

    st.markdown("---")

    
    st.markdown("**30/60 day Growth per Category**")
    col = st.columns((3, 3), gap='medium')
    
    with col[0]:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='category', y='growth (30 days)', data=df)
        plt.xticks(rotation=90)
        plt.title('30-Day Growth per Category Distribution')
        plt.xlabel('Category')
        plt.ylabel('30-Day Growth')
        st.pyplot(plt)

    with col[1]:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='category', y='growth (60 days)', data=df)
        plt.xticks(rotation=90)
        plt.title('60-Day Growth per Category Distribution')
        plt.xlabel('Category')
        plt.ylabel('60-Day Growth')
        st.pyplot(plt)

  

# Data Cleaning Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")
    
    st.write("The installs column is composed of 'milestones' meaning it shows how many times the game was downloaded. It does not show the accurate number of installs of a game, rather it depicts a milestone of how many times the game has been downloaded, thus it will be converted to represent it numerically to improve the models.")
    st.write("This code will be used:")
    
    code0 = """
        #converting the object type in order to get the average numerical sense of it
        #installs is aparently a milestone, I had to replace
        install_ranges = ['100.0 k', '500.0 k', '1.0 M', '5.0 M', '10.0 M', '50.0 M', '100.0 M', '500.0 M', '1000.0 M']
        
        install_encoder = OrdinalEncoder(categories=[install_ranges], handle_unknown='use_encoded_value', unknown_value=-1)
        
        df['installsNumber'] = install_encoder.fit_transform(df[['installs']])
        
        #The reason why I didn't use 1m to 1000000 is because the installs column simply showcases a milestone,
        #meaning that a game wasn't simply installed around 1 million, it simply reached over 1 million downloads.
        #This is to make the model compact, simplified and not contain high values.
    """
    st.code(code0, language='python')
    
    st.write("To fully incorporate the category of the title of the model, this code will also be used")
    code000 = """
        #I use label encode here since it is a category, unlike installs, the higher the value is, the better
        category_order = [
            'GAME ACTION', 'GAME ADVENTURE', 'GAME ARCADE', 'GAME BOARD',
               'GAME CARD', 'GAME CASINO', 'GAME CASUAL', 'GAME EDUCATIONAL',
               'GAME MUSIC', 'GAME PUZZLE', 'GAME RACING', 'GAME ROLE PLAYING',
               'GAME SIMULATION', 'GAME SPORTS', 'GAME STRATEGY', 'GAME TRIVIA',
               'GAME WORD'
        ]
        category_encoder = LabelEncoder()
        
        df['categoryLabel'] = category_encoder.fit_transform(df['category'])
    """
    st.code(code000, language='python')
    
#Data Destroying
    st.subheader("Game Card and Game Word Category...")

    def bruh():
        plt.figure(figsize=(12, 6))
        sns.countplot(x='category', data=df, order=df['category'].value_counts().index)
        plt.xticks(rotation=90)
        plt.title('Number of Titles per Category')
        plt.xlabel('Category')
        plt.ylabel('Number of Titles')
        plt.tight_layout()
        st.pyplot(plt)
    bruh()

    st.write("In the graph above, it showcased the number of titles in each category, which brought light to a problem: there are a total of 26 games in gamecard category and 4 in gameword when the dataset should only be displaying 100 games in each category.")
    
    st.write("Here's some examples:")
    
    duplicates = df[df.duplicated(subset=['title', 'rank', 'total ratings', 'installs'], keep=False)]
    
    # Check if there are any duplicates
    if duplicates.empty:
        st.write("No duplicate ranks found.")
    else:
        st.dataframe(duplicates)

    st.write("In order to eliminate this, the code below will be used to eliminate the data with similar name, rank, total ratings and number of installs")
    code4 = """
    df = df.drop_duplicates(subset=['title', 'rank', 'total ratings', 'installs'], keep='first')
    """
    st.code(code4, language='python') 

    st.write("Here's the graph after removing the duplicates:")
    df = df.drop_duplicates(subset=['title', 'rank', 'total ratings', 'installs'], keep='first')
    def bruh():
        plt.figure(figsize=(12, 6))
        sns.countplot(x='category', data=df, order=df['category'].value_counts().index)
        plt.xticks(rotation=90)
        plt.title('Number of Titles per Category no Dupes')
        plt.xlabel('Category')
        plt.ylabel('Number of Titles')
        plt.tight_layout()
        st.pyplot(plt)
    bruh()
    
    
    st.markdown("---")
#I put training code here in this part.

    #RF Reg

    st.header("Train-Test Split")
    st.subheader("Random Forest Regressor")
    st.write("Random Forest Regressor will be used to predict the average rating of a title. It is an ensemble learning method that combines multiple decision trees to make predictions. The data will be split and trained to predict the average rating of a title. The features used is 5 stars and 1 star ratings in order to have a most accurate prediction of average rating.")

    code1 = """
        # Define features and target
        X = df[['5 star ratings', '1 star ratings']] #These 2 shows utmost importance, exceeding 0.1
        y = df['average rating']  # Target variable
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    """
    st.code(code1, language='python')

    category_order = [
        'GAME ACTION', 'GAME ADVENTURE', 'GAME ARCADE', 'GAME BOARD',
           'GAME CARD', 'GAME CASINO', 'GAME CASUAL', 'GAME EDUCATIONAL',
           'GAME MUSIC', 'GAME PUZZLE', 'GAME RACING', 'GAME ROLE PLAYING',
           'GAME SIMULATION', 'GAME SPORTS', 'GAME STRATEGY', 'GAME TRIVIA',
           'GAME WORD'
    ]
    category_encoder = LabelEncoder()
    df['categoryLabel'] = category_encoder.fit_transform(df['category'])
    
    install_ranges = ['100.0 k', '500.0 k', '1.0 M', '5.0 M', '10.0 M', '50.0 M', '100.0 M', '500.0 M', '1000.0 M']
    install_encoder = OrdinalEncoder(categories=[install_ranges], handle_unknown='use_encoded_value', unknown_value=-1)
    df['installsNumber'] = install_encoder.fit_transform(df[['installs']])
    
    duplicate_ranks = df[df.duplicated(subset=['category', 'rank', 'installs', 'total ratings'], keep=False)]

    if st.checkbox("Show Feature Importance Graph for RF Regressor no filter"):
        # Define features and target variable
        X = df[['rank', 'total ratings', 'installsNumber', 'growth (30 days)', 'growth (60 days)', 'price', 'categoryLabel', '5 star ratings', '4 star ratings', '3 star ratings', '2 star ratings', '1 star ratings', 'paid']]
        y = df['average rating']
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize and train the Random Forest Regressor
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        
        # Evaluate feature importance
        feature_importances = rf_model.feature_importances_
        features = X.columns
        
        # Plot feature importances
        fig, ax = plt.subplots(figsize=(10, 6))
        indices = feature_importances.argsort()[::-1]  # Sort features by importance
        ax.barh(range(len(indices)), feature_importances[indices], color='skyblue')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([features[i] for i in indices])
        ax.set_xlabel("Feature Importance")
        ax.set_ylabel("Feature")
        ax.set_title("Feature Importance for Predicting Average Rating")
        
        # Show plot in Streamlit
        st.pyplot(fig)
    
    st.subheader('Decision Tree')
    st.write("Decision Tree is used for both classifying ranks for this project. It creates a tree-like model of decisions and their possible consequences, with branches representing decision points and leaves representing outcomes. It’s used to reorder the rank of the title using other features in the dataset.")

    code2 = """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    """
    st.code(code2, language='python')

    category_order = [
        'GAME ACTION', 'GAME ADVENTURE', 'GAME ARCADE', 'GAME BOARD',
           'GAME CARD', 'GAME CASINO', 'GAME CASUAL', 'GAME EDUCATIONAL',
           'GAME MUSIC', 'GAME PUZZLE', 'GAME RACING', 'GAME ROLE PLAYING',
           'GAME SIMULATION', 'GAME SPORTS', 'GAME STRATEGY', 'GAME TRIVIA',
           'GAME WORD'
    ]
    category_encoder = LabelEncoder()

    df['categoryLabel'] = category_encoder.fit_transform(df['category'])
    install_ranges = ['100.0 k', '500.0 k', '1.0 M', '5.0 M', '10.0 M', '50.0 M', '100.0 M', '500.0 M', '1000.0 M']
    
    install_encoder = OrdinalEncoder(categories=[install_ranges], handle_unknown='use_encoded_value', unknown_value=-1)
    
    df['installsNumber'] = install_encoder.fit_transform(df[['installs']])
    duplicate_ranks = df[df.duplicated(subset=['category', 'rank', 'installs', 'total ratings'], keep=False)]
    
    if st.checkbox("Show Feature Importance Graph for DT Regressor no filter"):
        X = df[['total ratings', 'installsNumber', 'average rating', 'growth (30 days)', 'growth (60 days)', 'price', 'categoryLabel', '5 star ratings', '4 star ratings', '3 star ratings', '2 star ratings', '1 star ratings', 'paid']]
        y = df['rank']
        
        # 2. Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 3. Train the DecisionTreeRegressor
        tree_model = DecisionTreeRegressor(random_state=42)
        tree_model.fit(X_train, y_train)
        
        # 4. Feature Importance Plot
        feature_importances = tree_model.feature_importances_
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(X.columns, feature_importances, color="skyblue")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.set_title("Feature Importance in Decision Tree Model")
        plt.tight_layout()
        
        # Show plot in Streamlit
        st.pyplot(fig)

    # Your content for the DATA CLEANING / PREPROCESSING page goes here



# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    # Your content for the MACHINE LEARNING page goes here

    st.markdown("**Random Forest Regression**")
    st.write("A Random Forest Regressor is initialized with n_estimators=100 (meaning it uses 100 decision trees) and random_state=42 for reproducibility. The model is trained using the scaled training data (X_train_scaled, y_train).")
    code3 = """
    X = df[['5 star ratings', '1 star ratings']] #These 2 shows utmost importance, exceeding 0.1
    y = df['average rating']  # Target variable
    
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train the Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Evaluate feature importance
    feature_importances = rf_model.feature_importances_
    features = X.columns
    
    # Plot feature importances
    plt.figure(figsize=(10, 8))
    indices = feature_importances.argsort()[::-1]  # Sort features by importance
    plt.barh(range(len(indices)), feature_importances[indices], color='skyblue')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance for Predicting Average Rating")
    plt.show()
    """
    
    st.code(code3, language='python')

    X = df[['5 star ratings', '1 star ratings']]
    y = df['average rating']
        
        # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
        
        # Initialize and train the Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    if st.checkbox("Show Feature Importance Graph (final)"):
        # Define features and target variable        
        # Evaluate feature importance
        feature_importances = rf_model.feature_importances_
        features = X.columns
        
        # Plot feature importances
        fig, ax = plt.subplots(figsize=(10, 6))
        indices = feature_importances.argsort()[::-1]  # Sort features by importance
        ax.barh(range(len(indices)), feature_importances[indices], color='skyblue')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([features[i] for i in indices])
        ax.set_xlabel("Feature Importance")
        ax.set_ylabel("Feature")
        ax.set_title("Feature Importance for Predicting Average Rating")
        
        # Show plot in Streamlit
        st.pyplot(fig)
    y_pred = rf_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.subheader("Model Performance:")
    st.write("Mean Squared Error (MSE):", mse)
    st.write("R-squared (R²):", r2)

    st.markdown("---")
        
#DT
    st.markdown("**Decision Tree Regression**")

    st.write("This project will utilize Decision Tree Regressor to predict the rank of an Android game based on its total ratings, 5 star ratings, and category(label encoded). This model could be valuable for developers and marketers to gauge the potential success of a game based on its early performance indicators.")
    
    code333 = """
    X = df[['total ratings', '5 star ratings', 'categoryLabel']]
    y = df['rank']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    tree_model = DecisionTreeRegressor(random_state=42)
    tree_model.fit(X_train, y_train)

    feature_importances = tree_model.feature_importances_
    
    plt.figure(figsize=(10, 6))
    plt.barh(X.columns, feature_importances)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance in Decision Tree Model")
    plt.tight_layout()
    plt.show()
    """
    st.code(code333, language='python')
    
    category_order = [
        'GAME ACTION', 'GAME ADVENTURE', 'GAME ARCADE', 'GAME BOARD',
           'GAME CARD', 'GAME CASINO', 'GAME CASUAL', 'GAME EDUCATIONAL',
           'GAME MUSIC', 'GAME PUZZLE', 'GAME RACING', 'GAME ROLE PLAYING',
           'GAME SIMULATION', 'GAME SPORTS', 'GAME STRATEGY', 'GAME TRIVIA',
           'GAME WORD'
    ]
    category_encoder = LabelEncoder()
    df['categoryLabel'] = category_encoder.fit_transform(df['category'])
    
    install_ranges = ['100.0 k', '500.0 k', '1.0 M', '5.0 M', '10.0 M', '50.0 M', '100.0 M', '500.0 M', '1000.0 M']
    install_encoder = OrdinalEncoder(categories=[install_ranges], handle_unknown='use_encoded_value', unknown_value=-1)
    df['installsNumber'] = install_encoder.fit_transform(df[['installs']])
    
    duplicate_ranks = df[df.duplicated(subset=['category', 'rank', 'installs', 'total ratings'], keep=False)]
    
    X = df[['total ratings', '5 star ratings', 'categoryLabel']]
    y = df['rank']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    tree_model = DecisionTreeRegressor(random_state=42)
    tree_model.fit(X_train, y_train)   
    
    if st.checkbox("Show Feature Importance Graph"):

        feature_importances = tree_model.feature_importances_
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(X.columns, feature_importances, color="skyblue")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.set_title("Feature Importance in Decision Tree Model")
        plt.tight_layout()
        
        # Show plot in Streamlit
        st.pyplot(fig)
        
    df['predicted_rank'] = tree_model.predict(X)
    df_sorted = df.sort_values(by='predicted_rank', ascending=True)  
    df['predicted_rank_ordinal'] = df['predicted_rank'].round().astype(int)
    
    y_pred = tree_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    st.subheader("Model Performance:")
    st.write("Mean Squared Error (MSE):", mse)
    st.write("R-squared (R²):", r2)



# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")

    # Your content for the PREDICTION page goes here
#RF
    X = df[['5 star ratings', '1 star ratings']] #These 2 shows utmost importance, exceeding 0.1
    y = df['average rating']  # Target variable
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    y_pred = rf_model.predict(X_test_scaled)
    

    st.title("Random Game Rating Prediction")
    st.write("This will predict the average rating of a random game based on its 5-star and 1-star ratings.")
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    # Calculate R-squared (R2)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Performance:")
    st.write("Mean Squared Error (MSE):", mse)
    st.write("R-squared (R²):", r2)

    
    
    if st.button("Random Game"):
        col = st.columns((1, 1, 1), gap='medium')
        with col[0]:
            # Randomly select a game for prediction
            random_game = df.sample(1)
        
            # Get the game's features and scale them using the same scaler
            game_features = random_game[['5 star ratings', '1 star ratings']]
            game_features_scaled = scaler.transform(game_features)
        
            # Predict the average rating
            predicted_rating = rf_model.predict(game_features_scaled)[0]
        
            # Display the results
            st.subheader("Game 1")
            st.write(f"**Game Title**: {random_game['title'].values[0]}")
            st.write(f"**Category**: {random_game['category'].values[0]}")
            st.write(f"**Installs**: {random_game['installs'].values[0]}")
            st.write(f"**Rank**: {random_game['rank'].values[0]}")
            st.write(f"**Actual Average Rating**: {random_game['average rating'].values[0]}")
            st.write(f"**Predicted Average Rating**: {predicted_rating:.2f}")
        with col[1]:
            # Randomly select a game for prediction
            random_game = df.sample(1)
        
            # Get the game's features and scale them using the same scaler
            game_features = random_game[['5 star ratings', '1 star ratings']]
            game_features_scaled = scaler.transform(game_features)
        
            # Predict the average rating
            predicted_rating = rf_model.predict(game_features_scaled)[0]
        
            # Display the results
            st.subheader("Game 2")
            st.write(f"**Game Title**: {random_game['title'].values[0]}")
            st.write(f"**Category**: {random_game['category'].values[0]}")
            st.write(f"**Installs**: {random_game['installs'].values[0]}")
            st.write(f"**Rank**: {random_game['rank'].values[0]}")
            st.write(f"**Actual Average Rating**: {random_game['average rating'].values[0]}")
            st.write(f"**Predicted Average Rating**: {predicted_rating:.2f}")
        with col[2]:
            # Randomly select a game for prediction
            random_game = df.sample(1)
        
            # Get the game's features and scale them using the same scaler
            game_features = random_game[['5 star ratings', '1 star ratings']]
            game_features_scaled = scaler.transform(game_features)
        
            # Predict the average rating
            predicted_rating = rf_model.predict(game_features_scaled)[0]
        
            # Display the results
            st.subheader("Game 3")
            st.write(f"**Game Title**: {random_game['title'].values[0]}")
            st.write(f"**Category**: {random_game['category'].values[0]}")
            st.write(f"**Installs**: {random_game['installs'].values[0]}")
            st.write(f"**Rank**: {random_game['rank'].values[0]}")
            st.write(f"**Actual Average Rating**: {random_game['average rating'].values[0]}")
            st.write(f"**Predicted Average Rating**: {predicted_rating:.2f}")

    #col = st.columns((1, 1, 1), gap='medium')
    #with col[0]:
    #with col[1]:
    st.markdown("---")

    category_order = [
        'GAME ACTION', 'GAME ADVENTURE', 'GAME ARCADE', 'GAME BOARD',
           'GAME CARD', 'GAME CASINO', 'GAME CASUAL', 'GAME EDUCATIONAL',
           'GAME MUSIC', 'GAME PUZZLE', 'GAME RACING', 'GAME ROLE PLAYING',
           'GAME SIMULATION', 'GAME SPORTS', 'GAME STRATEGY', 'GAME TRIVIA',
           'GAME WORD'
    ]
    category_encoder = LabelEncoder()
    df['categoryLabel'] = category_encoder.fit_transform(df['category'])
    
    install_ranges = ['100.0 k', '500.0 k', '1.0 M', '5.0 M', '10.0 M', '50.0 M', '100.0 M', '500.0 M', '1000.0 M']
    install_encoder = OrdinalEncoder(categories=[install_ranges], handle_unknown='use_encoded_value', unknown_value=-1)
    df['installsNumber'] = install_encoder.fit_transform(df[['installs']])
    
    duplicate_ranks = df[df.duplicated(subset=['category', 'rank', 'installs', 'total ratings'], keep=False)]

    X = df[['total ratings', '5 star ratings', 'categoryLabel']]
    y = df['rank']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    tree_model = DecisionTreeRegressor(random_state=42)
    tree_model.fit(X_train, y_train)
   
    
    st.title("Game Rank Reassigned")
    st.write("This shows the reassigned rank of the titles per category using Decision Tree Regressor.")

    # Make predictions and add them to the DataFrame
    df['predicted_rank'] = tree_model.predict(X)
    
    # Loop through each category and plot original vs predicted ranks
    for category in df['category'].unique():
        # Filter the DataFrame for the specific category
        category_df = df[df['category'] == category]
        
        # Sort the category dataframe by predicted rank in increasing order
        category_df = category_df.sort_values(by='predicted_rank')
    
        # Extract titles, original ranks, and predicted ranks
        titles = category_df['title'].values
        original_ranks = category_df['rank'].values  # Original ranks (unsorted)
        predicted_ranks = category_df['predicted_rank'].values  # Predicted ranks (sorted)
    
        indices = np.arange(len(titles))  # Indices for x-axis
    
        # Create a bar chart
        plt.figure(figsize=(12, 6))
        plt.bar(indices, original_ranks, color='skyblue', alpha=0.7, label="Original Rank")
        plt.bar(indices, predicted_ranks, color='orange', alpha=0.5, label="Predicted Rank")
    
        # Customize the plot
        plt.xticks(indices, titles, rotation=90)  # Rotate titles on x-axis for readability
        plt.xlabel("Title")
        plt.ylabel("Rank")
        plt.title(f"Original vs. Predicted Ranks for Category: {category}")
        plt.legend()
        plt.tight_layout()
        plt.grid(True, linestyle="--", alpha=0.6)
        
        # Show plot in Streamlit
        st.pyplot(plt)
        plt.clf()  # Clear the figure to avoid overlapping plots
    
        


# Conclusions Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")
    st.write("This project explores the Android game market dataset using two machine learning models: Random Forest Regression and Decision Tree Regression. The model achieved a high R-squared (R2) value in the random forest model. This indicates that the model explains a significant portion of the variance in the average rating. Meanwhile, the R2 value for predicting  ranks is lower compared to the Random Forest model. However, it properly predicted the correct ranks of most of the titles in each category.") 
    st.write("It is concluded that the most important features when predicting the average rating of the game is its total number of 5 stars and 1 star per title. It is also concluded that the most important factor when assigning a rank to a title are the number of ratings it contains and how many of the ratings contain 5 stars per category. It had nothing to do with how many people installed the game and the growth of the game in under a month or 2. This, though raises a concern, which is whether these amount of ratings are rated by a person or not. Due to the amount of ratings and the number of 5 stars, a game can easily be boosted up to the top 100 games in Google Play 2021 if the game reviews are botted. This can affect the kind of games seen in the top ranks which hinder a high-quality game that deserves to be on that rank.")
    

    # Your content for the CONCLUSION page goes here
