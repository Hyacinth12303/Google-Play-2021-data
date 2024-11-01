#######################
# Import libraries

import streamlit as st

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
    st.write("4. Machine Learning - Training three supervised classification models: ARIMA, Linear Regression, and Random Forest. This also includes model evaluation, feature importance, and tree plot")
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
        st.write("Missing Values")
        
    

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

    st.write("The data contains 1730 rows and 15 columns that doesn't contain any null values.\n")
    df.info()

    st.write("The installs column is composed of 'milestones' meaning it shows how many times the game was downloaded. It does not show the accurate number of installs of a game, rather it depicts a milestone of how many times the game has been downloaded, thus it will be converted to represent it numerically to improve the models.")
    st.write("This code will be used:")
    
    code0 = """
    label_encoder = LabelEncoder()
    install_ranges = OrdinalEncoder(categories=[['100.0 k', '500.0 k', '1.0 M', '5.0 M', '10.0 M', '50.0 M', '100.0 M', '500.0 M', '1000.0 M']], handle_unknown='use_encoded_value', unknown_value=-1)
    df['installsNumber'] = label_encoder.fit_transform(df['installs'])
    """

    st.code(code0, language='python')
    
#labelling code
    label_encoder = LabelEncoder()
    install_ranges = OrdinalEncoder(categories=[['100.0 k', '500.0 k', '1.0 M', '5.0 M', '10.0 M', '50.0 M', '100.0 M', '500.0 M', '1000.0 M']], handle_unknown='use_encoded_value', unknown_value=-1)
    df['installsNumber'] = label_encoder.fit_transform(df['installs'])
    
#I put training code here in this part.

    st.write("In this part, features and labels will be selected here for different types of models")

    #ARIMA model training

    st.markdown('**For the ARIMA model**')
    st.write("The ARIMA model will be used in order to predict the growth over 2 months using the rank and the 1 month growth, thus the 30/60 days growth will only be used to predict the rank of the game.")

    code1 = """
    Adt = df[['growth (30 days)', 'growth (60 days)']]
    y = Adt['growth (60 days)']
    exog = Adt[['growth (30 days)']]

    #split
    train_y = y[:-30]
    test_y = y[-30:]
    train_exog = exog[:-30]
    test_exog = exog[-30:]
    """
    
    #executable one
    Adt = df[['growth (30 days)', 'growth (60 days)']]
    y = Adt['growth (60 days)']
    exog = Adt[['growth (30 days)']]

    #split
    train_y = y[:-30]
    test_y = y[-30:]
    train_exog = exog[:-30]
    test_exog = exog[-30:]

    st.code(code1, language='python')
    
    print(f"Train Exog (X_train):\n\n{train_exog}\n\n"
      f"Test Exog (X_test):\n\n{test_exog}\n\n"
      f"Train y (y_train):\n\n{train_y}\n\n"
      f"Test y (y_test):\n\n{test_y}")

      
    st.markdown('**For the Linear Regression and Random Forest model**')

    st.write("There will be 2 models used to determine the rank of the game, using different sets of features.\n The linear regression model will use the growth and the number of installs to predict the rank of the game. This will measure the rank basing on the activeness of the game or how often the users engage with the game.\n The random forest on the other hand shall use the average rating, installs, and growth(30 days) to determine the rank of the game. This code shall be used to train and split the data.")

    code2 = """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    """
    #Displaying 2 separate things could interfere with how the output for the prediction will turn out...
    
    # Your content for the DATA CLEANING / PREPROCESSING page goes here

# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    # Your content for the MACHINE LEARNING page goes here

    st.markdown("**ARIMA model**")
    st.write("This employs the ARIMA (Autoregressive Integrated Moving Average) technique to forecast the 60-day growth of a game title based on its 30-day growth. ARIMA models are widely used for time series analysis and forecasting, leveraging past data patterns to predict future values. By utilizing historical growth data, this model aims to project the game's growth trajectory over the subsequent two months. This prediction can be valuable for understanding the long-term performance potential of a game and making informed decisions about marketing strategies and resource allocation.")

    st.write("In order to choose the best order for arima, an auto arima is used to determine the best order for the dataset.")
    code3 = """
    model = auto_arima(train_y, exogenous=train_exog,
      start_p=0, start_q=0,
      max_p=3, max_q=3, m=12, # Adjustable
      start_P=0, seasonal=True,
      d=None, D=1, trace=True,
      error_action='ignore',
      suppress_warnings=True,
      stepwise=True)
    """
    st.write("The result regarding the most optimal order is 0/0/0, however 0/1/0 is used for the model since it depicted a more interesting prediction")

#ARIMA model code
    Amodel = ARIMA(train_y, exog=train_exog, order=(0, 1, 0))  #basing from previous cell, all 0s is the best, but other values can be used.. 0,1,0 gave interesting results tho
    model_fit = Amodel.fit()
    Apredictions = model_fit.predict(start=len(train_y), end=len(y)-1, exog=test_exog)
    mse = mean_squared_error(test_y, Apredictions)
    
    print(f'Mean Squared Error: {mse}')

#Feature Importance
    st.markdown("**Linear Regression model**")

    st.write("This utilizes linear regression to predict the rank of a game title based on its growth in 30 and 60 days and the number of installs. This model could be valuable for developers and marketers to gauge the potential success of a game based on its early performance indicators.")
    
    X = df[['average rating', 'installsNumber', 'growth (30 days)', 'growth (60 days)', 'paid']]  # Include all relevant features
    y = df['rank']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    # Train the Linear Regression model
    LRM = LinearRegression()  # Create an instance of the model
    LRM.fit(X_train, y_train)  # Train the model on the training data

    importances = LRM.coef_  
    feature_names = X_train.columns  # Assuming X_train contains your feature names

    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances, y=feature_names)
    plt.title('Feature Importance in Linear Regression Model')
    plt.xlabel('Coefficient Value')  # Change x-axis label to 'Coefficient Value'
    plt.ylabel('Feature')
    plt.show()

    y_pred = LRM.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    
    st.write("In this graph, it shows that the average rating and the number of installs gives negative influence while whether the game is paid or not shows the highest value, since free games tend to go on top in stores. This means that the amount of installs and how high the rating affects the output.")

    st.markdown("**Random Forest model**")
    st.write("This will utilize the Random Forest algorithm, an ensemble learning method, to predict the rank of a game title based on its average rating, number of installs, and 30-day growth. Random Forest combines multiple decision trees to create a robust and accurate prediction model. By considering these key performance indicators, this model aims to estimate a game's ranking on the Google Play Store. This information can be valuable for understanding the factors that influence game rankings and for making data-driven decisions to improve a game's visibility and discoverability.")
    
#RANDOM FOREST model code

    X = df[['average rating', 'installsNumber', 'growth (30 days)', 'growth (60 days)', 'paid']]  # Include all relevant features
    y = df['rank']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  # Adjust hyperparameters as needed
    
    rf_model.fit(X_train, y_train)
    importances = rf_model.feature_importances_
    feature_names = X_train.columns  # Get feature names from X_train
    
    # Create a bar plot
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances, y=feature_names)
    plt.title('Feature Importance in Random Forest Model')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.show()

    st.write("Unlike the Linear Regression model, this graph shows that whether the game is free or not does not have much impact the model.　It shows that 60-day growth has the highest influence, followed by the number of installs and the 30-day growth. The average rating of the game also doesn't have much importance in determining the rank.")
    

    y_pred = rf_model.predict(X_test)
    
    st.write("Evaluation:")
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    y_pred_discrete = np.round(y_pred)
    accuracy = accuracy_score(y_test, y_pred_discrete)  # Ensure y_pred is discrete
    classification_rep = classification_report(y_test, y_pred_discrete)
    print(f"Accuracy: {accuracy * 100:.2f}%") #insert skull emoji




    

    

# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")

    # Your content for the PREDICTION page goes here
    

    # Your content for the EDA page goes here

    
    st.markdown("###ARIMA model random game prediction visualizer")

    def visualize_predictions():
        sample_indices = random.sample(range(len(df)), 15)
        sample_indices.sort()

        titles = []
        actual_ranks = []
        predicted_ranks = []

        for index in sample_indices:
            new_data = df[['growth (30 days)', 'growth (60 days)', 'installsNumber']].iloc[[index]]
            predicted_rank = LRM.predict(new_data)[0]

            titles.append(df['title'].iloc[index])
            actual_ranks.append(df['rank'].iloc[index])
            predicted_ranks.append(predicted_rank)

        plt.figure(figsize=(12, 6))
        plt.plot(titles, actual_ranks, marker='o', label='Actual Rank')
        plt.plot(titles, predicted_ranks, marker='x', label='Predicted Rank')
        plt.xlabel('Title')
        plt.ylabel('Rank')
        plt.title('Actual vs. Predicted Ranks for 15 Random Titles')
        plt.xticks(rotation=90)
        plt.legend()
        plt.tight_layout()
            
        #Displayer
        st.pyplot(plt)
        
        # Create the button
        if st.button("Randomize!"):
            visualize_predictions()


    st.markdown("###Linear Regression model random game prediction visualizer")
    
    
    
    
    #col = st.columns((1.5, 4.5, 3), gap='medium')
    #with col[0]:
    #with col[1]:
    
    
    










# Conclusions Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")

    # Your content for the CONCLUSION page goes here
