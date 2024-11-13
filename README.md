# Google Playstore Top Games (2021) data Visualizer using Streamlit

**[Disclaimer]: None of the assets/data belonged to me**

A streamlit application that analyzes this [dataset](https://www.kaggle.com/datasets/dhruvildave/top-play-store-games/data)

**Streamlit Link:** [Google Playstore Top Games (2021) Data Analysis](https://gplaystore-top100gamesdt2021.streamlit.app/)

**[Streamlit Pages]**

1. Dataset - Brief description of the Top 100 Google Playstore Games dataset used in this dashboard.

2. EDA - Exploratory Data Analysis of the games dataset. Highlighting the distribution of Iris species and the relationship between the features. It includes graphs such as Pie Chart, Violinplots, Barplots, Boxplots, and Scatterplots.

3. Data Cleaning / Pre-processing - Data cleaning and pre-processing steps such as encoding the installs column for training and testing sets.

4. Machine Learning - Training two supervised classification models: Random Forest Regression and Decision Tree. This also includes model details.

5. Prediction - Prediction page where rank and average rating is reassigned predicted and compared.

6. Conclusion - Summary of the insights and observations from the EDA and model training.




**I. Data Understanding** 

The dataset contains the details of top 100 games in Google Play Store in 2021, composed 
of 15  columns and 1730 rows.

**[Rank]:** The game's rank in the list of games.

**[Title]:** The name of the game.

**[Total Ratings]:** The total number of ratings the game has received.

**[Installs]:** The total number of downloads the game has reached.

**[Average Rating]:** The average rating score of the game. (1-5 stars)

**[Growth (30 days)]:** The growth in the total number of downloads and ratings over the last 30 days

**[Growth (60 days)]:** The growth in the total number of downloads and ratings over the last 60 days

**[Price]:** The price of the game.

**[Category]:** The genre or category to which the game belongs (e.g., Action, Adventure, Puzzle).

**[5 star ratings]:** The total number of 5-star ratings the game has received.

**[4 star ratings]:** The total number of 5-star ratings the game has received.

**[3 star ratings]:** The total number of 5-star ratings the game has received.

**[2 star ratings]:** The total number of 5-star ratings the game has received.

**[1 star ratings]:** The total number of 5-star ratings the game has received.

**[Paid]:** A boolean value indicating whether the game is a paid game (True) or free (False).



**II. Objectives**

This project aims to:

- Gain insights into the Android game market by exploring and visualizing data on game categories, installs, ratings, and growth.
  
- Understand the trend of what makes a game reach the top 100 in a category
  
- Clean and prepare the data for modeling.
  
- Determine the important features that affect the rank of a game.
  
- Build predictive models to estimate average ratings and reassign game ranks.
  



**III.Scope & Limitations**

The dataset contains the rank of the games from different categories in Google Play during 
2021 and its average rating, amount of ratings and downloads, price and if it’s paid or not. Its 
objective is to gain insights and understand the trends that make a game reach a top rank 
in the Play Store. It also aims to provide insights into the importance of different features in 
predicting the average rating, helping you understand which factors influence ratings most. 
The model is limited to predicting the average rating of a title and the rank of a title using features 
that have a value higher than 0.1 to improve the accuracy of the model.



**IV. Data Preparation** 

Duplicate titles with very similar data will be filtered out for the model training. 
A label encoder will also be used to enumerate the ‘installs’ column, listing 100.0k(0) as the 
lowest value and 1000.0M(8) as the highest.  



**V. Modeling**

The project will use Random Forest to predict the average rating of a title. It is an ensemble 
learning method that combines multiple decision trees to make predictions. 

Decision Tree is used for both classifying ranks for this project. It creates a tree-like model 
of decisions and their possible consequences, with branches representing decision points 
and leaves representing outcomes. 
It’s used to reorder the rank of the title using other features in the dataset. 



**VI. Evaluation**

The model achieved a high R-squared (R2) value in the random forest model. This indicates 
that the model explains a significant portion of the variance in the average rating.

Meanwhile, the R2 value for predicting ranks using the Decision Tree is lower compared to the Random Forest 
model. However, it properly predicted the correct ranks of most of the titles in each category.


**Members:**

1. Reyes, Joanna Hyacinth M.

   
