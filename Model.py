# 1. Library imports
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from pydantic import BaseModel
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
import numpy as np


# Define KMeans and Normalizer model, then combine them into pipeline
kmeans = KMeans(n_clusters=5)
normalizer = Normalizer()
from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(normalizer, kmeans)
​
# Set a seed, so the random factor of sklearn won't change
np.random.seed(123)
​
# Assign spreadsheet to file
file = '/Users/marinaazaredo/QnC_V3.xlsx'
​
class UserType (BaseModel):
    Physical Health: int
    Feelings: int 
    Relationship: int
    Self_Reflection: int
    Self_Motivation: int 
    Personal Growth: int
    Open topic: int

class UserModel:

    # PIPELINE TO QUESTIONS' RATINGS DATA
    # Deploy questions' ratings DataFrame from spreadsheet
    questions_df = pd.read_excel(file, sheet_name='Test_data', nrows=85, usecols='A:B,H:N', index_col=0)
    # Create an array of samples (ratings of the questions)
    questions_array = questions_df[['Physical Health', 'Feelings', 'Relationship', 'Self_Reflection',
                                'Self_Motivation', 'Personal Growth', 'Open topic']].to_numpy()
    # Fit the questions to the pipeline
    pipeline.fit(questions_array)
    # Predict the questions' labels/targets
    labels = pipeline.predict(questions_array)
​
​
    # Organize the labels of questions in DataFrame
    df = pd.DataFrame({'labels': labels, 'questions': questions_df['Questions']})
    df_sorted = df.sort_values(by=['labels'])

def predict_user(Physical Health, Feelings, Relationship, Self_Reflection, Self_Motivation, Personal Growth, Open topic):
        data_in = [[Physical Health, Feelings, Relationship, Self_Reflection, Self_Motivation, Personal Growth, Open topic]]
        prediction = labels(data_in)
        return prediction










