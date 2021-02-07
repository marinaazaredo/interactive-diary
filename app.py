# 1. Library imports
import uvicorn
from fastapi import FastAPI
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from Model import ProfileAnswers
from Questions import get_user_category
from random import sample 
import random
from tabulate import tabulate


# Define KMeans and Normalizer model, then combine them into pipeline
kmeans = KMeans(n_clusters=5)
normalizer = Normalizer()
from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(normalizer, kmeans)

# Set a seed, so the random factor of sklearn won't change
np.random.seed(123)

# 2. Create app and model objects
app = FastAPI()

file = 'QnC_V3.xlsx'

questions_df = pd.read_excel(file, sheet_name='Test_data', nrows=85, usecols='A:B,H:N', index_col=0)
# Create an array of samples (ratings of the questions)
questions_array = questions_df[['Physical_Health', 'Feelings', 'Relationship', 'Self_Reflection',
                                'Self_Motivation', 'Personal_Growth', 'Open_topic']].to_numpy()

# Fit the questions to the pipeline
pipeline.fit(questions_array)
# Predict the questions' labels/targets
labels = pipeline.predict(questions_array)

# Organize the labels of questions in DataFrame
df = pd.DataFrame({'labels': labels, 'questions': questions_df['Questions']})
df_sorted = df.sort_values(by=['labels'])


fake_users_df = pd.DataFrame(np.random.randint(1, 5, size=(150,7)), columns=['Physical_Health', 'Feelings',
                'Relationship', 'Self_Reflection', 'Self_Motivation', 'Personal_Growth', 'Open_topic'])
fake_users_array = fake_users_df.to_numpy()

# Fake user profiles that needs to be provided by the API
user001 = np.array([[5, 3, 1, 2, 2, 5, 4]])
user002 = np.array([[1, 3, 4, 2, 1, 4, 5]])
user003 = np.array([[3, 2, 4, 1, 5, 5, 2]])

# Predict the labels of the users
fake_users_label = pipeline.predict(fake_users_array)
user001_label = pipeline.predict(user001)
user002_label = pipeline.predict(user002)
user003_label = pipeline.predict(user003)

fake_users_label_df = pd.DataFrame({'LABELS': fake_users_label})

perguntas = pd.read_csv('question_labels.csv')




class UserLabel(int):
    Physical_Health: int 
    Feelings: int 
    Relationship: int 
    Self_Reflection: int 
    Self_Motivation: int 
    Personal_Growth: int



# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted flower species with the confidence
@app.post('/predict_category')
def predict_user (user_profile_answers: ProfileAnswers):
    category=get_user_category(user_profile_answers)

    questions_matching_category = df.query(f"labels == {category}")
    print(tabulate(questions_matching_category, headers='keys', tablefmt='psql'))

    sample_df = questions_matching_category.sample()
    print("Subset containing one random row", sample_df)
    
    question = sample_df['questions'].values[0]

    return question


# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
