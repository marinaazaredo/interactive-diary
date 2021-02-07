from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
import pandas as pd
import numpy as np


# Define KMeans and Normalizer model, then combine them into pipeline
kmeans = KMeans(n_clusters=5)
normalizer = Normalizer()
from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(normalizer, kmeans)

# Set a seed, so the random factor of sklearn won't change
np.random.seed(123)

# Assign spreadsheet to file
file = 'QnC_V3.xlsx'


# PIPELINE TO QUESTIONS' RATINGS DATA
# Deploy questions' ratings DataFrame from spreadsheet
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
print(df_sorted)


# 150 Fake user profiles to deploy .csv file
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

# Print out users' clusters
print("user001 is categorised to cluster {}".format(user001_label))
print("user002 is categorised to cluster {}".format(user002_label))
print("user003 is categorised to cluster {}".format(user003_label))

# Calculate how the model performs using inertia
print("KMeans model's inertia: {}".format(kmeans.inertia_))

# -----------------
# Organize the labels of fake users in DataFrame
fake_users_label_df = pd.DataFrame({'LABELS': fake_users_label})
# Create the .csv file
result = pd.concat([fake_users_df, fake_users_label_df], axis=1)
result_csv = result.to_csv(index=False)

def get_user_category(user_profile_answer):
    category = pipeline.predict(np.array([
        user_profile_answer.answers
    ]))
    return category