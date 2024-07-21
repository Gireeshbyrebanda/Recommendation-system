# Recommendation-system
# Importing libraries
import warnings 
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv(r'C:\Users\haree\OneDrive\Desktop\anime.csv')
data.head(5)
data.shape
data.info()
data.isnull().sum() #checking for missing values
df1 = data.dropna() #dropping the null values(rows)
df1.isnull().sum()
# checking for duplicate values

df1.duplicated().sum()
df1.columns
df1.info()
df1.describe()
df1.nunique() # checking unique values
df1['genre'].value_counts()
plt.figure(figsize=(10, 6))
plt.hist(df1['rating'], bins=20, edgecolor='black')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.grid(axis='y',alpha = 0.2 , linestyle="--") 
#linestyle is type of line
#alpha is the opacity of the grid
plt.show()
top_genres = df1['genre'].value_counts().head(10)

plt.figure(figsize=(10, 6))
top_genres.plot(kind='barh', color='skyblue', edgecolor='black')
plt.title('Top 10 Genres Distribution')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()
# Converting categorical features "genre" into numerical representations using One-hot encode

genres = df1['genre'].str.get_dummies(sep=', ')
df_numerical = pd.concat([df1.drop('genre', axis=1), genres], axis=1)
# standardization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numerical_cols = ['rating', 'members']
df_numerical[numerical_cols] = scaler.fit_transform(df_numerical[numerical_cols])

df_numerical.head()
genre_counts = df1['genre'].value_counts().head(10)

plt.figure(figsize=(12, 6))
plt.bar(genre_counts.index, genre_counts.values, color='skyblue')
plt.title('Distribution of Genres')
plt.xlabel('Genres')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
plt.figure(figsize=(16, 16))

# Distribution of Ratings
plt.hist(df_numerical['rating'], bins=20, color='#003666', alpha=0.7, label='Rating')

# Distribution of Members
plt.hist(df_numerical['members'], bins=20, color='skyblue', alpha=0.7, label='Members')

plt.title('Distribution of Ratings and Members')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', alpha=0.7)
plt.show()
from sklearn.metrics.pairwise import cosine_similarity


def recommend_similar_anime(df, target_anime, threshold=0.5):
    df['episodes'] = pd.to_numeric(df['episodes'], errors='coerce')
    df = df.dropna(subset=['episodes'])

    # Extract features for similarity computation
    features = df.drop(['name', 'type'], axis=1)
    
    # Find the target anime's features
    target_row = features[df['name'] == target_anime]
    
    if target_row.empty:
        return f"Anime '{target_anime}' not found in the dataset."
    
    similarities = cosine_similarity(target_row, features).flatten()
    

    similar_indices = np.where((similarities > threshold) & (df['name'] != target_anime))[0] #get index of similar anime leaving the target anime
    
    recommended_anime = df.iloc[similar_indices]['name'].tolist() #return list of recommended animes
    
    return recommended_anime
    recommendations = recommend_similar_anime(df_numerical, 'Nana', threshold=0.5)
recommendations_series = pd.Series(recommendations)
top_5_recommendations = recommendations_series.value_counts().head(5)
print(top_5_recommendations)
df_numerical['liked'] = ((df_numerical['rating'] >= 1)).astype(int)
df_numerical.head()
df_numerical.isnull().sum()
df1=df_numerical.dropna()
df1.isnull().sum()
df1.describe()
print(df1['liked'].value_counts())
# spilitting data into training and testing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X = df1.drop(['name', 'type','anime_id'], axis=1)
y = df1['liked']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
