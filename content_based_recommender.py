#############################
# Content Based Recommendation
#############################

#############################
# Developing Recommendations Based on Movie Reviews
#############################


"""
--> budget: movie budget
--> genres : genre
--> homepage : homepages
--> id: ids in the data set
--> imdb_id: id in imdb
--> overview: description
"""


# 1. Creating the TF-IDF Matrix
# 2. Creating the Cosine Similarity Matrix
# 3. Making Suggestions Based on Similarities
# 4. Preparation of Working Script


#################################
# 1. Creating the TF-IDF Matrix
#################################

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# https://www.kaggle.com/rounakbanik/the-movies-dataset
df = pd.read_csv(r"dataset\the_movies_dataset\movies_metadata.csv", low_memory=False) 
df.head()
df.shap

df["overview"].head()
# We need to process these texts

tfidf = TfidfVectorizer(stop_words="english")
# Commonly used expressions that do not have any measurement value, such as in, on, an, should be removed from the data set.


df[df['overview'].isnull()]

df['overview'] = df['overview'].fillna('')  # NaN ones replaced with ''
tfidf_matrix = tfidf.fit_transform(df['overview'])
tfidf_matrix.shape   #(45466, 75827)  (descriptions, unique words)
df['title'].shape

tfidf.get_feature_names()
tfidf_matrix.toarray()


#################################
# 2. Creating the Cosine Similarity Matrix
#################################

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
# takes the matrix whose similarity is desired to be calculated, can be entered as one argument or two arguments


cosine_sim.shape   #overviews
cosine_sim[1]
# Row 1 contains the similarity score of the first movie with all other movies

#################################
# 3. Making Suggestions Based on Similarities
#################################

indices = pd.Series(df.index, index=df['title'])
# The name of the movie was given to the index of the series, and next to it, numerical information was given in which index the movie with this name was placed.


indices.index.value_counts()
# Multiples in titles have been deleted
indices = indices[~indices.index.duplicated(keep='last')]  

indices["Cinderella"]

indices["Sherlock Holmes"]

movie_index = indices["Sherlock Holmes"]

cosine_sim[movie_index] 
# If you go to cosine_sim with this index, sherlock holmes will be selected, in this case similar scores between sherlock and other movies will be accessed.

similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])

movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
# Top 10 movies ranked in descending order

df['title'].iloc[movie_indices]

#################################
# 4. Preparation of Working Script
#################################

def content_based_recommender(title, cosine_sim, dataframe):
    indices = pd.Series(dataframe.index, index=dataframe['title'])
    indices = indices[~indices.index.duplicated(keep='last')]
    # title'ın index'ini yakalama
    movie_index = indices[title]
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    return dataframe['title'].iloc[movie_indices]

content_based_recommender("Sherlock Holmes", cosine_sim, df)

content_based_recommender("The Matrix", cosine_sim, df)

content_based_recommender("The Godfather", cosine_sim, df)

content_based_recommender('The Dark Knight Rises', cosine_sim, df)


def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    dataframe['overview'] = dataframe['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(dataframe['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = calculate_cosine_sim(df)
content_based_recommender('The Dark Knight Rises', cosine_sim, df)

