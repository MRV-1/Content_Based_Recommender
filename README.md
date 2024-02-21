# Content Based Recommender
It is a recommendation system that makes suggestions based on the similarities of the contents 📰

### 1. Creating the TF-IDF Matrix
### 2. Creating the Cosine Similarity Matrix
### 3. Making Suggestions Based on Similarities
### 4. Preparation of Working Script



### Business Problem

The newly established online movie viewing platform wants to make movie recommendations to its users.

Since the login rate of its users is very low, it cannot collect user habits. For this reason, collaborative filtering
cannot develop product recommendations using these methods.

However, it knows which movies users watched from their browser footprints (cookies). Make movie suggestions based on this information.

### Dataset Information

movies_metadata.csv contains basic information about 45000 movies.

--> budget: movie budget
--> genres : genre
--> homepage : homepages
--> id: ids in the data set
--> imdb_id: id in imdb
--> overview: description

It is the overview variable that is required for us in this project.
Includes overview movie descriptions.


# 1. Creating the TF-IDF Matrix

A standardization process was carried out by taking into account the effects of each document both within itself and on the whole document. <br/>
Commonly used expressions that do not have any measurement value, such as in, on, an, should be removed from the data set. <br/>
There is a problem caused by the values in the tf-idf matrix to be created, so stop_words="english" was used. <br/>
If somehow two films containing ten expressions turn out to be close to each other, this would be biasing our results. <br/>



# 2. Creating the Cosine Similarity Matrix

Takes the matrix whose similarity is desired to be calculated, can be entered as one argument or two arguments


# 3. Making Suggestions Based on Similarities

In addition to the movie name in the index, a Series was created to indicate which index the movie with that name was in.

Multiples in titles have been deleted.


# 4. Preparation of Working Script


 # BONUS

### QUESTION: How can this work be carried out at the database level on a large scale?
### ANSWER: The 100 or 200 movies that users watch most are determined.
--> The operations performed here are performed for each of the 100 most popular movies reduced to a subset, and a recommendation set is created for each of them and this is kept in a table.

--> id [suggested ids] <br/>
--> [90, 12, 23, 45, 67] <br/>
--> [90, 12, 23, 45, 67] <br/>





