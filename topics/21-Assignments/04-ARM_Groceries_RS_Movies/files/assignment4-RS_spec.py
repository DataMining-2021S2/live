# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Assignment 3: Recommender Systems

# %% [markdown]
# We use the well-known MovieLens dataset (in this case the small version). You may find the following useful to obtain the data from the GroupLens repository, and to read it into a dataframe.

# %%
import os, requests
import numpy as np

#mlSize = "ml-1m"
#mlSize = "ml-100k"
mlSize = "ml-latest-small"
zipUrl = 'http://files.grouplens.org/datasets/movielens/'+mlSize+'.zip'
zipFile = 'data/'+mlSize+'.zip'
dataFile = zipFile
url = zipUrl
dataDir = 'data'
if not os.path.exists(dataDir):
    os.makedirs(dataDir)
if not os.path.isfile(zipFile):
    r = requests.get(zipUrl)
    with open(zipFile, 'wb') as f:
        f.write(r.content)

# Need to unzip the file to read its contents
import zipfile
with zipfile.ZipFile(zipFile,"r") as zip_ref:
    zip_ref.extractall(dataDir)

# %%
# Read the ratings data into a dataframe
import pandas as pd
fn = 'ratings.csv'
colNames = ['UserID','MovieID','Rating','Timestamp']
ratingsDf = pd.read_csv('data/'+mlSize+'/'+fn, names=colNames, skiprows=1, sep=',', engine='python')
ratingsDf.head()

# %%
# Timestamps are difficult for humans to read, so convert them into a more readable format
import time
ts = time.gmtime()
print("Timestamp now is {} which is {}".format(str(ts),time.strftime("%Y-%m-%d %H:%M:%S", ts)))

# %%
ratingsDf['DateTime'] = pd.to_datetime(ratingsDf['Timestamp'],unit='s')
ratingsDf.drop(columns=['Timestamp'], inplace=True)
ratingsDf.head()

# %% [markdown]
# __Task 2.1__: Based on the code above, read the `movies.csv` data files into data frames. In the past,
# GroupLens also included user data. Comment on why that is no longer the case and what this means for
# recommendation algorithms.
#

# %%
## BEGIN YOUR ANSWER HERE

# %%

# %% [markdown]
# **Comments**
#
#  *

# %%
## END YOUR ANSWER HERE

# %% [markdown]
# The following code can be used to filter the number of Movies. Choosing a large threshold (like 200) ensures that only "blockbuster" movies with that number of aggregate ratings will be considered. This is convenient (much reduced runtimes!) when developing your solution, but a less stringent threshold should be used for the result you hand in (100 is required).

# %%
#minMovieRatings = 200
minMovieRatings = 100
filterMovies = ratingsDf['MovieID'].value_counts() > minMovieRatings
filterMovies = filterMovies[filterMovies].index.tolist()
print('Filtered ratings - omitting movies with less than {} ratings results in {} ratings'.format(minMovieRatings, len(filterMovies)))

# %% [markdown]
#  __Task 2.2__
# <br /> a) You should apply a similar filter to the Users, selecting only those who rated at least 80 movies.
# <br /> b) You should then apply `filterUsers` and `filterMovies` filters to the ratings dataframe, you might find the `isin(filteredSet)` function useful.
# <br /> c) You are given some code below to help you visualise the distribution of counts of ratings by user. Hence or otherwise comment on the similarities and differences between the distributions of ratings by user and of ratings by Movie.
# <br /> d) Save the filter dataframe as `filteredRatingsDf`.

# %%
## BEGIN YOUR ANSWER HERE

# %%

# %% [markdown]
# **Comments**
#
#  *

# %%
## END YOUR ANSWER HERE

# %% [markdown]
# Using the filtered ratings dataframe, count the ratings per User and plot this data in a histogram. 

# %%
# Get the userRated groupby object
userRated = filteredRatingsDf.groupby(['UserID'])[['Rating']].count().sort_values('Rating', ascending=False)
print(userRated.head())
print(userRated.describe())

# %%
# Compute summaries of the userRated object
medianNumRatingsPerUser = userRated.median()['Rating']
minNumRatingsPerUser = userRated.min()['Rating']
maxNumRatingsPerUser = userRated.max()['Rating']
numUniqueFilteredUsers = filteredRatingsDf['UserID'].nunique() 
print("There are {} users who rated movies, with the median and maximum number of movies rated per user being {} and {}".format(numUniqueFilteredUsers,medianNumRatingsPerUser,maxNumRatingsPerUser))

# %%
# Plot the (plain) distribution of rating counts by user
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline
rangeUserRatings = np.arange(minNumRatingsPerUser-1,maxNumRatingsPerUser)
plt.hist(userRated['Rating'], bins=30)
xlabel = 'How many movies were rated by each user'
ylabel = 'Number of Users'
title = 'Distribution of rating counts per User'
plt.ylabel(ylabel)
plt.xlabel(xlabel)
plt.axvline(x=medianNumRatingsPerUser, color='r')
plt.title(title)
plt.show()

# %%
# Plot the annotated distribution of rating counts per user
import seaborn as sns
g = sns.displot(data=userRated['Rating'], bins=30, kde=True, rug=True)
g.set_titles(title)
g.set_axis_labels(xlabel,ylabel)
plt.show()

# %% [markdown]
# __Task 2.3__: Repeat Task 2.2 above, but deriving the average ratings rather than their counts.
# The distribution of average ratings per user differs from the distribution of how many movies a user reviews.
# You can also look at the the distribution of average ratings per movie and the distribution of how many users rate each movie.
# From your understanding of a rating system, comment on the similarities and differences.

# %%
## BEGIN YOUR ANSWER HERE

# %%

# %% [markdown]
# **Comments**
#
#  *

# %%
## END YOUR ANSWER HERE

# %% [markdown]
# __Task 2.4__: Load the (filtered) movies ratings data from the dataframe we have been exploring into the preferred 3-column format used by the `scikit-suprise` package. Now benchmark the performance (in terms of RMS error, time to fit, and time to generate predictions for test data) of the `SVD()`, `SlopeOne()`, `NMF()`, `KNNBasic()` recommendation algorithms. Discuss the strengths and weaknesses of each algorithm, based on its benchmarked results.
#

# %%
# 1. `scikit-surprise` provides a `cross_validate` function that can be used to estimate the test error in the test data, using the requested error metric.
# 2. When collecting the benchmark data, it is convenient to loop over the algorithms and to add the results for each algorithm as a row to the benchmark dataframe.
# 3. The following python code can be used to add `results` as a row to a `benchmark` dataframe.
# You are advised to plot the results, and to pay attention to fit, test and overall times and how they vary between the algorithms. Comment on what you find.

# %%
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate
from surprise import KNNBasic, NMF, SlopeOne, SVD

reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(filteredRatingsDf[['UserID', 'MovieID', 'Rating']], reader)

# %%
benchmark = pd.DataFrame()
algorithms = [KNNBasic(), SVD(), SlopeOne(), NMF()]

# Iterate over all algorithms
for algorithm in algorithms:
    # Perform cross validation
    results = cross_validate(algorithm, data, measures=['RMSE','MAE'], cv=3, verbose=False)
    
    # Get results & append algorithm name
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    benchmark = benchmark.append(tmp, ignore_index=True)
    
benchmark.set_index('Algorithm').sort_values('test_rmse')
benchmark['comp_time'] = benchmark['fit_time'] + benchmark['test_time']

print(benchmark)

# %%
## BEGIN YOUR ANSWER HERE

# %%

# %% [markdown]
# **Comments**
#
#  *

# %%
## END YOUR ANSWER HERE
