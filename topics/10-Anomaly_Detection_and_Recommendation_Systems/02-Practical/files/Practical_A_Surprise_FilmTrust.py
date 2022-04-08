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
# # Recommendation systems using Surprise library
#
# This notebook indicates how surprise can be used to generate recommendations given (user,item) ratings. In this case we use the FilmTrust data on movie ratings.
#
# It is based on https://blog.cambridgespark.com/tutorial-practical-introduction-to-recommender-systems-dbe22848392b
#
# The following cell shows the commands that should be run to install the necessary packages, using `conda`. See
# https://surprise.readthedocs.io/en/stable/getting_started.html:
#
# `conda install -c conda-forge scikit-surprise`
#
# `conda install -c anaconda joblib`

# %% [markdown]
# Need to get the data (a zip file) and to use just the ratings.txt file within it

# %%
import surprise
import pandas as pd
import numpy as np
import requests
import io
import os
import zipfile

# %% [markdown]
# Download zip file from url if it has not been downloaded already
# See https://codereview.stackexchange.com/a/231221

# %%
def download_zipfile(url, zipF):
  if not os.path.isfile(zipF):
    with open(zipF, 'wb') as out:
      out.write(requests.get(url).content)


def read_zipfile_item(zipF, fn):
  with zipfile.ZipFile(zipF) as zip_file:
    with zip_file.open(fn) as f:
      return f.read().decode('utf8')

url = 'https://guoguibing.github.io/librec/datasets/filmtrust.zip'
zipF = 'data/filmtrust.zip'
fn = 'ratings.txt'
download_zipfile(url, zipF)
# io.StringIO converts the stream into a file-like object, as expected by pd.read_table()
ratings = pd.read_table(io.StringIO(read_zipfile_item(zipF, fn)), sep = ' ', names = ['uid', 'iid', 'rating'])
print(ratings.head())

# %% [markdown]
# Need to check that the ratings have the expected range. Suprise defaults to [1,5] but filmtrust uses [0.5, 4] as seen below.

# %%
# Get the rating range
lowest_rating = ratings['rating'].min()
highest_rating = ratings['rating'].max()
print('Ratings range from {0} to {1}'.format(lowest_rating, highest_rating))

# %% [markdown]
# Tell surprise about the rating_scale used by this data

# %%
reader = surprise.Reader(rating_scale=(lowest_rating, highest_rating))
data = surprise.Dataset.load_from_df(ratings, reader)

# %% [markdown]
# Now use the SVD-based recommender algorithm, treating all the data as the training set
# Of course, it is generally much better to split into separate training and test sets.

# %%
alg = surprise.SVD()
output = alg.fit(data.build_full_trainset())

# %% [markdown]
# Now check how well the recommender predicts when uid=50 and iid=52 (which is a known rating)

# %%
# The uids and iids need to be converted to strings for prediction
pred = alg.predict(uid='1', iid='4')
predScore = pred.est
actualScore = ratings.query('uid == "1" & iid == "4"').iloc[0]['rating']
print('Actual score is {0} and predicted score (using SVD-based recommender algorithm) is {1}'.format(actualScore,predScore))

# %% [markdown]
# More practically, we would like to generate ratings for all movies that were not rated by a given user

# %%
# Get a list of all movie ids
iids = ratings['iid'].unique()
# Get a list of iids that user '50' has rated
iids50 = ratings.loc[ratings['uid']==50, 'iid']
# Remove the  iids that user 50 has rate from all the list of movie ids
iids_to_pred = np.setdiff1d(iids, iids50)

# %% [markdown]
# We arbitrarly set the rating to -1 (standing in for 'unrated') for the iids to be predicted
# We can then predict a batch of ratings together (in the testset)

# %%
testset = [[50, iid, -1] for iid in iids_to_pred]
predictions = alg.test(testset)

# %% [markdown]
# Now that we have the predicted ratings, we can use pick one with the highest rating for recommendation purposes

# %%
pred_ratings = np.array([pred.est for pred in predictions])
# Find the index of the maximum predicted rating
i_max = pred_ratings.argmax()
# Use this to find the corresponding iid to recommend
iid = iids_to_pred[i_max]
print('Top item for user 50 has iid {0} with predicted rating {1}'.format(iid, pred_ratings[i_max]))

# %%
# Exercise: Identify the top 5 items to suggest to this user.
# You might find Numpy's argpartition function useful for this purpose.
