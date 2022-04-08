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
# # Example using pyod

# %%
import pandas as pd
import scipy.io
import os.path as path
from zipfile import ZipFile
import urllib.request
from scipy.io import loadmat

# %% [markdown]
# The dataset was sourced from ODDS and downloaded from [here](https://www.dropbox.com/s/n3wurjt8v9qi6nc/mnist.mat)

# %%
vNames = ["v"+str(i) for i in range(100)]
zipDataFile = 'data/mnist.zip'
matDataFile = 'data/mnist.mat'
if path.exists(matDataFile):
    print("Read {}".format(matDataFile))
    mnist = loadmat(matDataFile)
    X = pd.DataFrame(data=mnist['X'], columns=vNames)
    y = pd.DataFrame(data=mnist['y'], columns=['label'])
elif path.exists(zipDataFile):
    print("Read {}".format(zipDataFile))
    with ZipFile(zipDataFile, mode='r') as dataZip:
        # Read the predictor (X) matrix based on the selected pixel columns
        with dataZip.open('X.csv') as mnistX:
            X = pd.DataFrame(data=pd.read_csv(mnistX,header=None), columns=vNames)
        # Read the label (y) vector where
        with dataZip.open('y.csv') as mnistY:
            y = pd.DataFrame(data=pd.read_csv(mnistY,header=None), columns=['label'])
else:
    #with urllib.request.urlopen('https://www.dropbox.com/s/n3wurjt8v9qi6nc/mnist.mat') as response:
    #    with open("mnist.mat", "wb") as f:
    #        f.write(response.read())
    print('download the file using wget')
X.shape

# %% [markdown]
# Take a quick look at the data (note that its dimensions have already been reduced to 100)

# %%
X.head()

# %% [markdown]
# Now have a look at the labels, which are 0 (inlier) and 1 (outlier) - as decided by a human observer. This is taken as the ground truth.

# %%
y

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
n_test = len(y_test.index)
n_all = len(y.index)

# %% [markdown]
# Convert to simple numpy arrays for comparison

# %%
yTrain = y_train['label'].to_numpy()
yTest = y_test['label'].to_numpy()

# %% [markdown]
# Assign the outlier_fraction. Note that we are "cheating" here, because the data is labeled and so we "know" the outliers. However, if the data was unlabeled we would need to estimate the outlier_fraction.

# %%
n_outlier = len(y[(y['label']==1)])
outlier_fraction = n_outlier / float(n_all)
print('The entire set has {} rows with {} outliers so the outlier fraction is {}'.format(n_all,n_outlier,outlier_fraction))

# %%
from pyod.models.knn import KNN

knn=KNN(contamination=outlier_fraction)
knn.fit(X_train)

# get the prediction labels of the training data
y_train_pred = knn.labels_ 
y_train_pred

# %% [markdown]
# Get the outlier scores of the training data

# %%
y_train_scores = knn.decision_scores_
y_train_scores

# %% [markdown]
# Get the outlier predictions on the test data

# %%
y_test_pred = knn.predict(X_test)  
y_test_pred

# %% [markdown]
# Get the outlier scores of the test data

# %%
y_test_scores = knn.decision_function(X_test)
y_test_scores

# %% [markdown]
# Find the number of 'misclassified' digits in the test set

# %%
n_errors = (y_test_pred != yTest).sum()
print('No of Errors when applying knn to test set: {}'.format(n_errors))

# %% [markdown]
# Compute the accuracy of the outlier detection classifier on the test set

# %%
accuracy = (n_test-n_errors)/float(n_test)
print('Accuracy when applying knn to test set: {}'.format(accuracy))

# %% [markdown]
# Derive the probabilities of class 0 (inlier) and class 1 (outlier) for each digit. Note that these probabilities sum to 1. The digit is an outlier if the latter (class=1) probability is greater than the former (class=0).

# %%
y_test_score_prob = knn.predict_proba(X_test, method='linear')
y_test_score_prob

# %% [markdown]
# Outlier detection can be viewed as a classification so we can use the scikit-learn classification report to vierw how well the outlier classifier did.

# %%
from sklearn.metrics import classification_report
print(classification_report(y_test, y_test_pred))

# %%
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_test_pred)

# %%
trueInlier, falseOutlier, falseInlier, trueOutlier = confusion_matrix(y_test, y_test_pred).ravel()
print("There were {} digits, of which".format(n_test))
print("{} inliers were classified correctly; {} outliers were classified correctly.".format(trueInlier,trueOutlier))
print("{} outliers were classified incorrectly; {} inliers were classified incorrectly.".format(falseOutlier,falseInlier))

# %% [markdown]
# Note that the false negative count is approximately the same as the false positive count, because the `outlier_fraction` is approximately correct, so the score threshold is approximately correct. However, more outliers were misclassified than were classified correctly. This is disappointing but is often found when the data is so unbalanced.
#
# _Exercise_
#
# 1. We used the knn outlier detection algorithm and implictly used its default hyperparameter values: `n_neighbours` = 5, `method` = 'largest', `metric` = 'minkowski' and `p` = 2. Note that a Minkowski distance metric with p = 2 implies Euclidean distance. Try other hyperparameter value combinations, e.g. `k` = [3, 5, 7]; `method` = ['largest', 'mean', 'median']. Which combination gives the best performance?
# 2. Try other outlier detection algorithms from pyod, such as those mentioned in class. Note that pyod offers many algorithms. Some example output from [pyod documentation](https://pyod.readthedocs.io/en/latest/) is shown below

# %%
from IPython.display import IFrame
IFrame("../data/selected_pyod.pdf", width=700, height=600)
