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
# # Assignment 3: Association Analysis

# %% [markdown]
# To read Excel files, you might need to install the `xlrd` package, using something like:
#
# Select the conda environment you use for this module (skip this step if you have not created a separate environment for this 
#
#     conda activate myEnvironment  # where myEnvironment is the conda environment you use for this module
#
# then install as usual
#
#     conda install xlrd
#
# Note:
#
#  * To run these command from within a notebook you prefix command with !  
#  * You will also need the package `mlxtend` which you installed as part of the Week 10 - ARM practical.

# %% [markdown]
# You may find the following useful to obtain the data from the UCI data repository, and to read it into a dataframe.

# %%
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

import requests, os
csvUrl = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/groceries.csv"
csvFile = 'data/groceries.csv'
xlUrl = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx'
xlFile = 'data/Online Retail.xlsx'
dataFile = xlFile
url = xlUrl
if not os.path.exists('data'):
    os.makedirs('data')
if not os.path.isfile(dataFile):
    r = requests.get(url)
    with open(dataFile, 'wb') as f:
        f.write(r.content)
if (dataFile == xlFile):
    df = pd.read_excel(dataFile)
else:
    df = pd.read_csv(dataFile)
df.head()

# %% [markdown]
# The following lines tidy up the description column, ensure that every row is assigned an invoice number, and that they represent actual transactions.

# %%
df['Description'] = df['Description'].str.strip()
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
df['InvoiceNo'] = df['InvoiceNo'].astype('str')
df = df[~df['InvoiceNo'].str.contains('C')]
df.shape

# %%
# Get the unique list of countries
print(df['Country'].unique())

# %% [markdown]
# __Task 1.1__: Select the transactions arising from the `Country` having _9042_ records in the dataframe and convert them to the OneHotEncoded form, where each column has (0,1) values representing the (absence,presence) of that product in a given basket, where each basket (row) is labeled by its `InvoiceNo`.
# Use mlxtend's `apriori` function to find the frequent itemsets where the minimum support threshold is set to 0.02. You should check the number of frequent itemsets &mdash; you should find there are 528. 
#
# Hints
# 1. Use `groupby` and `size()` to determined the number of rows per `Country`.
# 2. Use `groupby` and `sum()` on the `Quantity` to encode as 0 and positive integer, and `reset_index()` so that the rows are labeled by `InvoiceNo`. Remember to set any positive numbers to 1 rather than a frequency count.

# %%
## BEGIN YOUR ANSWER HERE

# %%

# %%
## END YOUR ANSWER HERE

# %% [markdown]
# __Task 1.2__: Use mlxtend's `association_rules` function to find the association rules where the minimum lift threshold is 1.
# Sort them in non-increasing order of lift (largest to smallest).
# You should then check the number of such rules &mdash; you should find there are 738 such rules.

# %%
## BEGIN YOUR ANSWER HERE

# %%

# %%
## END YOUR ANSWER HERE

# %%
rules.sort_values(by='lift',ascending=False)
rules.head()

# %% [markdown]
# add new column storing the rule length

# %%
rules["rule_len"] = rules.apply(lambda row: len(row["antecedents"])+len(row["consequents"]), axis=1)
rules[rules["rule_len"]==2].sort_values(by='lift',ascending=False).head()

# %% [markdown]
# __Task 1.3__: Comparing row indexes 452 and 453 above, which have the same lift value (32.642857), by reviewing the rule metrics above, would it be better to suggest 'BLUE VINTAGE SPOT BEAKER' to someone who already had 'PINK VINTAGE SPOT BEAKER', or vice-versa? Give reasons for your answer.


# %%
## BEGIN YOUR ANSWER HERE

# %%

# %% [markdown]
# **Comments**
#
#  *

# %%
## END YOUR ANSWER HERE
