
# %% import
import pandas as pd
import numpy as np
import functions as func
import seaborn as sns
import matplotlib.pyplot as plt

# %% read data
path = "../data/data_g.csv"
data = pd.read_csv(path, encoding="latin-1", sep=";")
amountOfUsers = pd.unique(data.id)
columsn = list(data.columns)
print("Amount of users is " + str(len(amountOfUsers)))

# %% Take a test sample
dataSmall = data.head(200000)
dataSmall = func.prepare(dataSmall)
print(dataSmall.describe)

# %% Summarize data
data_cleaned = data.drop(columns="timestamp")
data_cleaned = data.dropna()
data_cleaned2 = func.prepare(data)
# Very slow
groups = data_cleaned.groupby('id').agg(lambda x: x.tolist())

summ = data_cleaned.describe(include='all')

# %% Cleaning

# Pivot data by id (mean values by default)
piv_a = data_cleaned.pivot_table(index='id')

# Take ended and on-going
ended = data_cleaned.loc[data_cleaned.poistunut > 0]
ongoing = data_cleaned.loc[data_cleaned.poistunut == 0]

ended_liik = data_cleaned.loc[data_cleaned.liikenne_poistunut == 0]
ended_fetu = data_cleaned.loc[data_cleaned.kasko_poistunut == 0]
ended_kasko = data_cleaned.loc[data_cleaned.fetu_poistunut == 0]


#groups_ended = ended.groupby('id').agg(lambda x: x.tolist())
#groups_ongoing = ongoing.groupby('id').agg(lambda x: x.tolist())


print("Amount that ended " + str(ended.size/(ended.size + ongoing.size)))
# Compare "Ended vs. Ongoing"
comp = (ended.describe() - ongoing.describe())

# Preview the first 5 lines of the loaded data
data_cleaned.head()


# %% Plot
# Basic correlogram
print(data_cleaned.columns)
sns.distplot(data_cleaned.ika)
sns.distplot(data_cleaned.ika)

# sns.pairplot(data)
# sns.plt.show()

# %% Testing
# T-test example
# Create a 3-column subset from full data (0/1-variables) and run t-test)
b = data.loc[:, func.getVar(indx_names, [2, 3, 4, 77])]
result_ttest = func.tt2df(b, 'poistunut')
