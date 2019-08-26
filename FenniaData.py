# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 18:00:42 2019

@author: h17163
"""

import pandas as pd
from scipy.stats import ttest_ind

# %% FUNCTIONS
# Get multiple items from a list


def getVar(searchList, ind): return [searchList[i] for i in ind]

# 2-sample t-test


def tt2df(df, yvar_col='target', p_thre=0.05, full=0):
    """ Performs 'two-sample t-tests' for selected outcome
    variable in a dataframe with columns containing binomial variables
    yvar_col = [outcome variable] p_thre = [probability threshold] """

    yvar = df[yvar_col]
    # Create dataframe of x:s by excluding outcome variable (default = last col)
    df_x = df.drop(yvar_col, axis=1)

    def __split(io):
        "Split xvar on the basis of outcome variable: one or zero ('io')"
        a = yvar.apply(lambda x: x == io)
        a = a[a == True]
        return xvar.loc[xvar.index.isin(a.index)]
    # Create empty dataframe for results
    res = pd.DataFrame()
    # FOR all columns in df of x:s
    for n in range(0, df_x.shape[1]):
        xvar = df_x.iloc[:, n]
        # Test splitted x:s
        t = ttest_ind(__split(0), __split(1))
        # Concatenate results into one dataframe and rename columns after loop
        res = pd.concat([
            res,
            pd.DataFrame([t[0], t[1]], columns=[xvar.name]).transpose()
        ])
    res = res.rename(columns={0: 't_val', 1: 'p_val'})
    # Insert column of p_value check against threshold (default = 0.05)
    res['DEP_'+str(yvar.name)] = res['p_val'].apply(lambda x: x <= p_thre)

    # OUTPUT results: either TRUE/FALSE-list or full dataframe
    print("Statistical significance test@p_threshold="+str(p_thre))
    if res.iloc[:, -1][res.iloc[:, -1] == True].empty:
        print("No statistically significant dependencies")
    elif full == 0:
        return res.iloc[:, -1][res.iloc[:, -1] == True]
    else:
        return res


# %% READ DATA

path = "../data/data_g.csv"
data = pd.read_csv(path, encoding="latin-1", sep=";")

# %% WORKING WITH DATA...
# Get id:s and calculate the number of unique
indx = pd.unique(data.id)
indx_names = list(data.columns)
len(indx)

# Summarize data
summ = data.describe(include='all')

# T-test example
# Create a 3-column subset from full data (0/1-variables) and run t-test)
b = data.loc[:, getVar(indx_names, [2, 3, 4, 77])]
result_ttest = tt2df(b, 'poistunut')

# %% OTHER STUFF
# Pivot data by id (mean values by default)
piv_a = data.pivot_table(index='id')

# Take ended and on-going
ended = piv_a.loc[piv_a.poistunut > 0]
ongoing = piv_a.loc[piv_a.poistunut == 0]

# Compare "Ended vs. Ongoing"
comp = (ended.describe() - ongoing.describe())
