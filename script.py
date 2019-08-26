
import pandas as pd
import numpy as np
import functions as func


path = "../data/data_g.csv"
data = pd.read_csv(path, encoding="latin-1", sep=";")
indx = pd.unique(data.id)
indx_names = list(data.columns)
print("Amount of data is" + str(len(indx)))
# Summarize data
summ = data.describe(include='all')
# T-test example
# Create a 3-column subset from full data (0/1-variables) and run t-test)
b = data.loc[:, func.getVar(indx_names, [2, 3, 4, 77])]
result_ttest = func.tt2df(b, 'poistunut')

# Pivot data by id (mean values by default)
piv_a = data.pivot_table(index='id')

# Take ended and on-going
ended = piv_a.loc[piv_a.poistunut > 0]
ongoing = piv_a.loc[piv_a.poistunut == 0]


print("Amount that ended" + str(ended.size/len(indx)))
# Compare "Ended vs. Ongoing"
comp = (ended.describe() - ongoing.describe())

# Preview the first 5 lines of the loaded data
data.head()
test = data.sort_values(["id", "timestamp"]).head(10000)
test['kanava_myynti'] = test['kanava_myynti'].fillna(value='0')
test = func.prepare(test)
