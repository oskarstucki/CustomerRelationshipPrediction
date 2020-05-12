
# %% import
import pandas as pd
import numpy as np
import functiontions as function
import plot as plotter
import preprocessing as pre
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn_pandas import DataFrameMapper


# %% read data
path = "../data/data_g.csv"
data = pd.read_csv(path, encoding="latin-1", sep=";")
amountOfUsers = pd.unique(data.id)
cols = list(data.columns)
print("Amount of users is " + str(len(amountOfUsers)))
data_drop_duplicates = data.drop_duplicates(subset=['id'], keep='last')
churners = data_drop_duplicates.where(
    data_drop_duplicates["poistunut"] == 1).dropna(subset=["poistunut"])
stayed = data_drop_duplicates.where(
    data_drop_duplicates["poistunut"] == 0).dropna(subset=["poistunut"])
print("Percentage of churners is " +
      str(len(churners.index)/len(data_drop_duplicates.index)))
sns.distplot(data_drop_duplicates["ika"])
# %% Check the distribution
target_count = data_drop_duplicates.poistunut.value_counts()
print('Stayed:', target_count[0])
print('Left:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

target_count.plot(kind='bar', title='Distribution of the')

# %% Split the data to two classes
data_sampled = A = pd.concat([churners, stayed.head(len(churners.index))], axis=0)
data_drop_duplicates = data_drop_duplicates.sample(frac=1)

# %% Check the distribution
target_count = data_sampled.poistunut.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

target_count.plot(kind='bar', title='Count (target)')

# %% Take a test sample

dataSample = data_drop_duplicates.head(60000)
dataPros = pre.prepare(dataSample.copy())
columns = list(dataPros.columns)

dataSample_n = data_drop_duplicates.head(60000)
dataPros_n = pre.prepare(dataSample_n.copy())

# print(dataPros.describe)
# %% Check the distribution
target_count = dataSample.poistunut.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

target_count.plot(kind='bar', title='Count (target)')
# %% Oversample

dataProsSample = pre.oversample(dataPros)


# %% Label

y = dataPros.loc[:, "poistunut"]
drop = ["poistunut", "kasko_poistunut", "fetu_poistunut", "liikenne_poistunut"]
dataLabel, labels = pre.labeling(dataPros.drop(columns=drop))

y_n = dataPros_n.loc[:, "poistunut"]
dataLabel_n, labels_n = pre.labeling(dataPros_n.drop(columns=drop))
# %% Scale

scaler = StandardScaler()
# dataProsSampleScaled = scaler.fit_transform(dataLabel)

mapper = DataFrameMapper([(dataLabel.columns, StandardScaler())])
scaled_features = mapper.fit_transform(dataLabel.copy())
dataProsSampleScaled = pd.DataFrame(scaled_features,
                                    index=dataLabel.index,
                                    columns=dataLabel.columns)

dataProsSample_n = scaler.fit_transform(dataLabel_n)
# dict_labels["scaler"] = scaler.get_params(deep=True)

# %% Feature selection
dataProsSampleScaledSelected, classes = function.featureSelection(
    dataProsSampleScaled, y)
dataProsSampleScaledSelected_n, classes_n = function.featureSelection(
    dataProsSample_n, y_n)
# %% Transform data
pca = PCA(n_components=2)

X = pca.fit_transform(dataProsSampleScaledSelected)
x_n = pca.fit_transform(dataProsSampleScaledSelected_n)
plotter.plot_2d_space(X, y, 'Imbalanced dataset (2 PCA components)')

# %%
smote = SMOTE(ratio='minority')
X_sm, y_sm = smote.fit_sample(X, y)

plotter.plot_2d_space(X_sm, y_sm, 'SMOTE over-sampling')


# %%

matrixes = function.run(dataProsSampleScaledSelected_n, y_n, dataLabel.columns)

# %%
dataTail = data.tail(1000)  # .where(data["poistunut"] == 1)
dataTail = dataTail.dropna(subset=["poistunut"])
dataTailPros = pre.prepare(dataTail.copy())

y_tail = dataTailPros.loc[:, "poistunut"]
drop = ["poistunut", "kasko_poistunut", "fetu_poistunut", "liikenne_poistunut"]
dataLabelTail, labelsTail = pre.labeling(dataTailPros.drop(columns=drop))


# %%


x_train, x_test, y_train, y_test = train_test_split(
    dataProsSampleScaledSelected_n, y_n, test_size=0.25, random_state=0)

rfc = RandomForestClassifier(n_estimators=100, max_depth=10,
                             random_state=0)
rfc.fit(x_train, y_train)
function.report(y_test, rfc.predict(x_test), columns, "RFC")

dt = DecisionTreeClassifier().fit(x_train, y_train)
function.report(y_test, dt.predict(x_test), columns, "DT")
function.report(y_n, dt.predict(dataProsSampleScaledSelected_n), columns, "DT")
param_grid = {"criterion": ["gini", "entropy"],
              "min_samples_split": [2, 10, 20],
              "max_depth": [None, 2, 5, 10],
              "min_samples_leaf": [1, 5, 10],
              "max_leaf_nodes": [None, 5, 10, 20],
              }

dt_s = DecisionTreeClassifier()
n_iter_search = 20
random_search = GridSearchCV(dt_s, param_grid=param_grid,
                             scoring="accuracy", cv=5, n_jobs=-1)
random_search.fit(x_train, y_train)
function.report(y_test, random_search.predict(x_test), columns, "DT with search")


# %% Correlation

corr = dataTail.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0, len(data.columns), 1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)
plt.show()

sns.heatmap(corr, annot=True, cmap=plt.cm.Reds)
plt.show()
