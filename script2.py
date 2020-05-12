# -*- coding: utf-8 -*-

from sklearn.datasets import make_blobs
from sklearn import svm
import numpy as np
import functions as func
import pandas as pd
import plot as plotter
import preprocessing as pre
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from tensorflow.keras.layers import Dropout
from tensorflow.keras.constraints import MaxNorm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# %% read data
path = "../data/data_g.csv"
data_org = pd.read_csv(path, encoding="latin-1", sep=";")
data = data_org.copy().drop_duplicates(subset=['id'],
                                       keep='last').sample(frac=1)
amountOfUsers = pd.unique(data.id)
cols = list(data.columns)
print("Amount of users is " + str(len(amountOfUsers)))

churners = data.where(data["poistunut"] == 1).dropna(subset=["poistunut"])
stayed = data.where(data["poistunut"] == 0).dropna(subset=["poistunut"])
print("Percentage of churners is " +
      str(len(churners.index) / len(data.index)))

print(data.info())

# %%

data_prepared = pre.prepare(data.copy())
customer_time = data_prepared.groupby(
    "asiakkuus_kesto").poistunut.value_counts(normalize=True).plot(kind="bar")

# %%
significant = [
    "E_LASKU", "asiakkuus_kesto", "fetu_sairaus_voim", "fetu_elain_voim",
    "fetu_irtaimisto_voim", "henkivakuutus", "kasko_ero_markkina_HA_sum",
    "kolari_tuleva_muutos_HA_sum", "kolari_tuleva_muutos_EIHA_sum",
    "liikenne_tuleva_muutos_HA_sum", "liikenne_tuleva_muutos_EIHA_sum",
    "kieli", 'liikenne_eralkm', "fetu_eralkm", "kasko_eralkm", "merkkikasko",
    "sukupuoli", "viim_muutto", "viim_myynti", "poistunut"
]

data_sig = data_prepared[significant].copy()
# %%
cat_data = data_prepared.select_dtypes(
    include=['object']).copy().astype('category')

cat_data = list(cat_data.columns)

for i in cat_data:
    data_prepared[i] = data_prepared[i].astype('category')

cat_data_sig = data_sig.select_dtypes(
    include=['object']).copy().astype('category')

cat_data_sig = list(cat_data_sig.columns)

for i in cat_data_sig:
    data_sig[i] = data_sig[i].astype('category')

print(data_prepared.info())

# %% Check the distribution
target_count = data_prepared.asiakkuus_kesto.value_counts()
print('Stayed:', target_count[0])
print('Left:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

target_count.plot(kind='bar', title='Distribution of customer durations')

# %% Label and split the values
bool_cols = [
    col for col in data_prepared
    if np.isin(data_prepared[col].dropna().unique(), [0, 1]).all()
]

scaling = data_prepared.drop(bool_cols + cat_data, axis=1)
others = data_prepared[[
    c for c in data_prepared.columns if c in bool_cols + cat_data
]]

bool_cols_sig = [
    col for col in data_sig
    if np.isin(data_sig[col].dropna().unique(), [0, 1]).all()
]

scaling_sig = data_sig.drop(bool_cols_sig + cat_data_sig, axis=1)
others_sig = data_sig[[
    c for c in data_sig.columns if c in bool_cols_sig + cat_data_sig
]]

drop = ["poistunut"]
# dataLabel, labels = pre.labeling(others.drop(columns=drop))
dataLabel = pd.get_dummies(others, columns=list(cat_data))
dataLabel_sig = pd.get_dummies(others_sig, columns=list(cat_data_sig))

# %% Scale

mapper = DataFrameMapper([(scaling.columns, StandardScaler())])
scaled_features = mapper.fit_transform(scaling.copy())
dataProsSampleScaled = pd.DataFrame(scaled_features,
                                    index=scaling.index,
                                    columns=scaling.columns)
frames = [dataLabel, dataProsSampleScaled]
X_data = pd.concat(frames, axis=1)

mapper_sig = DataFrameMapper([(scaling_sig.columns, StandardScaler())])
scaled_features_sig = mapper_sig.fit_transform(scaling_sig.copy())
dataProsSampleScaled_sig = pd.DataFrame(scaled_features_sig,
                                        index=scaling_sig.index,
                                        columns=scaling_sig.columns)
frames_sig = [dataLabel_sig, dataProsSampleScaled_sig]
X_data_sig = pd.concat(frames_sig, axis=1)

# %%

data_oldies_sig = X_data_sig.loc[X_data["asiakkuus_kesto_10y+"] == 1]
data_new_sig = X_data_sig.loc[X_data["asiakkuus_kesto_10y+"] != 1]

data_oldies = X_data.loc[X_data["asiakkuus_kesto_10y+"] == 1]
data_new = X_data.loc[X_data["asiakkuus_kesto_10y+"] != 1]

y = X_data.loc[:, "poistunut"]
y_sig = X_data_sig.loc[:, "poistunut"]
y_new = data_new.loc[:, "poistunut"]
y_old = data_oldies.loc[:, "poistunut"]

X_data = X_data.drop(columns="poistunut")
X_data_sig = X_data_sig.drop(columns="poistunut")
data_oldies_sig = data_oldies_sig.drop(columns="poistunut")
data_new_sig = data_new_sig.drop(columns="poistunut")
data_oldies = data_oldies.drop(columns="poistunut")
data_new = data_new.drop(columns="poistunut")

extra_clf = ExtraTreesClassifier(n_estimators=50, n_jobs=-1)
extra_clf = extra_clf.fit(X_data, y)
selected_x = SelectFromModel(extra_clf, threshold="1*mean", prefit=True)
feature_idx = selected_x.get_support()
feature_name = list(X_data.columns[feature_idx])
feature_names_list = pd.concat(
    [pd.DataFrame(data=feature_idx),
     pd.DataFrame(list(X_data.columns))],
    axis=1)
feature_names_sel = [
    "Age", "Price dif to market", "Traf ins paym", "Customer 10y+",
    "Number of Insurances"
]
skplt.estimators.plot_feature_importances(extra_clf, max_num_features=5)
selected_x_set = selected_x.transform(X_data)
selected_x_set_new = selected_x.transform(data_new)
selected_x_set_old = selected_x.transform(data_oldies)
# %%

models_old_sig = {}
models_new_sig = {}
models_whole_sig = {}
models_whole = {}
models_old_s = {}
models_new_s = {}
models_whole_s = {}

models_old_sig = func.run(data_oldies_sig, y_old)
models_new_sig = func.run(data_new_sig, y_new)
models_whole_sig = func.run(X_data_sig, y_sig)
models_whole = func.run(X_data, y)
models_old_s = func.run(selected_x_set_old, y_old)
models_new_s = func.run(selected_x_set_new, y_new)
models_whole_s = func.run(selected_x_set, y)

test = func.run_ann(X_data, y)

models_old_sig["ann_basic"] = func.run_ann(data_oldies_sig, y_old)
models_new_sig["ann_basic"] = func.run_ann(data_new_sig, y_new)
models_whole_sig["ann_basic"] = func.run_ann(X_data_sig, y_sig)
models_whole["ann_basic"] = func.run_ann(X_data, y)
models_old_s["ann_basic"] = func.run_ann(selected_x_set_old, y_old)
models_new_s["ann_basic"] = func.run_ann(selected_x_set_new, y_new)
models_whole_s["ann_basic"] = func.run_ann(selected_x_set, y)

models_old_sig["ann"] = func.run_ann_be(data_oldies_sig, y_old)
models_new_sig["ann"] = func.run_ann_be(data_new_sig, y_new)
models_whole_sig["ann"] = func.run_ann_be(X_data_sig, y_sig)
models_whole["ann"] = func.run_ann_be(X_data, y)
models_old_s["ann"] = func.run_ann_be(selected_x_set_old, y_old)
models_new_s["ann"] = func.run_ann_be(selected_x_set_new, y_new)
models_whole_s["ann"] = func.run_ann_be(selected_x_set, y)

x_train_old_sig, x_test_old_sig, y_train_old_sig, y_test_old_sig = train_test_split(
    data_oldies_sig, y_old, test_size=0.25, random_state=0)

x_train_new_sig, x_test_new_sig, y_train_new_sig, y_test_new_sig = train_test_split(
    data_new_sig, y_new, test_size=0.25, random_state=0)

x_train_sig, x_test_sig, y_train_sig, y_test_sig = train_test_split(
    X_data_sig, y_sig, test_size=0.25, random_state=0)

x_train, x_test, y_train, y_test = train_test_split(X_data,
                                                    y,
                                                    test_size=0.25,
                                                    random_state=0)

x_train_old_s, x_test_old_s, y_train_old_s, y_test_old_s = train_test_split(
    selected_x_set_old, y_old, test_size=0.25, random_state=0)

x_train_new_s, x_test_new_s, y_train_new_s, y_test_new_s = train_test_split(
    selected_x_set_new, y_new, test_size=0.25, random_state=0)

x_train_s, x_test_s, y_train_s, y_test_s = train_test_split(selected_x_set,
                                                            y,
                                                            test_size=0.25,
                                                            random_state=0)

# %%
model = "knn"
func.report(y_test_new_s, models_new_s[model].predict(x_test_new_s),
            "KNN_NC_S")
func.report_prob(y_test_new_s, models_new_s[model].predict_proba(x_test_new_s),
                 "KNN_NC_S")

func.report(y_test_old_s, models_old_s[model].predict(x_test_old_s), "RF_OC_S")
func.report_prob(y_test_old_s, models_old_s[model].predict_proba(x_test_old_s),
                 "RF_OC_S")

func.report(y_test_s, models_whole_s[model].predict(x_test_s), "AB_ALL_S")
func.report_prob(y_test_s, models_whole_s[model].predict_proba(x_test_s),
                 "AB_ALL_S")

func.report(y_test_old_sig, models_old_sig[model].predict(x_test_old_sig),
            "KNN_OC_G")
func.report_prob(y_test_old_sig,
                 models_old_sig[model].predict_proba(x_test_old_sig),
                 "KNN_OC_G")

func.report(y_test_new_sig, models_new_sig[model].predict(x_test_new_sig),
            "KNN_NC_G")
func.report_prob(y_test_new_sig,
                 models_new_sig[model].predict_proba(x_test_new_sig),
                 "KNN_NC_G")

func.report(y_test_sig, models_whole_sig[model].predict(x_test_sig),
            "KNN_ALL_G")
func.report_prob(y_test_sig, models_whole_sig[model].predict_proba(x_test_sig),
                 "KNN_ALL_G")

func.report(y_test, models_whole[model].predict(x_test), "AB_ALL")
func.report_prob(y_test, models_whole[model].predict_proba(x_test), "AB_ALL")

# %%
model = "ann"
func.report(y_test_new_s, models_new_s[model].predict(x_test_new_s).round(),
            "SVM_new_s")
func.report_prob_ann(y_test_new_s, models_new_s[model].predict(x_test_new_s),
                     "SVM_new_s")

func.report(y_test_old_s, models_old_s[model].predict(x_test_old_s).round(),
            "SVM_old_s")
func.report_prob_ann(y_test_old_s, models_old_s[model].predict(x_test_old_s),
                     "SVM_old_s")

func.report(y_test_s, models_whole_s[model].predict(x_test_s).round(), "SVM_s")
func.report_prob_ann(y_test_s, models_whole_s[model].predict(x_test_s),
                     "SVM_s")

func.report(y_test_old_sig,
            models_old_sig[model].predict(x_test_old_sig).round(), "AB_OC_G")
func.report_prob_ann(y_test_old_sig,
                     models_old_sig[model].predict(x_test_old_sig), "AB_OC_G")

func.report(y_test_new_sig,
            models_new_sig[model].predict(x_test_new_sig).round(), "ANN_NG_G")
func.report_prob_ann(y_test_new_sig,
                     models_new_sig[model].predict(x_test_new_sig), "ANN_NG_G")

func.report(y_test_sig, models_whole_sig[model].predict(x_test_sig).round(),
            "SVM_sig")
func.report_prob_ann(y_test_sig, models_whole_sig[model].predict(x_test_sig),
                     "SVM_sig")

func.report(y_test, models_whole[model].predict(x_test).round(), "SVM_all")
func.report_prob_ann(y_test, models_whole[model].predict(x_test), "SVM_all")

# %%
import talos

from tensorflow.keras.activations import relu, elu
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.losses import categorical_crossentropy, logcosh

p = {
    'lr': (0.1, 10, 10),
    'first_neuron': [4, 8, 16, 32, 64, 128],
    'batch_size': [100, 500, 1000],
    'epochs': [10, 20],
    'dropout': (0, 0.40, 10),
    'optimizer': [Adam, Nadam],
    'loss': ['binary_crossentropy'],
    'last_activation': ['sigmoid'],
    'weight_regulizer': [None]
}
# first we have to make sure to input data and params into the function
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.activations import sigmoid
from talos.utils import lr_normalizer


def iris_model(x_train, y_train, x_val, y_val, params):

    model = Sequential()
    model.add(
        Dense(params['first_neuron'],
              input_dim=x_train.shape[1],
              activation='relu'))

    model.add(Dropout(params['dropout']))
    model.add(Dense(1, activation=params['last_activation']))

    model.compile(optimizer=params['optimizer'](
        lr=lr_normalizer(params['lr'], params['optimizer'])),
                  loss=params['loss'],
                  metrics=['acc'])

    out = model.fit(x_train,
                    y_train,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    verbose=0,
                    validation_data=[x_val, y_val])

    return out, model


scan_object = talos.Scan(selected_x_set[:50000],
                         y.head(50000),
                         params=p,
                         model=iris_model,
                         experiment_name='iris',
                         fraction_limit=.001)
from talos import Evaluate, Analyze

a = Analyze(scan_object)
e = Evaluate(scan_object)
# %%
#from keras import models
#from keras import layers
import talos
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense
from keras.callbacks import TensorBoard
from talos.model.early_stopper import early_stopper

# track performance on tensorboard
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          batch_size=5000,
                          write_graph=False,
                          write_images=False)

# (1) Define dict of parameters to try
p = {
    'first_neuron': [10, 40, 160, 640, 1280],
    'hidden_neuron': [10, 40, 160],
    'hidden_layers': [0, 1, 2, 4],
    'batch_size': [1000],
    'optimizer': ['adam'],
    'kernel_initializer': ['uniform'],  #'normal'
    'epochs': [50],
    'dropout': [0.0, 0.25, 0.5],
    'last_activation': ['sigmoid']
}


# (2) create a function which constructs a compiled keras model object
def numerai_model(x_train, y_train, x_val, y_val, params):
    print(params)

    model = Sequential()

    ## initial layer
    model.add(
        Dense(params['first_neuron'],
              input_dim=x_train.shape[1],
              activation='relu',
              kernel_initializer=params['kernel_initializer']))
    model.add(Dropout(params['dropout']))

    ## hidden layers
    for i in range(params['hidden_layers']):
        print(f"adding layer {i+1}")
        model.add(
            Dense(params['hidden_neuron'],
                  activation='relu',
                  kernel_initializer=params['kernel_initializer']))
        model.add(Dropout(params['dropout']))

    ## final layer
    model.add(
        Dense(1,
              activation=params['last_activation'],
              kernel_initializer=params['kernel_initializer']))

    model.compile(loss='binary_crossentropy',
                  optimizer=params['optimizer'],
                  metrics=['acc'])

    history = model.fit(
        x_train,
        y_train,
        validation_data=[x_val, y_val],
        batch_size=params['batch_size'],
        epochs=params['epochs'],
        callbacks=[tensorboard,
                   early_stopper(params['epochs'], patience=10)],
        verbose=0)
    return history, model


# (3) Run a "Scan" using the params and function created above

t = talos.Scan(x=selected_x_set,
               y=y,
               model=numerai_model,
               params=p,
               experiment_name="hurdur")

from talos import Evaluate, Analyze

a_t = Analyze(t)
e_t = Evaluate(t)

# %%
import talos

autom8 = talos.autom8.AutoScan('binary', experiment_name="testtest")
aut = autom8.start(x=x_train_s[:10000],
                   y=y_train_s.head(10000).values,
                   x_val=x_test_s[:10000],
                   y_val=y_test_s.head(10000).values,
                   fraction_limit=0.0001)

# %%
grid_e = {}
amountTest = 20000
grid_e["b"] = func.ANN_e_b(selected_x_set[:amountTest], y.head(amountTest))
grid_e["alg"] = func.ANN_alg(selected_x_set[:amountTest], y.head(amountTest))
grid_e["mom"] = func.ANN_mom(selected_x_set[:amountTest], y.head(amountTest))
grid_e["w"] = func.ANN_w(selected_x_set[:amountTest], y.head(amountTest))
grid_e["fa"] = func.ANN_f_a(selected_x_set[:amountTest], y.head(amountTest))
grid_e["d"] = func.ANN_d(selected_x_set[:amountTest], y.head(amountTest))
grid_e["h"] = func.ANN_n_h(selected_x_set[:amountTest], y.head(amountTest))

print("Best: %f using %s" %
      (grid_e["b"].best_score_, grid_e["b"].best_params_))
print("Best: %f using %s" %
      (grid_e["alg"].best_score_, grid_e["alg"].best_params_))
print("Best: %f using %s" %
      (grid_e["mom"].best_score_, grid_e["mom"].best_params_))
print("Best: %f using %s" %
      (grid_e["w"].best_score_, grid_e["w"].best_params_))
print("Best: %f using %s" %
      (grid_e["fa"].best_score_, grid_e["fa"].best_params_))
print("Best: %f using %s" %
      (grid_e["d"].best_score_, grid_e["d"].best_params_))
print("Best: %f using %s" %
      (grid_e["h"].best_score_, grid_e["h"].best_params_))

