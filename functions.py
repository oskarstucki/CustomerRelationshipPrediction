import pandas as pd
from scipy.stats import ttest_ind
import statsmodels.api as sm
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import plot
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import BaggingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from tensorflow.keras.layers import Dropout
from tensorflow.keras.constraints import MaxNorm
import preprocessing as pre
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import fbeta_score

import matplotlib.pyplot as plt


def processData(data):
    data = pre.prepare(data.copy())
    cat_data = data.select_dtypes(include=['object']).copy().astype('category')
    cat_data = list(
        cat_data.columns) + ["liikenne_voim", "kasko_voim", "fetu_voim"]
    bool_cols = [
        col for col in data
        if np.isin(data[col].dropna().unique(), [0, 1]).all()
    ]
    scaling = data.drop(bool_cols + cat_data, axis=1)
    others = data[[c for c in data.columns if c in bool_cols + cat_data]]

    y = data.loc[:, "poistunut"]
    drop = [
        "poistunut", "kasko_poistunut", "fetu_poistunut", "liikenne_poistunut"
    ]
    dataLabel, labels = pre.labeling(others.drop(columns=drop))

    mapper = DataFrameMapper([(scaling.columns, StandardScaler())])
    scaled_features = mapper.fit_transform(scaling.copy())
    dataProsSampleScaled = pd.DataFrame(scaled_features,
                                        index=scaling.index,
                                        columns=scaling.columns)
    frames = [dataLabel, dataProsSampleScaled]
    return pd.concat(frames, axis=1), y


def report(y_test, y_pred, model):
    print('Accuracy Score of ' + model + ' : ' +
          str(accuracy_score(y_test, y_pred)))
    print('Precision Score ' + model + ' : ' +
          str(precision_score(y_test, y_pred)))
    print('Recall Score ' + model + ': ' + str(recall_score(y_test, y_pred)))
    print('F1 Score' + model + ' : ' + str(f1_score(y_test, y_pred)))
    print('Fb Score' + model + ' : ' +
          str(fbeta_score(y_test, y_pred, beta=1.5)))
    print('AUC Score' + model + ' : ' + str(roc_auc_score(y_test, y_pred)))
    plot.plot_confusion_matrix(y_test, y_pred, model)
    plt.show()


def report_prob(y_test, y_pred, model):
    from sklearn.metrics import mean_squared_error
    import scikitplot as skplt
    print('MSE ' + model + ' : ' +
          str(mean_squared_error(y_test, y_pred[:, 1])))
    # skplt.metrics.plot_roc(y_test, y_pred, plot_micro=False, plot_macro=False)
    plot.plot_ROC_ann(y_test, y_pred[:, 1], model)
    plt.show()


def report_prob_ann(y_test, y_pred, model):
    from sklearn.metrics import mean_squared_error
    import scikitplot as skplt
    print('MSE ' + model + ' : ' + str(mean_squared_error(y_test, y_pred)))

    plot.plot_ROC_ann(y_test, y_pred.ravel(), model)
    plt.show()


def run_ann_be(data, y):
    from tensorflow.keras.optimizers import SGD
    validation = False

    x_train, x_test, y_train, y_test = train_test_split(data,
                                                        y,
                                                        test_size=0.25,
                                                        random_state=0)

    model = Sequential()
    model.add(
        Dense(x_train.shape[1] / 2,
              input_dim=x_train.shape[1],
              activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(x_train.shape[1] / 4, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = SGD(lr=0.3, momentum=0.9)
    model.compile(loss="binary_crossentropy",
                  optimizer=optimizer,
                  metrics=['acc'])
    history = model.fit(x_train,
                        y_train,
                        batch_size=500,
                        verbose=0,
                        validation_data=(x_test, y_test),
                        epochs=100,
                        use_multiprocessing=True,
                        workers=8)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    report(y_test, model.predict(x_test).round(), "ANN")
    if validation:
        report(y_test_org,
               model.predict(data_org_test).round(), "ANN validation ")
    return model


def run_ann(data, y):
    validation = False

    x_train, x_test, y_train, y_test = train_test_split(data,
                                                        y,
                                                        test_size=0.25,
                                                        random_state=0)

    model = Sequential()
    model.add(Dense(x_train.shape[1] / 2, input_dim=x_train.shape[1]))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss="binary_crossentropy",
                  optimizer='adam',
                  metrics=['acc'])
    history = model.fit(x_train,
                        y_train,
                        validation_data=(x_test, y_test),
                        epochs=15,
                        use_multiprocessing=True,
                        workers=8)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    report(y_test, model.predict(x_test).round(), "ANN")
    if validation:
        report(y_test_org,
               model.predict(data_org_test).round(), "ANN validation ")
    return model


def run(data, y):

    validation = False

    x_train, x_test, y_train, y_test = train_test_split(data,
                                                        y,
                                                        test_size=0.25,
                                                        random_state=0)

    models = {}
    #    # Dummy Classifier
    #    clf = DummyClassifier(strategy='stratified').fit(x_train, y_train)
    #    report(y_test, clf.predict(x_test).round(),  "Dummy")
    #
    #    class_weight = {1: 0.7,
    #                    0: 0.3}
    #    model = Sequential()
    #    model.add(Dense(
    #        20, input_dim=x_train.shape[1], activation='relu', kernel_constraint=MaxNorm(3)))
    #    model.add(Dropout(rate=0.2))
    #    model.add(Dense(8, activation='relu', kernel_constraint=MaxNorm(3)))
    #    model.add(Dropout(rate=0.2))
    #    model.add(Dense(1, activation='sigmoid'))
    #    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['acc'])
    #    history = model.fit(x_train, y_train, validation_data=(
    #        x_test, y_test), epochs=15, class_weight=class_weight,use_multiprocessing=True, workers=8)
    #    models["ann"]=model
    #
    #    plt.plot(history.history['acc'])
    #    plt.plot(history.history['val_acc'])
    #    plt.title('model accuracy')
    #    plt.ylabel('accuracy')
    #    plt.xlabel('epoch')
    #    plt.legend(['train', 'test'], loc='upper left')
    #    plt.show()
    #
    #    report(y_test, model.predict(x_test).round(),  "ANN")
    #    if validation:
    #        report(y_test_org, model.predict(
    #            data_org_test).round(), "ANN validation ")
    #

    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    models["lr"] = lr
    report(y_test, lr.predict(x_test), "LR")
    if validation:
        report(y_test_org, lr.predict(data_org_test), "LR validation ")


# knn_n = KNeighborsClassifier(n_jobs=-1)
    knn = KNeighborsClassifier(metric="manhattan",
                               n_neighbors=19,
                               weights="distance",
                               n_jobs=-1)
    knn.fit(x_train, y_train)
    # knn_n.fit(x_train,y_train)
    models["knn"] = knn
    #models["knn_n"]=knn_n
    report(y_test, knn.predict(x_test), "KNN")
    if validation:
        report(y_test_org, knn.predict(data_org_test), "KNN validation ")

    #feature_map_nystroem = Nystroem(gamma=.2, random_state=1, n_components=300)
    #data_transformed = feature_map_nystroem.fit_transform(x_train)
    svm = LinearSVC(max_iter=500)
    svm.fit(x_train, y_train)
    report(y_test, svm.predict(x_test), "SVM")
    if validation:
        report(y_test_org, svm.predict(data_org_test), "SVM validation ")
    models["svm"] = svm

    #    n_estimators = 10
    #    svm_b = BaggingClassifier(LinearSVC(
    #        max_iter=500), max_samples=1.0 / n_estimators, n_estimators=n_estimators, n_jobs=-1)
    #    svm_b.fit(x_train, y_train)
    #    report(y_test, svm_b.predict(x_test), "SVM_B")
    #    if validation:
    #        report(y_test_org, svm_b.predict(data_org_test), "SVM_B validation ")
    #    models["svm_b"]=svm_b
    #rfc_n = RandomForestClassifier(n_jobs=-1)
    rfc = RandomForestClassifier(n_estimators=1000,
                                 min_impurity_decrease=0,
                                 min_samples_leaf=2,
                                 max_features='auto',
                                 max_depth=70,
                                 bootstrap=False,
                                 random_state=0,
                                 n_jobs=-1)
    rfc.fit(x_train, y_train)
    #rfc_n.fit(x_train,y_train)
    report(y_test, rfc.predict(x_test), "RFC")
    if validation:
        report(y_test_org, rfc.predict(data_org_test), "RFC validation ")
    models["rfc"] = rfc
    # models["rfc_n"]=rfc_n

    #    dt = DecisionTreeClassifier(criterion="gini", max_depth=10, max_leaf_nodes=None,
    #                                min_samples_leaf=10, min_samples_split=10).fit(x_train, y_train)
    #    report(y_test, dt.predict(x_test), "DT")
    #    if validation:
    #        report(y_test_org, dt.predict(data_org_test), "DT validation ")
    #    models["dt"]=dt

    # ada_n = AdaBoostClassifier()
    ada = AdaBoostClassifier(DecisionTreeClassifier(criterion="gini",
                                                    max_depth=10,
                                                    max_leaf_nodes=None,
                                                    min_samples_leaf=10,
                                                    min_samples_split=10),
                             n_estimators=200).fit(x_train, y_train)
    #ada_n.fit(x_train,y_train)
    report(y_test, ada.predict(x_test), "ADA")
    if validation:
        report(y_test_org, ada.predict(data_org_test), "ADA validation ")
    models["ada"] = ada
    # models["ada_n"]=ada_n
    return models


def knn_test(x_train, y_train, x_test, y_test):
    k_range = range(1, 20)
    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        scores.append(knn.score(x_test, y_test))
    plt.figure()
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.scatter(k_range, scores)
    plt.xticks([0, 5, 10, 15, 20])


def ols(data, y):

    X2 = sm.add_constant(data)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    print(est2.summary())


def featureSelection(X, y):
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_selection import SelectFromModel
    clf = ExtraTreesClassifier(n_estimators=50, n_jobs=-1)
    clf = clf.fit(X, y)
    model = SelectFromModel(clf, prefit=True)
    feature_idx = model.get_support()
    feature_name = X.columns[feature_idx]
    return model.transform(X), feature_name


def getVar(searchList, ind):
    return [searchList[i] for i in ind]


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
        res = pd.concat(
            [res,
             pd.DataFrame([t[0], t[1]], columns=[xvar.name]).transpose()])
    res = res.rename(columns={0: 't_val', 1: 'p_val'})
    # Insert column of p_value check against threshold (default = 0.05)
    res['DEP_' + str(yvar.name)] = res['p_val'].apply(lambda x: x <= p_thre)

    # OUTPUT results: either TRUE/FALSE-list or full dataframe
    print("Statistical significance test@p_threshold=" + str(p_thre))
    if res.iloc[:, -1][res.iloc[:, -1] == True].empty:
        print("No statistically significant dependencies")
    elif full == 0:
        return res.iloc[:, -1][res.iloc[:, -1] == True]
    else:
        return res


# Utility function to report best scores
def report_search(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def ANN_e_b(X, Y):
    # Use scikit-learn to grid search the batch size and epochs
    import numpy
    from sklearn.model_selection import GridSearchCV
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasClassifier

    # Function to create model, required for KerasClassifier
    def create_model():
        # create model
        model = Sequential()
        model.add(Dense(20, input_dim=42, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    # create model
    model = KerasClassifier(build_fn=create_model, verbose=2)
    # define the grid search parameters
    batch_size = [200, 500, 1000]
    epochs = [10, 50, 100]
    param_grid = dict(batch_size=batch_size, epochs=epochs)
    grid = GridSearchCV(estimator=model,
                        param_grid=param_grid,
                        n_jobs=1,
                        verbose=2)
    grid_result = grid.fit(X, Y)
    # summarize results
    print("Best: %f using %s" %
          (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return grid


def ANN_alg(X, Y):
    # Use scikit-learn to grid search the batch size and epochs
    import numpy
    from sklearn.model_selection import GridSearchCV
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasClassifier

    # Function to create model, required for KerasClassifier
    def create_model(optimizer='adam'):
        # create model
        model = Sequential()
        model.add(Dense(20, input_dim=42, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        return model

    # create model
    model = KerasClassifier(build_fn=create_model,
                            epochs=50,
                            batch_size=1000,
                            verbose=2)
    # define the grid search parameters
    optimizer = [
        'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
    ]
    param_grid = dict(optimizer=optimizer)
    grid = GridSearchCV(estimator=model,
                        param_grid=param_grid,
                        n_jobs=1,
                        verbose=2)
    grid_result = grid.fit(X, Y)
    # summarize results
    print("Best: %f using %s" %
          (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return grid


def ANN_mom(X, Y):
    # Use scikit-learn to grid search the learning rate and momentum
    import numpy
    from sklearn.model_selection import GridSearchCV
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasClassifier
    from keras.optimizers import SGD

    # Function to create model, required for KerasClassifier
    def create_model(learn_rate=0.01, momentum=0):
        # create model
        model = Sequential()
        model.add(Dense(20, input_dim=42, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        # Compile model
        optimizer = SGD(lr=learn_rate, momentum=momentum)
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        return model

    # create model
    model = KerasClassifier(build_fn=create_model,
                            epochs=50,
                            batch_size=1000,
                            verbose=2)
    # define the grid search parameters
    learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
    momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    param_grid = dict(learn_rate=learn_rate, momentum=momentum)
    grid = GridSearchCV(estimator=model,
                        param_grid=param_grid,
                        n_jobs=1,
                        verbose=2)
    grid_result = grid.fit(X, Y)
    # summarize results
    print("Best: %f using %s" %
          (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return grid


def ANN_w(X, Y):
    # Use scikit-learn to grid search the weight initialization
    import numpy
    from sklearn.model_selection import GridSearchCV
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasClassifier

    # Function to create model, required for KerasClassifier
    def create_model(init_mode='uniform'):
        # create model
        model = Sequential()
        model.add(
            Dense(20,
                  input_dim=42,
                  kernel_initializer=init_mode,
                  activation='relu'))
        model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    # create model
    model = KerasClassifier(build_fn=create_model,
                            epochs=50,
                            batch_size=1000,
                            verbose=2)
    # define the grid search parameters
    init_mode = [
        'uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal',
        'glorot_uniform', 'he_normal', 'he_uniform'
    ]
    param_grid = dict(init_mode=init_mode)
    grid = GridSearchCV(estimator=model,
                        param_grid=param_grid,
                        n_jobs=1,
                        verbose=2)
    grid_result = grid.fit(X, Y)
    # summarize results
    print("Best: %f using %s" %
          (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return grid


def ANN_f_a(X, Y):
    # Use scikit-learn to grid search the activation function
    import numpy
    from sklearn.model_selection import GridSearchCV
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasClassifier

    # Function to create model, required for KerasClassifier
    def create_model(activation='relu'):
        # create model
        model = Sequential()
        model.add(
            Dense(20,
                  input_dim=42,
                  kernel_initializer='uniform',
                  activation=activation))
        model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    # create model
    model = KerasClassifier(build_fn=create_model,
                            epochs=50,
                            batch_size=1000,
                            verbose=2)
    # define the grid search parameters
    activation = [
        'softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid',
        'hard_sigmoid', 'linear'
    ]
    param_grid = dict(activation=activation)
    grid = GridSearchCV(estimator=model,
                        param_grid=param_grid,
                        n_jobs=1,
                        verbose=2)
    grid_result = grid.fit(X, Y)
    # summarize results
    print("Best: %f using %s" %
          (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return grid


def ANN_d(X, Y):
    # Use scikit-learn to grid search the dropout rate
    import numpy
    from sklearn.model_selection import GridSearchCV
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.wrappers.scikit_learn import KerasClassifier
    from keras.constraints import maxnorm

    # Function to create model, required for KerasClassifier
    def create_model(dropout_rate=0.0, weight_constraint=0):
        # create model
        model = Sequential()
        model.add(
            Dense(20,
                  input_dim=42,
                  kernel_initializer='uniform',
                  activation='linear',
                  kernel_constraint=maxnorm(weight_constraint)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    # create model
    model = KerasClassifier(build_fn=create_model,
                            epochs=50,
                            batch_size=1000,
                            verbose=2)
    # define the grid search parameters
    weight_constraint = [1, 2, 3, 4, 5]
    dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    param_grid = dict(dropout_rate=dropout_rate,
                      weight_constraint=weight_constraint)
    grid = GridSearchCV(estimator=model,
                        param_grid=param_grid,
                        n_jobs=1,
                        verbose=2)
    grid_result = grid.fit(X, Y)
    # summarize results
    print("Best: %f using %s" %
          (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return grid


def ANN_n_h(X, Y):
    # Use scikit-learn to grid search the number of neurons
    import numpy
    from sklearn.model_selection import GridSearchCV
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.wrappers.scikit_learn import KerasClassifier
    from keras.constraints import maxnorm

    # Function to create model, required for KerasClassifier
    def create_model(neurons=1):
        # create model
        model = Sequential()
        model.add(
            Dense(neurons,
                  input_dim=42,
                  kernel_initializer='uniform',
                  activation='linear',
                  kernel_constraint=maxnorm(4)))
        model.add(Dropout(0.2))
        model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    # create model
    model = KerasClassifier(build_fn=create_model,
                            epochs=50,
                            batch_size=1000,
                            verbose=2)
    # define the grid search parameters
    neurons = [1, 5, 10, 15, 20, 25, 30]
    param_grid = dict(neurons=neurons)
    grid = GridSearchCV(estimator=model,
                        param_grid=param_grid,
                        n_jobs=1,
                        verbose=2)
    grid_result = grid.fit(X, Y)
    # summarize results
    print("Best: %f using %s" %
          (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return grid