# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


# %%
def rf_grid(data, y):
    rf_grid = {
        'bootstrap': [True, False],
        'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        'n_estimators':
        [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    }

    rfc_g = RandomForestClassifier()

    rf_random = RandomizedSearchCV(estimator=rfc_g,
                                   param_distributions=rf_grid,
                                   n_iter=100,
                                   cv=3,
                                   verbose=2,
                                   random_state=42,
                                   n_jobs=-1)
    rf_random.fit(data, y)
    func.report(y_test, rf_random.predict(x_test), "random RF")

    def evaluate(model, test_features, test_labels):

        predictions = model.predict(test_features)
        errors = abs(predictions - test_labels)
        mape = 100 * np.mean(errors / test_labels)
        accuracy = 100 - mape
        print('Model Performance')
        print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
        print('Accuracy = {:0.2f}%.'.format(accuracy))

        return accuracy


# %%
def dt_grid(data):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        "criterion": ["gini", "entropy"],
        "min_samples_split": [2, 10, 20],
        'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        "min_samples_leaf": [1, 5, 10],
        "max_leaf_nodes": [None, 5, 10, 20],
    }

    dt_grid = DecisionTreeClassifier()
    n_iter_search = 20
    dt_search = GridSearchCV(dt_grid,
                             param_grid=param_grid,
                             scoring="accuracy",
                             cv=5,
                             n_jobs=-1)
    dt_search.fit(x_train, y_train)
    func.report(y_test, dt_search.predict(x_test), "DT with search")


# %%
def knn_grid():
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier

    knn_params = {
        'n_neighbors': [3, 5, 11, 19],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    knn_grid = GridSearchCV(KNeighborsClassifier(),
                            param_grid=knn_params,
                            scoring="accuracy",
                            cv=5,
                            verbose=1,
                            n_jobs=-1)
    knn_grid.fit(x_train[:50000], y_train[:50000])
    func.report(y_test, knn_grid.predict(x_test), "KNN with search")


#%%
def lr_grid():
    from sklearn import linear_model
    import numpy as np
    from sklearn.model_selection import GridSearchCV

    # Create regularization penalty space
    penalty = ["l2", "l1"]
    solver = ["liblinear", "saga"]
    # Create regularization hyperparameter space
    C = np.logspace(0, 4, 10)
    logistic = linear_model.LogisticRegression()
    # Create hyperparameter options
    hyperparameters = dict(C=C, penalty=penalty, solver=solver)
    clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=1, n_jobs=-1)
    best_model = clf.fit(selected_x_set, y)
    print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
    print('Best C:', best_model.best_estimator_.get_params()['C'])


#%%
def L_svc__grid():
    from sklearn.svm import LinearSVC
    import numpy as np
    from sklearn.model_selection import GridSearchCV

    # Create regularization penalty space
    penalty = ["l2"]
    loss = ["hinge", "squared_hinge"]
    # Create regularization hyperparameter space
    C = np.logspace(0, 4, 10)
    logistic = LinearSVC()
    # Create hyperparameter options
    hyperparameters = dict(C=C, penalty=penalty, loss=loss)
    clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=1, n_jobs=-1)
    best_model = clf.fit(selected_x_set[:50000], y.head(50000))
    print(best_model.best_estimator_.get_params())
    print(best_model.best_score_)


#%%
def svc__grid():
    from sklearn.svm import SVC
    import numpy as np
    from sklearn.model_selection import GridSearchCV

    # Create regularization penalty space
    param_grid = {
        'C': [1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.001, 0.0001],
        'kernel': ['linear', 'rbf']
    }
    logistic = SVC()

    clf = GridSearchCV(logistic, param_grid, cv=5, verbose=1, n_jobs=-1)
    best_model_svc = clf.fit(selected_x_set[:50000], y.head(50000))
    print(best_model_svc.best_estimator_.get_params())
    print(best_model_svc.best_score_)


# %%
#
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
#
# config = tf.ConfigProto(
#    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
#    # device_count = {'GPU': 1}
# )
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)
# set_session(session)
#
# amountTest=50000
# func.ANN_e_b(X_data.head(amountTest),y.head(amountTest))
# func.ANN_alg(X_data.head(amountTest),y.head(amountTest))
# func.ANN_mom(X_data.head(amountTest),y.head(amountTest))
# func.ANN_w(X_data.head(amountTest),y.head(amountTest))
# func.ANN_f_a(X_data.head(amountTest),y.head(amountTest))
# func.ANN_d(X_data.head(amountTest),y.head(amountTest))
# func.ANN_n_h(X_data.head(amountTest),y.head(amountTest))

# %%