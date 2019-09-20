import pandas as pd
from scipy.stats import ttest_ind
import statsmodels.api as sm
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score
import plot
from sklearn.ensemble import BaggingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier


import matplotlib.pyplot as plt

def report(y_test, y_pred, classes, model):
    print('Accuracy Score of '+model+' : ' + str(accuracy_score(y_test,y_pred)))
    print('Precision Score '+model+' : ' + str(precision_score(y_test,y_pred)))
    print('Recall Score '+model+': ' + str(recall_score(y_test,y_pred)))
    print('F1 Score'+model+' : ' + str(f1_score(y_test,y_pred)))
    plot.plot_confusion_matrix(y_test, y_pred, model)
    plt.show()
    

def run(data, y, columns):
    
    x_train, x_test, y_train, y_test = train_test_split(
        data, y, test_size=0.25, random_state=0)
    
    matrixes = {}
    #Dummy Classifier
    clf = DummyClassifier(strategy= 'stratified').fit(x_train,y_train)
    report(y_test, clf.predict(x_test).round(), columns, "Dummy")

    model = Sequential()
    model.add(Dense(20, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(8, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(rate=0.2))
    model.add(Dense(1, activation='sigmoid'))
    
    # Fitting the model for 150 epochs with 10 as batch size
    model.fit(x_train, y_train, epochs=20, batch_size=10)

    report(y_test, model.predict(x_test).round(), columns, "ANN")
    #report(validation_y, model.predict(validation_x).round(), columns, "ANN validation ")
    plt.plot(model.history['acc'])
    plt.plot(model.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    lr = LogisticRegression(max_iter=10000, solver="lbfgs")
    lr.fit(x_train, y_train)

    report(y_test, lr.predict(x_test), columns, "LR")
    #report(validation_y, lr.predict(validation_x), columns, "LR validation ")

    knn = KNeighborsClassifier(n_jobs=-1)
    knn.fit(x_train, y_train)
    
    report(y_test, knn.predict(x_test), columns, "KNN")
    #report(validation_y, knn.predict(validation_x), columns, "KNN validation ")

    svm = LinearSVC(max_iter=500)
    svm.fit(x_train, y_train)
    report(y_test, svm.predict(x_test), columns, "SVM")
    #report(validation_y, svm.predict(validation_x), columns, "SVM validation ")

    
    n_estimators = 10
    svm_b = BaggingClassifier(LinearSVC(max_iter=500), max_samples=1.0 / n_estimators, n_estimators=n_estimators, n_jobs=-1)
    svm_b.fit(x_train, y_train)
    report(y_test, svm_b.predict(x_test), columns, "SVM_B")
    #report(validation_y, svm_b.predict(validation_x), columns, "SVM_B validation ")

    
    rfc = RandomForestClassifier(n_estimators=1000, max_depth=1,
                                 random_state=0)
    rfc.fit(x_train, y_train)
    report(y_test, rfc.predict(x_test), columns, "RFC")
    #report(validation_y, rfc.predict(validation_x), columns, "RFC validation ")

    
    clf = DecisionTreeClassifier(max_depth=1).fit(x_train, y_train)
    report(y_test, clf.predict(x_test), columns, "CLF")
    #report(validation_y, clf.predict(validation_x), columns, "CLF validation ")
    
    ada = AdaBoostClassifier( DecisionTreeClassifier(max_depth=1)
                                    ,n_estimators=200).fit(x_train, y_train)
    report(y_test, ada.predict(x_test), columns, "ADA")
    #report(validation_y, ada.predict(validation_x), columns, "ADA validation ")
    
    mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 2), random_state=1).fit(x_train, y_train)
    report(y_test, mlp.predict(x_test), columns, "mlp")
    #report(validation_y, mlp.predict(validation_x), columns, "MLP validation ")

    
    return matrixes

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
    return model.transform(X)
    
    


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
