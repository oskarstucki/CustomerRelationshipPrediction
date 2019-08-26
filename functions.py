import pandas as pd
from scipy.stats import ttest_ind
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def xlsxToCsv():
    data_xls = pd.read_excel('../data/startData.xlsx')
    data_xls.to_csv('start.csv', encoding='utf-8', index=False)


def prepare(data):
    dates = ["timestamp", "viimeisin_sattpvm", "asiakkuus_alkpvm"]
    # Econding dates
    for i in dates:
        data[i] = pd.to_datetime(data[i])
        data[i].apply(lambda x: x.timestamp())

    # remove dates
    data = data.drop(columns=dates)

    # which are categorical
    cat_types = data.select_dtypes(exclude=["number", "bool_", "object_"])
    cols = data.columns
    num_cols = data._get_numeric_data().columns
    categorical = list(set(cols) - set(num_cols))

    # Encoding categorical data
    genderEncode = LabelEncoder()
    data["sukupuoli"] = genderEncode.fit_transform(
        data["sukupuoli"])
    languageEncode = LabelEncoder()
    data["kieli"] = languageEncode.fit_transform(
        data["kieli"])
    yryhtEncode = LabelEncoder()
    data["yryht_laatu"] = yryhtEncode.fit_transform(
        data["yryht_laatu"])
    locationEncode = LabelEncoder()
    data["alue"] = locationEncode.fit_transform(
        data["alue"])
    sellingLocEncode = LabelEncoder()
    data["kanava_myynti"] = sellingLocEncode.fit_transform(
        data["kanava_myynti"])
    moveEncode = LabelEncoder()
    data["viim_muutto"] = moveEncode.fit_transform(
        data["viim_muutto"])
    lastSoldEncode = LabelEncoder()
    data["viim_myynti"] = lastSoldEncode.fit_transform(
        data["viim_myynti"])
    return data


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
