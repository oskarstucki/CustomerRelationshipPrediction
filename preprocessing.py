import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import utils
import datetime as dt


def timePreparing(data):
    dates = ["viimeisin_sattpvm", "asiakkuus_alkpvm"]

# remove dates and empty  values
    for i in dates:
        data[i] = pd.to_datetime(data[i])
    now = dt.datetime.now()

    data["days_from_sattpvm"] = data.agg({dates[0]: lambda x: (now-x).days})
    data["days_from_customer"] = data.agg({dates[1]: lambda x: (now-x).days})
    return data.drop(columns=dates)


def prepare(data):
    if utils.isIllogical(data):
        exit("Data makes no sense")

    data = timePreparing(data)

    # remove unnecessary fields
    unnecessaryFields = ["id", "timestamp",
                         "time_of_interestX", "validation_split"]

    data = data.drop(columns=unnecessaryFields)

    values = {'liikenne_poistunut': "0",
              'kasko_poistunut': "0", 'fetu_poistunut': "0", 'alue': "empty", 
              'kanava_myynti': "empty", "sukupuoli":"E", "kieli":"0", 
              "yryht_laatu": "yryht_puuttuu"}
    data = data.fillna(value=values)
    data = data.fillna("0")
    return data


def labeling(data):
    # which are categorical
    #cat_types = data.select_dtypes(exclude=["number", "bool_", "object_"])
    cols = data.columns
    num_cols = data._get_numeric_data().columns
    #categorical = list(set(cols) - set(num_cols))
    dict_labels = {}
    # Encoding categorical data
    genderEncode = LabelEncoder()
    data["sukupuoli"] = genderEncode.fit_transform(
        data["sukupuoli"])
    dict_labels["gender"] = dict(
        zip(genderEncode.classes_, genderEncode.transform(genderEncode.classes_)))

    languageEncode = LabelEncoder()
    data["kieli"] = languageEncode.fit_transform(
        data["kieli"])
    dict_labels["kieli"] = dict(
        zip(languageEncode.classes_, languageEncode.transform(languageEncode.classes_)))

    yryhtEncode = LabelEncoder()
    data["yryht_laatu"] = yryhtEncode.fit_transform(
        data["yryht_laatu"])
    dict_labels["yryht_laatu"] = dict(
        zip(yryhtEncode.classes_, yryhtEncode.transform(yryhtEncode.classes_)))

    locationEncode = LabelEncoder()
    data["alue"] = locationEncode.fit_transform(
        data["alue"])
    dict_labels["alue"] = dict(
        zip(locationEncode.classes_, locationEncode.transform(locationEncode.classes_)))

    sellingLocEncode = LabelEncoder()
    data["kanava_myynti"] = sellingLocEncode.fit_transform(
        data["kanava_myynti"])
    dict_labels["kanava_myynti"] = dict(
        zip(sellingLocEncode.classes_, sellingLocEncode.transform(sellingLocEncode.classes_)))
    moveEncode = LabelEncoder()
    data["viim_muutto"] = moveEncode.fit_transform(
        data["viim_muutto"])
    dict_labels["viim_muutto"] = dict(
        zip(moveEncode.classes_, moveEncode.transform(moveEncode.classes_)))
    lastSoldEncode = LabelEncoder()
    data["viim_myynti"] = lastSoldEncode.fit_transform(
        data["viim_myynti"])
    dict_labels["viim_myynti"] = dict(
        zip(lastSoldEncode.classes_, lastSoldEncode.transform(lastSoldEncode.classes_)))

    return data, dict_labels

def oversample(data):
    # Class count
    count_class_0, count_class_1 = data.poistunut.value_counts()
    
    # Divide by class
    df_class_0 = data[data['poistunut'] == 0]
    df_class_1 = data[data['poistunut'] == 1]
    df_class_1_over = df_class_1.sample(count_class_0, replace=True)
    df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)
    
    print('Random over-sampling:')
    print(df_test_over.poistunut.value_counts())
    
    df_test_over.poistunut.value_counts().plot(kind='bar', title='Count (target)');
    return df_test_over

def scaling(data):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    dict_labels["scaler"] = scaler.get_params(deep=True)
    return data, dict_labels
