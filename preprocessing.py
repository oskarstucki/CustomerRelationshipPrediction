import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import utils
import datetime as dt
import seaborn as sns


def get_customer_dist(data):
    dates = ["viimeisin_sattpvm", "asiakkuus_alkpvm", "timestamp"]

    for i in dates:
        data[i] = pd.to_datetime(data[i])

    def calculate_days(vec):
        time = vec[0]
        time_interest = vec[1]
        return (time_interest - time).days

    data["asiakkuus_kesto"] = data[[dates[1], dates[2]]].apply(calculate_days,
                                                               axis=1)
    sns.distplot(data["asiakkuus_kesto"])
    return data["asiakkuus_kesto"]


def timePreparing(data):
    dates = ["viimeisin_sattpvm", "asiakkuus_alkpvm", "timestamp"]

    # remove dates and empty  values
    for i in dates:
        data[i] = pd.to_datetime(data[i])

    def make_dates(vec):
        time = vec[0]
        time_interest = vec[1]
        if (time_interest - time).days <= 180:
            matchVar = "New"
        elif (time_interest - time).days <= 360 and (time_interest -
                                                     time).days >= 180:
            matchVar = "1y"
        elif (time_interest - time).days <= 360 * 2 and (time_interest -
                                                         time).days >= 360:
            matchVar = "2y"
        elif (time_interest - time).days <= 360 * 3 and (time_interest -
                                                         time).days >= 360 * 2:
            matchVar = "3y"
        elif (time_interest - time).days <= 360 * 4 and (time_interest -
                                                         time).days >= 360 * 3:
            matchVar = "4y"
        elif (time_interest - time).days <= 360 * 5 and (time_interest -
                                                         time).days >= 360 * 4:
            matchVar = "5y"
        elif (time_interest - time).days <= 360 * 6 and (time_interest -
                                                         time).days >= 360 * 5:
            matchVar = "6y"
        elif (time_interest - time).days <= 360 * 7 and (time_interest -
                                                         time).days >= 360 * 6:
            matchVar = "7y"
        elif (time_interest - time).days <= 360 * 8 and (time_interest -
                                                         time).days >= 360 * 7:
            matchVar = "8y"
        elif (time_interest - time).days <= 360 * 9 and (time_interest -
                                                         time).days >= 360 * 8:
            matchVar = "9y"
        else:
            matchVar = "10y+"
        return matchVar

    data['days_from_sattpvm'] = data[[dates[0], dates[2]]].apply(make_dates,
                                                                 axis=1)
    data["asiakkuus_kesto"] = data[[dates[1], dates[2]]].apply(make_dates,
                                                               axis=1)

    return data.drop(columns=dates)


def prepare(data):
    if utils.isIllogical(data):
        exit("Data makes no sense")

    data = timePreparing(data)

    # remove unnecessary fields
    unnecessaryFields = [
        "id", "time_of_interestX", "validation_split", "kasko_poistunut",
        "fetu_poistunut", "liikenne_poistunut"
    ]

    data = data.drop(columns=unnecessaryFields)

    values = {
        'liikenne_poistunut': "3",
        'kasko_poistunut': "3",
        'fetu_poistunut': "3",
        'alue': "empty",
        'kanava_myynti': "empty",
        "sukupuoli": "E",
        "kieli": "E",
        "yryht_laatu": "yryht_puuttuu"
    }
    data = data.fillna(value=values)
    data = data.fillna("0")
    return data


def labeling(data):
    # which are categorical
    # cat_types = data.select_dtypes(exclude=["number", "bool_", "object_"])
    cols = data.columns
    num_cols = data._get_numeric_data().columns
    # categorical = list(set(cols) - set(num_cols))
    dict_labels = {}
    # Encoding categorical data
    genderEncode = LabelEncoder()
    data["sukupuoli"] = genderEncode.fit_transform(data["sukupuoli"])
    dict_labels["gender"] = dict(
        zip(genderEncode.classes_,
            genderEncode.transform(genderEncode.classes_)))

    languageEncode = LabelEncoder()
    data["kieli"] = languageEncode.fit_transform(data["kieli"])
    dict_labels["kieli"] = dict(
        zip(languageEncode.classes_,
            languageEncode.transform(languageEncode.classes_)))

    yryhtEncode = LabelEncoder()
    data["yryht_laatu"] = yryhtEncode.fit_transform(data["yryht_laatu"])
    dict_labels["yryht_laatu"] = dict(
        zip(yryhtEncode.classes_, yryhtEncode.transform(yryhtEncode.classes_)))

    locationEncode = LabelEncoder()
    data["alue"] = locationEncode.fit_transform(data["alue"])
    dict_labels["alue"] = dict(
        zip(locationEncode.classes_,
            locationEncode.transform(locationEncode.classes_)))

    sellingLocEncode = LabelEncoder()
    data["kanava_myynti"] = sellingLocEncode.fit_transform(
        data["kanava_myynti"])
    dict_labels["kanava_myynti"] = dict(
        zip(sellingLocEncode.classes_,
            sellingLocEncode.transform(sellingLocEncode.classes_)))
    moveEncode = LabelEncoder()
    data["viim_muutto"] = moveEncode.fit_transform(data["viim_muutto"])
    dict_labels["viim_muutto"] = dict(
        zip(moveEncode.classes_, moveEncode.transform(moveEncode.classes_)))
    lastSoldEncode = LabelEncoder()
    data["viim_myynti"] = lastSoldEncode.fit_transform(data["viim_myynti"])
    dict_labels["viim_myynti"] = dict(
        zip(lastSoldEncode.classes_,
            lastSoldEncode.transform(lastSoldEncode.classes_)))

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

    df_test_over.poistunut.value_counts().plot(kind='bar',
                                               title='Count (target)')
    return df_test_over


def scaling(data):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    dict_labels = dict()
    dict_labels["scaler"] = scaler.get_params(deep=True)
    return data, dict_labels
