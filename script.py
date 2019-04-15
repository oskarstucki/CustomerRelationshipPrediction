from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def main():

    path = "../data/data_g.csv"
    data = pd.read_csv(path, encoding="latin-1", sep=";")
    # Preview the first 5 lines of the loaded data
    data.head()
    test = data.sort_values(["id", "timestamp"]).head(10000)
    test['kanava_myynti'] = test['kanava_myynti'].fillna(value='0')


def preproc(data):
    dates = ["timestamp", "viimeisin_sattpvm", "asiakkuus_alkpvm"]
    # Econding dates
    for i in dates:
        data[i] = pd.to_datetime(data[i])
        data[i].apply(lambda x: x.timestamp())

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


if __name__ == "__main__":
    main()
