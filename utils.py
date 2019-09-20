
import pandas as pd


def xlsxToCsv():
    data_xls = pd.read_excel('../data/startData.xlsx')
    data_xls.to_csv('start.csv', encoding='utf-8', index=False)


def isIllogical(data):
    if data.loc[(data["liikenne_voim"] == 0) & (data["liikenne_poistunut"] == 1), "id"].size != 0:
        return True
    if data.loc[(data["kasko_voim"] == 0) & (data["kasko_poistunut"] == 1), "id"].size != 0:
        return True
    if data.loc[(data["fetu_voim"] == 0) & (data["fetu_poistunut"] == 1), "id"].size != 0:
        return True
    return False
