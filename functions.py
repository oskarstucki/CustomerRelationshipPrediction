import pandas as pd


def xlsxToCsv():
    data_xls = pd.read_excel('../data/startData.xlsx')
    data_xls.to_csv('start.csv', encoding='utf-8', index=False)
