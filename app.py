from flask import Flask, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
from babel.numbers import format_decimal


app = Flask(__name__)


df = pd.read_csv("Dataset/house_prices.csv")


def calculate_carpetArea():
    total = 0
    count = 0
    for i in df["Carpet Area"]:
        try:
            if i.lower() != "nan":
                total += int(i.split(" ")[0])
                count += 1
        except:
            pass
    return total, count

def convert_rupees(amount_str):
    try:
        parts=amount_str.split()
        amount=float(parts[0])

        if len(parts)>1:
            unit=parts[1].strip()
            if unit=='Lac':
                amount*=100000
            elif unit=='Cr':
                amount*=10000000
        return amount
    except(ValueError,IndexError):
        return None
    
def label_encode_multiple(df, columns):
    le = LabelEncoder()
    for column in columns:
        df[column] = le.fit_transform(df[column])
    return df


def pre_processedData(df):
    df = df.drop(['Plot Area', 'Car Parking','Dimensions','Balcony','Super Area','Title','Society','Description','Status','overlooking','Ownership','Transaction'],axis='columns')
    df["Price (in rupees)"] = df["Price (in rupees)"].transform(lambda x: x.fillna(x.mean()))
    ca, count = calculate_carpetArea()
    df['Carpet Area'] = df['Carpet Area'].transform(lambda x: x.fillna(float(ca/count)))
    df["Carpet Area"] = df["Carpet Area"].astype(str).transform(lambda x: x.split(" ")[0])
    df["Carpet Area"] = df["Carpet Area"].astype(float)
    df['Amount(in rupees)']= df['Amount(in rupees)'].apply(convert_rupees)
    df.rename(columns={'Amount(in rupees)': 'Amount_in_rupees', 'Price (in rupees)': 'Price_in_rupees', 'Carpet Area': 'Carpet_Area_in_sqft'}, inplace=True)
    df["Bathroom"] = pd.to_numeric(df["Bathroom"], errors='coerce')
    label_encode_columns = ['Floor', 'Furnishing', 'facing', 'location']
    df = label_encode_multiple(df, label_encode_columns)
    df["Amount_in_rupees"] = df["Amount_in_rupees"].transform(lambda x: x.fillna(x.mean()))
    df["Bathroom"] = df["Bathroom"].transform(lambda x: x.fillna(x.mean()))
    df.drop("Index",axis='columns',inplace=True)


# pre_processedData(df["Price (in rupees)"])

dataset = pd.read_csv("Dataset/processed_house_prices.csv")

column_list = ["Price_in_rupees", "location", "Carpet_Area_in_sqft", "Floor", "Furnishing", "facing", "Bathroom"]
totalDict = {}

for i in column_list:
    col_value = dataset[i].values
    col_value = [col if col == "nan" else col for col in col_value]
    totalDict[i] = list(set(col_value))

for (key,value) in totalDict.items():
    if key == "Price_in_rupees":
        print(max(totalDict[key]))
        totalDict[key] = [format_decimal(x, locale='en_IN') for x in range(100000,int(max(totalDict[key])), 100000)]
    if key == "location":
        totalDict[key] = [x.capitalize() for x in totalDict[key]]

    if key == "Carpet_Area_in_sqft":
        totalDict[key] = [(format_decimal(x, locale='en_IN')) for x in range(1000,int(max(totalDict[key])), 1000)]

    if key == "Furnishing" or key == "facing":
        totalDict[key] = [x for x in totalDict[key] if "nan" not in str(x)]
    if key == "Bathroom":
        totalDict[key] = [int(x) for x in totalDict[key] if "nan" not in str(x)]


@app.route('/')
def hello():
    return render_template('index.html', totalDict=totalDict)




