from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
from babel.numbers import format_decimal, format_currency
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


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
    return df


#Train the model
def train_model(df):
    X = df.drop('Amount_in_rupees',axis='columns')
    Y = df.Amount_in_rupees
    dataset = pd.read_csv("Dataset/processed_house_prices.csv")

    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=10)

    model = RandomForestRegressor(random_state=42)
    print("******Started Model Training***********")
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    scoree = model.score(x_test,y_test)
    print(f'R-squared: {r2:.2f}')
    print(f'Mean Absolute Error (MAE): {mae:.2f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
    print(f'Accuracy of Model:{scoree:.2f}')
    joblib.dump(model, "housePriceModel.pkl")
    print("******Ended Model Training***********")
    return dataset, model









try:
    model = joblib.load("housePriceModel.pkl")
    dataset = pd.read_csv("Dataset/processed_house_prices.csv")
    print("Open model from saved one")
except:

    df = pre_processedData(df)
    dataset, model = train_model(df)

# dataset = pd.read_csv("Dataset/processed_house_prices.csv")

column_list = ["Price_in_rupees", "location", "Carpet_Area_in_sqft", "Floor", "Furnishing", "facing", "Bathroom"]
totalDict = {}

for i in column_list:
    col_value = dataset[i].values
    col_value = [col if col == "nan" else col for col in col_value]
    totalDict[i] = list(set(col_value))

for (key,value) in totalDict.items():
    if key == "Price_in_rupees":
        totalDict[key] = [float(x) for x in range(100000,int(max(totalDict[key])), 100000)] #format_decimal(x, locale='en_IN')
    if key == "location":
        totalDict[key] = [x.capitalize() for x in totalDict[key]]

    if key == "Carpet_Area_in_sqft":
        totalDict[key] = [float(x) for x in range(1000,int(max(totalDict[key])), 1000)] #(format_decimal(x, locale='en_IN'))

    if key == "Furnishing" or key == "facing":
        totalDict[key] = [x for x in totalDict[key] if "nan" not in str(x)]
    if key == "Bathroom":
        totalDict[key] = [float(x) for x in totalDict[key] if "nan" not in str(x)]


@app.route('/')
def load_UI():
    return render_template('index.html', totalDict=totalDict)


@app.route('/predict', methods=['POST','GET'])
def getUserInput():
    form_data = {
        "Price_in_rupees": request.form["Price_in_rupees"],
        "location": request.form["location"], 
        "Carpet_Area_in_sqft": request.form["Carpet_Area_in_sqft"],
        "Floor": request.form["Floor"], 
        "Furnishing": request.form["Furnishing"],
        "facing": request.form["facing"], 
        "Bathroom": request.form["Bathroom"],
    }

    # Convert to DataFrame
    df_data = pd.DataFrame([form_data])
    
    def label_encode_multiple(df, columns):
        le = LabelEncoder()
        for column in columns:
            df[column] = le.fit_transform(df[column])
        return df

    label_encode_columns = ['Floor', 'Furnishing', 'facing', 'location']
    df_data = label_encode_multiple(df_data, label_encode_columns)
    

    df_data["Price_in_rupees"] = pd.to_numeric(df_data["Price_in_rupees"], errors='coerce')
    df_data["Carpet_Area_in_sqft"] = pd.to_numeric(df_data["Carpet_Area_in_sqft"], errors='coerce')
    df_data["Bathroom"] = pd.to_numeric(df_data["Bathroom"], errors='coerce')
    print(df_data.info())

    model = joblib.load("housePriceModel.pkl")

    prediction = model.predict(df_data)
    formatted_prediction = [format_currency(value, 'INR', locale='en_IN') for value in prediction]
    formatted_prediction = str(formatted_prediction[0])

    return render_template('index.html', formatted_prediction= formatted_prediction)
    # return {"Predicted Amount is": formatted_prediction}



