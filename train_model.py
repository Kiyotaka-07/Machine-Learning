import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")

size_to_avg_weight = {
    'TWO-SEATER': 1100,
    'MINICOMPACT': 1150,
    'SUBCOMPACT': 1250,
    'COMPACT': 1350,
    'STATION WAGON - SMALL': 1400,
    'MID-SIZE': 1600,
    'STATION WAGON - MID-SIZE': 1650,
    'SUV - SMALL': 1800,
    'SUV - STANDARD': 2200,
    'MINIVAN': 2200,
    'PICKUP TRUCK - SMALL': 2300,
    'PICKUP TRUCK - STANDARD': 2600,
    'VAN - PASSENGER': 2500,
    'VAN - CARGO': 2700,
    'SPECIAL PURPOSE VEHICLE': 2800
}

df['vehicle_class_num'] = df['Vehicle Class'].map(size_to_avg_weight)

df['trans_type'] = df['Transmission'].str.extract(r'([A-Z]+)')

df = pd.get_dummies(
    df,
    columns=['trans_type'],
    prefix='trans',
    dtype=int
)

df = pd.get_dummies(
    df,
    columns=['Fuel Type'],
    prefix='fuel',
    dtype=int
)

new_df = df[['Engine Size(L)',
             'Cylinders',
             'vehicle_class_num',
             'trans_A',
             'trans_AM',
             'trans_AS',
             'trans_AV',
             'trans_M',
             'fuel_D',
             'fuel_E',
             'fuel_N',
             'fuel_X',
             'fuel_Z',
             'CO2 Emissions(g/km)'
             ]]

new_df = new_df.dropna()

X = new_df.drop(columns=['CO2 Emissions(g/km)'])
y = new_df['CO2 Emissions(g/km)']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# save model
pickle.dump(model, open("model.pkl", "wb"))

# save feature columns
pickle.dump(X.columns.tolist(), open("feature_columns.pkl", "wb"))

print("Model saved!")