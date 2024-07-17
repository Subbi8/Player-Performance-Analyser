import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

batsman_data = r'C:\Users\Subham.Bhuyan\Desktop\Projects\FINAL\all_season_batting_card.csv'

for data in batsman_data:
    data.replace('-', np.nan , inpalce=True)
    data.fillna(0,inplace=True)
    data['fullName']= data['fullName'].str.lower()

batsman_numeric_columns = ['runs', 'ballsFaced', 'fours', 'sixes', 'strikeRate']
batsman_data[batsman_numeric_columns]= batsman_data[batsman_numeric_columns].apply(pd.to_numeric)

def predict_score(player_name , data):
    data["fullName"]= data["fullName"].str.lower()
    player_name=player_name.lower()

    player_data = data[data["fullName"]==player_name]
    if player_data.empty:
        return None, f"No data found for player: {player_name}"
    player_data["season"]= pd.to_datetime(player_data["season"] ,format='%Y' , errors= 'coerce' )
    training_data = player_data[(player_data['season'].dt.year >= 2008) & (player_data['season'].dt.year <= 2020)]

    season_data = training_data.groupby(training_data['season'].dt.year)['runs'].sum().reset_index()

    season_data.columns = ['season', 'total_runs']
    X_train = season_data['season']
    Y_train = season_data['total_runs']
    model = LinearRegression()
    model.fit(X_train , Y_train)
    future_season = np.array([2021 , 2022 , 2023]).reshape(1,-1)
    future_runs = model.predict(future_season)
    future_runs = future_runs.astype(int)
    predictions = pd.DataFrame({"season":future_season.flatten() , "predicted runs ":future_runs})                         


