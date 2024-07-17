import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from sklearn.linear_model import LinearRegression

# Load the CSV files and convert relevant columns to lowercase
batsman_data_path = r'C:\Users\Subham.Bhuyan\Desktop\Projects\FINAL\all_season_batting_card.csv'
batsman_data = pd.read_csv(batsman_data_path)

for data in [batsman_data]:
    data.replace("-", np.nan, inplace=True)
    data.fillna(0, inplace=True)
    # Convert relevant columns to lowercase
    data['fullName'] = data['fullName'].str.lower()

batsman_numeric_columns = ['runs', 'ballsFaced', 'fours', 'sixes', 'strikeRate']
batsman_data[batsman_numeric_columns] = batsman_data[batsman_numeric_columns].apply(pd.to_numeric)

def predict_future_scores(player_name, data):
    player_name = player_name.lower()  # Convert input to lowercase
    data['fullName'] = data['fullName'].str.lower()  # Convert CSV data to lowercase

    player_data = data[data['fullName'] == player_name]

    if player_data.empty:
        return None, f"No data found for player: {player_name}"

    # Convert the season column to datetime, assuming it's just the year
    player_data['season'] = pd.to_datetime(player_data['season'], format='%Y', errors='coerce')

    # Filter data for seasons 2008-2020
    training_data = player_data[(player_data['season'].dt.year >= 2008) & (player_data['season'].dt.year <= 2020)]

    # Aggregate the data to get total runs per season
    season_data = training_data.groupby(training_data['season'].dt.year)['runs'].sum().reset_index()

    # Rename columns for clarity
    season_data.columns = ['season', 'total_runs']

    # Feature engineering
    X_train = season_data[['season']]
    y_train = season_data['total_runs']

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict future runs for the next 3 seasons (2021, 2022, 2023)
    future_seasons = np.array([2021, 2022, 2023]).reshape(-1, 1)
    future_runs = model.predict(future_seasons)

    # Convert predicted runs to integers
    future_runs = future_runs.astype(int)

    # Create a DataFrame for the predictions
    predictions = pd.DataFrame({'season': future_seasons.flatten(), 'predicted_runs': future_runs})

    # Get actual runs for seasons 2021, 2022, 2023
    actual_data = player_data[player_data['season'].dt.year.isin([2021, 2022, 2023])]
    actual_runs = actual_data.groupby(actual_data['season'].dt.year)['runs'].sum().reset_index()

    # Merge predictions with actual runs
    predictions = predictions.merge(actual_runs, left_on='season', right_on='season', how='left')
    predictions.rename(columns={'runs': 'runs_actual'}, inplace=True)  # Rename 'runs' column from actual_runs

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, color='blue', label='Training Data')
    plt.scatter(actual_runs['season'], actual_runs['runs'], color='green', label='Actual Runs')
    plt.plot(future_seasons, future_runs, color='red', linestyle='dashed', marker='o', label='Predicted Runs')
    plt.xlabel('Season')
    plt.ylabel('Total Runs')
    plt.title(f'Performance Prediction for {player_name}')
    plt.legend()
    plt.grid(True)

    # Save the plot as a PNG image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    plt.show()  # Display plot in command line

    return predictions.to_dict(orient='records'), plot_url

if __name__ == '__main__':
    while True:
        player_name = input("Enter player's full name (or 'exit' to quit): ")
        if player_name.lower() == 'exit':
            break
        
        predictions, plot_url = predict_future_scores(player_name, batsman_data)
        if predictions:
            print("\nPredictions:")
            for prediction in predictions:
                print(f"Season: {prediction['season']}, Predicted Runs: {prediction['predicted_runs']}, Actual Runs: {prediction['runs_actual']}")

            print(f"\nPlot URL: {plot_url}")
        else:
            print(f"\nNo data found for player: {player_name}")
