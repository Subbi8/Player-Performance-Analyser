import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from flask import Flask, render_template, request, jsonify
import io
import base64
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)

base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full paths to the CSV files
batsman_data_path = os.path.join(base_dir, 'data', 'all_season_batting_card.csv')
bowler_data_path = os.path.join(base_dir, 'data', 'all_season_bowling_card.csv')

# Load the CSV files
batsman_data = pd.read_csv(batsman_data_path)
bowler_data = pd.read_csv(bowler_data_path)

# Handling missing data and non-numeric values for both datasets
for data in [batsman_data, bowler_data]:
    data.replace("-", np.nan, inplace=True)
    data.fillna(0, inplace=True)

# Ensure numeric columns are actually numeric for both datasets
batsman_numeric_columns = ['runs', 'ballsFaced', 'fours', 'sixes', 'strikeRate']
bowler_numeric_columns = ['maidens', 'conceded', 'wickets', 'economyRate', 'dots']
batsman_data[batsman_numeric_columns] = batsman_data[batsman_numeric_columns].apply(pd.to_numeric)
bowler_data[bowler_numeric_columns] = bowler_data[bowler_numeric_columns].apply(pd.to_numeric)

# Function to get all player names
def get_all_players():
    batsman_names = batsman_data['fullName'].tolist()
    bowler_names = bowler_data['fullName'].tolist()
    all_player_names = list(set(batsman_names + bowler_names))
    return all_player_names

# Function to create a histogram
def create_histogram(player_data, player_name):
    categories = list(player_data.keys())
    values = list(player_data.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bar_width = 0.5
    bar_positions = np.arange(len(categories))
    
    ax.bar(bar_positions, values, bar_width, alpha=0.7, color='b', label=player_name)
    
    ax.set_xlabel('Categories', fontsize=14)
    ax.set_ylabel('Values', fontsize=14)
    ax.set_title(f'Performance Histogram: {player_name}', fontsize=20, color='blue')
    
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=12)
    
    for i, value in enumerate(values):
        ax.text(i, value + 0.5, str(value), ha='center', fontsize=12)
    
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

# Function to create a line plot
def create_line_plot(player_data, player_name):
    fig, ax = plt.subplots(figsize=(8, 8))
    player_data.plot(ax=ax)
    plt.title(f'Line Plot for {player_name}', size=20)
    plt.ylabel('Values')
    plt.xlabel('Match Index')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

# Function to predict future scores
def predict_future_scores(player_name, data):
    player_data = data[data['fullName'] == player_name]

    if player_data.empty:
        return None, f"No data found for player: {player_name}"

    # Convert the season column to datetime, assuming it's just the year
    player_data['season'] = pd.to_datetime(player_data['season'], format='%Y', errors='coerce')

    # Drop rows where the season conversion failed
    player_data = player_data.dropna(subset=['season'])

    # Aggregate the data to get total runs per season
    season_data = player_data.groupby(player_data['season'].dt.year)['runs'].sum().reset_index()

    # Rename columns for clarity
    season_data.columns = ['season', 'total_runs']

    # Feature engineering
    X = season_data[['season']]
    y = season_data['total_runs']

    # Train the model
    model = LinearRegression()
    model.fit(X, y)

    # Predict future runs for the next 2, 4, 6, 8, and 10 seasons
    future_seasons = np.array([2024, 2025, 2026]).reshape(-1, 1)
    future_runs = model.predict(future_seasons)

    # Convert predicted runs to integers
    future_runs = future_runs.astype(int)

    # Create a DataFrame for the predictions
    predictions = pd.DataFrame({'season': future_seasons.flatten(), 'predicted_runs': future_runs})

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Actual runs')
    plt.plot(future_seasons, future_runs, color='red', linestyle='dashed', marker='o', label='Predicted runs')
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

    return predictions.to_dict(orient='records'), plot_url

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_player', methods=['POST'])
def check_player():
    player_name = request.form.get('playerName')
    player_exists = player_name in get_all_players()
    return jsonify({'exists': player_exists})

# Route for analysis page
@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    radar_img = None
    barplot_img = None
    lineplot_img = None
    piechart_img = None
    error_message = None
    player_name = None
    
    if request.method == 'POST':
        player_type = request.form.get('playerType', None)
        player_name = request.form.get('playerName', None)
        
        if player_type and player_name:
            if player_type == 'batsman':
                player_data = batsman_data[batsman_data['fullName'] == player_name]
                numeric_columns = batsman_numeric_columns
            elif player_type == 'bowler':
                player_data = bowler_data[bowler_data['fullName'] == player_name]
                numeric_columns = bowler_numeric_columns
            else:
                error_message = "Invalid player type selected"
                player_data = None
            
            if player_data is not None and not player_data.empty:
                aggregated_data = player_data[numeric_columns].sum().to_dict()
                radar_img = create_histogram(aggregated_data, player_name)
                barplot_img = create_line_plot(player_data[numeric_columns], player_name)
                
            else:
                error_message = f"No data found for player: {player_name}"
        else:
            error_message = "Player type or name not provided"
    
    return render_template('analysis.html', radar_img=radar_img, barplot_img=barplot_img, lineplot_img=lineplot_img, piechart_img=piechart_img, player_name=player_name, error_message=error_message)

# Route for predictor page
@app.route('/predictor', methods=['GET', 'POST'])
def predictor():
    if request.method == 'POST':
        player_name = request.form['playerName']
        predictions, plot_url = predict_future_scores(player_name, batsman_data)
        if predictions is None:
            return render_template('predictor.html', error_message=plot_url)
        else:
            return render_template('predictor.html', player_name=player_name, predictions=predictions, lineplot_img=plot_url)
    return render_template('predictor.html')

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    query = request.args.get('query', '')
    matches = [name for name in get_all_players() if name.lower().startswith(query.lower())]
    return jsonify(matches)

if __name__ == '__main__':
    app.run(debug=True)
