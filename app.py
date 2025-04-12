import plotly.express as px
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
import joblib

app = Flask(__name__)
model = joblib.load("asteroid_predictor_model.pkl")

# Sample data for demonstration (Replace with your actual data)
df = pd.DataFrame({
    "miss_distance_km": [18000000, 32000000, 1000000, 800000],
    "diameter_min_m": [300, 10, 400, 18],
    "diameter_max_m": [500, 25, 600, 45],
    "velocity_kph": [70000, 25000, 40000, 88000],
    "prediction": ["Hazardous", "Not Hazardous", "Hazardous", "Not Hazardous"],
    "close_approach_date": pd.to_datetime(['2025-03-01', '2025-03-05', '2025-03-07', '2025-03-10'])
})

# Load the HTML from a file
@app.route('/')
def home():
    with open("index.html", "r", encoding="utf-8") as f:
        html = f.read()
    return render_template_string(html)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        asteroid_data = pd.DataFrame([[
            data["miss_distance_km"],
            data["velocity_kph"],
            data["diameter_min_m"],
            data["diameter_max_m"]
        ]], columns=["miss_distance_km", "velocity_kph", "diameter_min_m", "diameter_max_m"])

        prediction = model.predict(asteroid_data)[0]
        result = "Hazardous" if prediction else "Not Hazardous"
        return jsonify({"prediction": result})
    
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/line-graph')
def line_graph():
    # Group by asteroid size range
    size_ranges = pd.cut(df["diameter_min_m"], bins=[0, 100, 200, 300, 400, 500], labels=["Small", "Medium", "Large", "Very Large", "Huge"])
    size_count = size_ranges.value_counts()

    # Create line graph
    fig = px.line(x=size_count.index, y=size_count.values, title="Asteroid Frequency by Size",
                  labels={"x": "Size Range", "y": "Number of Asteroids"})
    graph_html = fig.to_html(full_html=False)

    # Correctly render HTML using render_template_string
    return render_template_string("""
        <html>
            <head>
                <title>Line Graph: Asteroid Frequency by Size</title>
            </head>
            <body>
                <h1>Asteroid Frequency by Size</h1>
                {{ graph_html|safe }}
            </body>
        </html>
    """, graph_html=graph_html)

# Time Series - Hazard levels over time
@app.route('/time-series')
def time_series():
    # Count hazardous and non-hazardous asteroids over time
    df['year'] = df['close_approach_date'].dt.year
    df['month'] = df['close_approach_date'].dt.month
    hazard_over_time = df.groupby(['year', 'month', 'prediction']).size().unstack().fillna(0)

    # Create time series plot
    fig = px.line(hazard_over_time, title="Hazardous Asteroids Over Time",
                  labels={"year": "Year", "value": "Count of Asteroids"})
    graph_html = fig.to_html(full_html=False)
    return render_template_string(f"<html><body>{graph_html}</body></html>")

# Scatter plot for asteroid trajectory (velocity vs. miss distance)
@app.route('/trajectory')
def trajectory():
    # Scatter plot of velocity vs. miss distance
    fig = px.scatter(df, x='miss_distance_km', y='velocity_kph', color='prediction',
                     title="Asteroid Trajectory (Velocity vs. Miss Distance)",
                     labels={"miss_distance_km": "Miss Distance (km)", "velocity_kph": "Velocity (kph)"})
    graph_html = fig.to_html(full_html=False)
    return render_template_string(f"<html><body>{graph_html}</body></html>")

if __name__ == '__main__':
    app.run(debug=True)
