import plotly.express as px
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
import joblib

app = Flask(__name__)
model = joblib.load("asteroid_predictor_model.pkl")

# Sample data (replace with your actual processed dataset)
df = pd.DataFrame({
    "miss_distance_km": [18000000, 32000000, 1000000, 800000],
    "diameter_min_m": [300, 10, 400, 18],
    "diameter_max_m": [500, 25, 600, 45],
    "velocity_kph": [70000, 25000, 40000, 88000],
    "prediction": ["Hazardous", "Not Hazardous", "Hazardous", "Not Hazardous"],
    "close_approach_date": pd.to_datetime(['2025-03-01', '2025-03-05', '2025-03-07', '2025-03-10'])
})

@app.route('/')
def home():
    with open("index.html", "r", encoding="utf-8") as f:
        html = f.read()
    return render_template_string(html)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        asteroid_data = pd.DataFrame([[data["miss_distance_km"], data["velocity_kph"],
                                       data["diameter_min_m"], data["diameter_max_m"]]],
                                     columns=["miss_distance_km", "velocity_kph",
                                              "diameter_min_m", "diameter_max_m"])
        prediction = model.predict(asteroid_data)[0]
        result = "Hazardous" if prediction else "Not Hazardous"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/line-graph')
def line_graph():
    # Grouping based on diameter size
    df['size_category'] = pd.cut(df['diameter_min_m'],
                                 bins=[0, 50, 150, 300, 500, 1000],
                                 labels=["Tiny", "Small", "Medium", "Large", "Huge"])
    size_count = df['size_category'].value_counts().sort_index()

    fig = px.line(x=size_count.index, y=size_count.values,
                  labels={"x": "Asteroid Size Category", "y": "Count"},
                  title="Asteroid Frequency by Size Category")

    graph_html = fig.to_html(full_html=False)
    return render_template_string("""
        <html>
            <head><title>Line Graph</title></head>
            <body>
                <h1>Asteroid Frequency by Size</h1>
                {{ graph_html|safe }}
            </body>
        </html>
    """, graph_html=graph_html)

@app.route('/time-series')
def time_series():
    df['year_month'] = df['close_approach_date'].dt.to_period('M').astype(str)
    hazard_counts = df.groupby(['year_month', 'prediction']).size().unstack().fillna(0)

    fig = px.line(hazard_counts,
                  labels={"value": "Asteroid Count", "year_month": "Date"},
                  title="Hazardous vs Non-Hazardous Asteroids Over Time")
    
    graph_html = fig.to_html(full_html=False)
    return render_template_string("""
        <html>
            <head><title>Time Series</title></head>
            <body>
                <h1>Hazard Levels Over Time</h1>
                {{ graph_html|safe }}
            </body>
        </html>
    """, graph_html=graph_html)

@app.route('/trajectory')
def trajectory():
    fig = px.scatter(df, x="miss_distance_km", y="velocity_kph", color="prediction",
                     title="Asteroid Trajectory (Miss Distance vs Velocity)",
                     labels={"miss_distance_km": "Miss Distance (km)", "velocity_kph": "Velocity (kph)"})
    
    graph_html = fig.to_html(full_html=False)
    return render_template_string("""
        <html>
            <head><title>Asteroid Trajectory</title></head>
            <body>
                <h1>Asteroid Trajectory (Scatter)</h1>
                {{ graph_html|safe }}
            </body>
        </html>
    """, graph_html=graph_html)

if __name__ == '__main__':
    app.run(debug=True)
