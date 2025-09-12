import os
import joblib
import pandas as pd
from dash import Dash, html, dcc, Input, Output, State

# ---- Load saved pipeline (preprocess + RandomForest inside TransformedTargetRegressor) ----
PIPELINE_PATH = os.path.join(os.path.dirname(__file__), "models", "model_pipeline.pkl")
if not os.path.exists(PIPELINE_PATH):
    raise FileNotFoundError(
        "model_pipeline.pkl not found. Run the notebook cell that saves to app/models/model_pipeline.pkl"
    )
pipe = joblib.load(PIPELINE_PATH)

# ---- Your exact raw input schema/order ----
NUM_COLS = ["year", "engine", "max_power"]
CAT_COLS = ["fuel", "seller_type", "transmission"]
RAW_COLUMNS = NUM_COLS + CAT_COLS  # order matters

FUEL_OPTS = ["Petrol", "Diesel"]
SELLER_OPTS = ["Individual", "Dealer", "Trustmark Dealer"]
TRANS_OPTS = ["Manual", "Automatic"]

app = Dash(__name__)
server = app.server  # for gunicorn

def num_input(name, placeholder):
    return html.Div([
        html.Label(name, style={"fontWeight": 600}),
        dcc.Input(id=f"in-{name}", type="number", placeholder=placeholder, style={"width": "100%"})
    ])

def cat_dropdown(name, options):
    return html.Div([
        html.Label(name, style={"fontWeight": 600}),
        dcc.Dropdown(
            id=f"in-{name}",
            options=[{"label": v, "value": v} for v in options],
            placeholder=f"Select {name}"
        )
    ])

app.layout = html.Div(
    style={"maxWidth": "820px", "margin": "2rem auto", "fontFamily": "system-ui, sans-serif"},
    children=[
        html.H2("Car Price Predictor"),
        html.P("All input fields are required."),

        html.Div(style={"display": "grid", "gap": ".75rem"}, children=[
            num_input("year", "e.g., 2017"),
            num_input("engine", "cc (numeric)"),
            num_input("max_power", "bhp (numeric)"),
            cat_dropdown("fuel", FUEL_OPTS),
            cat_dropdown("seller_type", SELLER_OPTS),
            cat_dropdown("transmission", TRANS_OPTS),
        ]),

        html.Button("Predict", id="btn", style={"marginTop": "1rem", "padding": ".6rem 1rem"}),
        html.Div(id="pred", style={"marginTop": "1rem", "fontSize": "1.25rem", "fontWeight": 700}),
        html.Hr(),
        html.Div("ok", id="health")
    ]
)

@app.callback(
    Output("pred", "children"),
    Input("btn", "n_clicks"),
    State("in-year", "value"),
    State("in-engine", "value"),
    State("in-max_power", "value"),
    State("in-fuel", "value"),
    State("in-seller_type", "value"),
    State("in-transmission", "value"),
    prevent_initial_call=True
)
def predict(n, year, engine, max_power, fuel, seller_type, transmission):
    fields = {"year": year, "engine": engine, "max_power": max_power,
              "fuel": fuel, "seller_type": seller_type, "transmission": transmission}
    missing = [k for k, v in fields.items() if v in (None, "")]
    if missing:
        return f"Please fill: {', '.join(missing)}"

    row = {
        "year": float(year),
        "engine": float(engine),
        "max_power": float(max_power),
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
    }
    X = pd.DataFrame([row], columns=RAW_COLUMNS)

    try:
        y = float(pipe.predict(X)[0])
        return f"Predicted price: {y:,.2f}"
    except Exception as e:
        return f"Prediction error: {e}"

if __name__ == "__main__":
    # keep 0.0.0.0 for Docker; open http://localhost:8000 in your browser
    app.run(host="0.0.0.0", port=8000, debug=True)
