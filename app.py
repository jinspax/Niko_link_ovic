from flask import Flask, request, render_template, send_file, jsonify
import pandas as pd
import numpy as np
import xgboost as xgb
import tempfile
import openpyxl
import json
import os
from io import BytesIO
import base64

app = Flask(__name__)

# -----------------------------
# MODEL CACHE (like your PyScript version)
# -----------------------------
_MODEL_CACHE = None
_category_map = None


def get_model():
    global _MODEL_CACHE

    if _MODEL_CACHE is None:
        model = xgb.XGBRegressor(enable_categorical=True)
        model.load_model("xgb_model.json")
        _MODEL_CACHE = model

    return _MODEL_CACHE


def get_categories():
    global _category_map

    if _category_map is None:
        with open("categories.json", "r") as f:
            _category_map = json.load(f)

    return _category_map


# -----------------------------
# CORE PREDICTION PIPELINE
# -----------------------------
def predict_excel(df):
    model = get_model()
    category_map = get_categories()
    
    # 1. Align input data to the model's expected features
    X = df.reindex(columns=model.get_booster().feature_names, fill_value=0).copy()

    # 2. Efficiently cast categories
    for col, cat_list in category_map.items():
        if col in X.columns:
            X[col] = pd.Categorical(X[col].astype(str), categories=cat_list)

    # 3. Predict and format the result
    preds = model.predict(X)
    df["Result"] = pd.Series(np.expm1(preds)).apply(lambda x: "{:.2f}".format(x)).values

    return df


# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")



@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    df = pd.read_excel(file)

    df = predict_excel(df)
    df = df.replace({np.nan: None})
    # 1. JSON for UI table
    preview = df.head(50).to_dict(orient="records")

    # 2. Excel for download
    output = BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)

    encoded_file = base64.b64encode(output.read()).decode()

    return jsonify({
        "preview": preview,
        "file": encoded_file
    })



@app.route("/Evaluate_Batch")
def evaluate_batch():
    return render_template("Evaluate_Batch.html")

@app.route("/General_Info")
def info():
    return render_template("General_Info.html")


@app.errorhandler(500)
def handle_internal_server_error(e):
    # This captures any unhandled exceptions and sends JSON instead of HTML
    return jsonify({
        "error": "Internal Server Error",
        "message": str(e)
    }), 500

if __name__ == "__main__":
    app.run(debug=True)