import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)

# The saved pipeline already contains imputation, scaling and one-hot encoding,
# so the form can send raw values like "Month-to-month".
model = joblib.load(BASE_DIR / "model.joblib")
with open(BASE_DIR / "model_meta.json") as f:
    META = json.load(f)

FEATURES = META["features"]
CATEGORICAL = META["categorical"]
NUMERIC = META["numeric"]

# Human-readable labels and grouping for the form
LABELS = {
    "Contract": "Contract",
    "tenure": "Months as a customer",
    "PaymentMethod": "Payment method",
    "InternetService": "Internet service",
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
}
GROUPS = [
    ("Account", ["Contract", "tenure", "PaymentMethod"]),
    ("Internet services", ["InternetService", "OnlineSecurity", "OnlineBackup",
                           "DeviceProtection", "TechSupport"]),
]
# Add-ons that only exist when the customer has internet
INTERNET_ADDONS = [f for f in FEATURES if "No internet service" in CATEGORICAL.get(f, [])]

DEFAULTS = {
    "Contract": "Month-to-month",
    "tenure": "",
    "PaymentMethod": "Electronic check",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
}


def parse_form(form):
    """Validate the submitted form. Returns (record, errors)."""
    record, errors = {}, {}
    no_internet = form.get("InternetService", "").strip() == "No"

    for feat in FEATURES:
        if no_internet and feat in INTERNET_ADDONS:
            continue  # filled in below
        raw = form.get(feat, "").strip()
        if feat in NUMERIC:
            lo, hi = NUMERIC[feat]["min"], NUMERIC[feat]["max"]
            try:
                value = float(raw.replace(",", "."))
            except ValueError:
                errors[feat] = "Enter a number of months."
                continue
            if value < 0:
                errors[feat] = "Months can't be negative."
                continue
            if value > hi:
                # The model never saw values above the training max; say so instead of guessing
                errors[feat] = f"The training data only goes up to {int(hi)} months."
                continue
            record[feat] = value
        else:
            if raw not in CATEGORICAL[feat]:
                errors[feat] = "Choose one of the options."
                continue
            record[feat] = raw

    # Keep answers consistent: no internet means the add-ons can't exist
    if record.get("InternetService") == "No":
        for feat in INTERNET_ADDONS:
            record[feat] = "No internet service"
    else:
        for feat in INTERNET_ADDONS:
            if record.get(feat) == "No internet service":
                errors[feat] = "Pick Yes or No: this customer has internet."

    return record, errors


def render(values, result=None, errors=None):
    return render_template(
        "index.html",
        groups=GROUPS,
        labels=LABELS,
        categorical=CATEGORICAL,
        numeric=NUMERIC,
        internet_addons=INTERNET_ADDONS,
        values=values,
        errors=errors or {},
        result=result,
        meta=META,
    )


@app.route("/")
def index():
    return render(DEFAULTS)


@app.route("/predict", methods=["POST"])
def predict():
    values = {f: request.form.get(f, "") for f in FEATURES}
    record, errors = parse_form(request.form)
    if errors:
        return render(values, errors=errors), 400

    X = pd.DataFrame([record], columns=FEATURES)
    proba = float(model.predict_proba(X)[0, 1])
    result = {
        "probability": round(proba * 100),
        "churn": proba >= 0.5,
    }
    return render(values, result=result)


if __name__ == "__main__":
    app.run(debug=True)
