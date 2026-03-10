import os
import io
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.model import train_model
from src.confidence import get_confidence_scores
from src.drift import detect_drift
from src.trust_logic import trust_decision

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(32))

ALLOWED_EXTENSIONS = {"csv"}

# ---------------------------------------------------------------------------
# Train baseline model once at startup
# ---------------------------------------------------------------------------
_df = pd.read_csv("data/loan_data.csv")
_model, _X_train, _X_test, _y_train, _y_test = train_model(_df)
_confidence_scores = get_confidence_scores(_model, _X_test)
_drift_df = detect_drift(_X_train, _X_test)
_drift_exists = bool(_drift_df["drift_detected"].any())

_y_pred = _model.predict(_X_test)
_metrics = {
    "accuracy": round(accuracy_score(_y_test, _y_pred) * 100, 2),
    "precision": round(precision_score(_y_test, _y_pred, zero_division=0) * 100, 2),
    "recall": round(recall_score(_y_test, _y_pred, zero_division=0) * 100, 2),
    "f1": round(f1_score(_y_test, _y_pred, zero_division=0) * 100, 2),
}

_decisions = [trust_decision(c, _drift_exists) for c in _confidence_scores]
_decision_counts = pd.Series(_decisions).value_counts().to_dict()
_conf_hist, _conf_edges = np.histogram(_confidence_scores, bins=10)
_conf_labels = [f"{_conf_edges[i]:.2f}-{_conf_edges[i+1]:.2f}" for i in range(len(_conf_hist))]


def _allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# PUBLIC PAGES
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/check", methods=["GET", "POST"])
def check_eligibility():
    """Loan eligibility checker — any visitor can use this."""
    result = None
    if request.method == "POST":
        try:
            age = int(request.form["age"])
            income = int(request.form["income"])
            credit_score = int(request.form["credit_score"])
            loan_amount = int(request.form["loan_amount"])
            loan_term = int(request.form["loan_term"])
            employment = request.form["employment_status"]

            input_data = pd.DataFrame([{
                "Age": age,
                "Income": income,
                "Credit_Score": credit_score,
                "Loan_Amount": loan_amount,
                "Loan_Term": loan_term,
                "Employment_Status": employment,
            }])
            input_encoded = pd.get_dummies(input_data, drop_first=True)
            input_encoded = input_encoded.reindex(columns=_X_train.columns, fill_value=0)

            prediction = int(_model.predict(input_encoded)[0])
            proba = _model.predict_proba(input_encoded)[0]
            confidence = float(np.max(proba))
            decision = trust_decision(confidence, _drift_exists)

            # Personalized tips based on inputs
            tips = []
            if credit_score < 650:
                tips.append("Your credit score is below 650. Improving it by paying bills on time and reducing debt can significantly boost approval chances.")
            if income > 0 and loan_amount > income * 5:
                tips.append("Your requested loan is more than 5x your annual income. Lenders typically prefer a lower ratio — consider a smaller amount.")
            if employment == "Unemployed":
                tips.append("Being currently unemployed lowers approval odds. A co-signer or proof of upcoming employment can help.")
            if loan_term >= 60:
                tips.append("Longer loan terms mean more interest paid overall. A shorter term could save you money if you can afford higher monthly payments.")
            if not tips:
                tips.append("Your profile looks solid! Keep your credit utilization low and maintain steady income.")

            # Monthly EMI estimate
            annual_rate = 0.085  # illustrative 8.5%
            monthly_rate = annual_rate / 12
            n_payments = loan_term
            if monthly_rate > 0 and n_payments > 0:
                emi = loan_amount * monthly_rate * (1 + monthly_rate)**n_payments / ((1 + monthly_rate)**n_payments - 1)
            else:
                emi = loan_amount / max(n_payments, 1)

            result = {
                "prediction": "Likely Approved" if prediction == 1 else "Likely Rejected",
                "approved": prediction == 1,
                "confidence": round(confidence * 100, 2),
                "decision": decision,
                "probabilities": {
                    "reject": round(float(proba[0]) * 100, 2),
                    "approve": round(float(proba[1]) * 100, 2),
                },
                "tips": tips,
                "emi": round(emi, 2),
            }
        except (KeyError, ValueError) as e:
            result = {"error": f"Please fill in all fields correctly. ({e})"}

    return render_template("check.html", result=result)


@app.route("/emi", methods=["GET", "POST"])
def emi_calculator():
    """EMI calculator — a daily-use tool for anyone."""
    calc = None
    if request.method == "POST":
        try:
            principal = float(request.form["principal"])
            annual_rate = float(request.form["rate"]) / 100
            months = int(request.form["months"])
            monthly_rate = annual_rate / 12

            if monthly_rate > 0 and months > 0:
                emi = principal * monthly_rate * (1 + monthly_rate)**months / ((1 + monthly_rate)**months - 1)
            else:
                emi = principal / max(months, 1)

            total_payment = emi * months
            total_interest = total_payment - principal

            # Amortization schedule (first 12 months or full term)
            schedule = []
            balance = principal
            for m in range(1, min(months + 1, 361)):
                interest_part = balance * monthly_rate
                principal_part = emi - interest_part
                balance -= principal_part
                schedule.append({
                    "month": m,
                    "emi": round(emi, 2),
                    "principal": round(principal_part, 2),
                    "interest": round(interest_part, 2),
                    "balance": round(max(balance, 0), 2),
                })

            calc = {
                "emi": round(emi, 2),
                "total_payment": round(total_payment, 2),
                "total_interest": round(total_interest, 2),
                "principal": round(principal, 2),
                "months": months,
                "rate": float(request.form["rate"]),
                "schedule": schedule,
            }
        except (KeyError, ValueError, ZeroDivisionError) as e:
            calc = {"error": f"Please check your inputs. ({e})"}

    return render_template("emi.html", calc=calc)


@app.route("/analyze", methods=["GET", "POST"])
def analyze_dataset():
    """Let users upload their own CSV and get a trust/drift/confidence report."""
    report = None
    if request.method == "POST":
        file = request.files.get("dataset")
        if not file or file.filename == "":
            report = {"error": "Please select a CSV file to upload."}
        elif not _allowed_file(file.filename):
            report = {"error": "Only .csv files are supported."}
        else:
            try:
                raw = file.read()
                user_df = pd.read_csv(io.BytesIO(raw))

                # Validate required columns
                required = {"Age", "Income", "Credit_Score", "Loan_Amount", "Loan_Term", "Employment_Status", "Loan_Approved"}
                missing = required - set(user_df.columns)
                if missing:
                    report = {
                        "error": f"Your CSV is missing columns: {', '.join(sorted(missing))}. "
                                 f"Required columns: {', '.join(sorted(required))}",
                        "columns_found": list(user_df.columns),
                    }
                else:
                    # Train model on user data
                    u_model, u_X_train, u_X_test, u_y_train, u_y_test = train_model(user_df)
                    u_conf = get_confidence_scores(u_model, u_X_test)
                    u_drift = detect_drift(u_X_train, u_X_test)
                    u_drift_exists = bool(u_drift["drift_detected"].any())

                    u_y_pred = u_model.predict(u_X_test)
                    u_decisions = [trust_decision(c, u_drift_exists) for c in u_conf]
                    u_dec_counts = pd.Series(u_decisions).value_counts().to_dict()

                    u_hist, u_edges = np.histogram(u_conf, bins=10)
                    u_labels = [f"{u_edges[i]:.2f}-{u_edges[i+1]:.2f}" for i in range(len(u_hist))]

                    report = {
                        "rows": len(user_df),
                        "test_size": len(u_X_test),
                        "metrics": {
                            "accuracy": round(accuracy_score(u_y_test, u_y_pred) * 100, 2),
                            "precision": round(precision_score(u_y_test, u_y_pred, zero_division=0) * 100, 2),
                            "recall": round(recall_score(u_y_test, u_y_pred, zero_division=0) * 100, 2),
                            "f1": round(f1_score(u_y_test, u_y_pred, zero_division=0) * 100, 2),
                        },
                        "drift_data": u_drift.to_dict(orient="records"),
                        "drift_exists": u_drift_exists,
                        "decision_counts": json.dumps(u_dec_counts),
                        "conf_labels": json.dumps(u_labels),
                        "conf_values": json.dumps(u_hist.tolist()),
                        "sample_decisions": pd.DataFrame({
                            "confidence": u_conf,
                            "decision": u_decisions,
                        }).head(20).to_dict(orient="records"),
                    }
            except Exception as e:
                report = {"error": f"Could not process your file: {e}"}

    return render_template("analyze.html", report=report)


@app.route("/tips")
def credit_tips():
    return render_template("tips.html")


@app.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html")


# ---------------------------------------------------------------------------
# ADMIN DASHBOARD (model internals)
# ---------------------------------------------------------------------------
@app.route("/dashboard")
def dashboard():
    drift_data = _drift_df.to_dict(orient="records")
    sample_results = pd.DataFrame({
        "confidence": _confidence_scores,
        "decision": _decisions,
    }).head(20).to_dict(orient="records")

    return render_template(
        "dashboard.html",
        metrics=_metrics,
        drift_data=drift_data,
        drift_exists=_drift_exists,
        decision_counts=json.dumps(_decision_counts),
        conf_labels=json.dumps(_conf_labels),
        conf_values=json.dumps(_conf_hist.tolist()),
        sample_results=sample_results,
        total_samples=len(_y_test),
    )


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    try:
        input_data = pd.DataFrame([{
            "Age": int(data["age"]),
            "Income": int(data["income"]),
            "Credit_Score": int(data["credit_score"]),
            "Loan_Amount": int(data["loan_amount"]),
            "Loan_Term": int(data["loan_term"]),
            "Employment_Status": data["employment_status"],
        }])
        input_encoded = pd.get_dummies(input_data, drop_first=True)
        input_encoded = input_encoded.reindex(columns=_X_train.columns, fill_value=0)

        prediction = int(_model.predict(input_encoded)[0])
        proba = _model.predict_proba(input_encoded)[0]
        confidence = float(np.max(proba))
        decision = trust_decision(confidence, _drift_exists)

        return jsonify({
            "prediction": "Approved" if prediction == 1 else "Rejected",
            "confidence": round(confidence * 100, 2),
            "trust_decision": decision,
            "drift_detected": _drift_exists,
        })
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400


@app.route("/data/<path:filename>")
def download_sample(filename):
    """Serve sample CSV for download."""
    return send_from_directory("data", filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
