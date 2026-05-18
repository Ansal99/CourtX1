import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

CATEGORICAL = ["state", "case_type", "court_level",
               "evidence_strength", "opposite_lawyer_experience", "case_complexity"]
NUMERICAL   = ["case_duration_years", "has_documents", "num_witnesses",
               "legal_aid", "settlement_attempted", "bench_size", "num_cited_cases"]
FEATURES    = CATEGORICAL + NUMERICAL

VALID_STATES = [
    "Andhra Pradesh", "Assam", "Bihar", "Chandigarh", "Chhattisgarh",
    "DNH at Silvasa", "Delhi", "Diu and Daman", "Goa", "Gujarat",
    "Haryana", "Himachal Pradesh", "Jammu and Kashmir", "Jharkhand",
    "Karnataka", "Kerala", "Ladakh", "Madhya Pradesh", "Maharashtra",
    "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Orissa", "Puducherry",
    "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal"
]

ROLES = {
    "Criminal": [
        "Accused (You are the person who has been charged with the crime)",
        "Victim / Complainant (You are the person who filed the complaint against someone)",
    ],
    "Civil": [
        "Plaintiff (You are the person who started the case / is suing someone)",
        "Defendant (You are the person who has been sued / is defending the case)",
    ],
    "Consumer": [
        "Consumer / Complainant (You bought a product or service and faced a problem)",
        "Company / Seller (You are the business or seller being complained against)",
    ],
    "Family": [
        "Petitioner (You are the one who filed the case — divorce, custody, etc.)",
        "Respondent (You are the one who received the case notice)",
    ],
    "Labour": [
        "Employee (You are the worker who has filed the complaint against the employer)",
        "Employer (You are the company or employer defending against the employee's complaint)",
    ],
    "Motor Accident": [
        "Victim / Claimant (You were injured or suffered loss in the accident)",
        "Driver / Insurance Company (You are the driver or insurance company being claimed against)",
    ],
    "Property": [
        "Plaintiff (You are claiming ownership or rights over the property)",
        "Defendant (You are defending your ownership or possession of the property)",
    ],
    "Cheque Bounce": [
        "Payee (Someone gave you a cheque that bounced — money is owed to you)",
        "Drawer (You issued the cheque that has bounced)",
    ],
    "Tax": [
        "Taxpayer (You are the individual or business challenging the tax demand)",
        "Tax Department (You represent the government tax authority)",
    ],
}

DOC_CHECKLIST = {
    "Criminal": {
        "Accused (You are the person who has been charged with the crime)": {
            "your": [
                "Alibi proof (ticket / bill / hotel receipt showing you were elsewhere)",
                "Witness affidavit supporting your version",
                "Bail papers",
                "CCTV footage that clears you",
                "Medical report (your injury, if relevant)",
            ],
            "opp": [
                "FIR copy (the police complaint filed against you)",
                "Medical report of the victim's injury",
                "Forensic / DNA report",
                "Witness statements against you",
                "Panchnama / Scene inspection report",
            ],
        },
        "Victim / Complainant (You are the person who filed the complaint against someone)": {
            "your": [
                "FIR copy (the police complaint you filed)",
                "Medical report of your injury",
                "Witness statements supporting your version",
                "CCTV footage showing the incident",
                "Forensic / DNA report",
                "Panchnama / Scene inspection report",
            ],
            "opp": [
                "Alibi proof of the accused",
                "Bail papers of the accused",
                "Character certificate of the accused",
            ],
        },
    },
    "Civil": {
        "Plaintiff (You are the person who started the case / is suing someone)": {
            "your": [
                "Original agreement or contract",
                "Correspondence (letters / emails) as evidence",
                "Payment receipts",
                "Property papers (if relevant)",
                "Witness affidavits supporting your claim",
            ],
            "opp": [
                "Counter agreement claimed by the opposite party",
                "Proof of payment claimed by the opposite party",
                "Witness statements from the opposite side",
            ],
        },
        "Defendant (You are the person who has been sued / is defending the case)": {
            "your": [
                "Counter agreement supporting your position",
                "Proof of payment you have made",
                "Witness affidavits in your favor",
            ],
            "opp": [
                "Original agreement or contract filed by the opposite party",
                "Correspondence submitted by the opposite party",
                "Payment receipts submitted by the opposite party",
            ],
        },
    },
    "Consumer": {
        "Consumer / Complainant (You bought a product or service and faced a problem)": {
            "your": [
                "Purchase bill or receipt",
                "Warranty card",
                "Copy of complaint sent to the company",
                "Photos or videos of the defective product",
                "Medical bills (if the product caused injury)",
            ],
            "opp": [
                "Company's service records",
                "Warranty terms document",
                "Denial letter from the company",
            ],
        },
        "Company / Seller (You are the business or seller being complained against)": {
            "your": [
                "Service records showing the issue was handled",
                "Warranty terms document",
                "Denial or response letter sent to the consumer",
            ],
            "opp": [
                "Purchase bill or receipt",
                "Warranty card",
                "Complaint proof submitted by the consumer",
            ],
        },
    },
    "Family": {
        "Petitioner (You are the one who filed the case — divorce, custody, etc.)": {
            "your": [
                "Marriage certificate",
                "Birth certificates of children (if relevant)",
                "Income proof",
                "Bank statements",
                "Witness affidavits supporting your version",
            ],
            "opp": [
                "Opposite party's income proof",
                "Property documents of the opposite party",
                "Character evidence raised against you",
            ],
        },
        "Respondent (You are the one who received the case notice)": {
            "your": [
                "Your income proof",
                "Your property documents",
                "Character certificates in your favor",
            ],
            "opp": [
                "Marriage certificate submitted by the opposite party",
                "Birth certificates submitted by the opposite party",
                "Bank statements submitted by the opposite party",
            ],
        },
    },
    "Labour": {
        "Employee (You are the worker who has filed the complaint against the employer)": {
            "your": [
                "Appointment letter from the company",
                "Salary slips",
                "Termination letter",
                "Service record",
                "Witness statements from colleagues",
            ],
            "opp": [
                "Show-cause notice issued by the employer",
                "Performance reports filed by the employer",
                "Company policy document",
            ],
        },
        "Employer (You are the company or employer defending against the employee's complaint)": {
            "your": [
                "Show-cause notice you issued to the employee",
                "Performance reports",
                "Company policy document",
                "Attendance records",
            ],
            "opp": [
                "Appointment letter claimed by the employee",
                "Salary slips submitted by the employee",
                "Witness statements from colleagues",
            ],
        },
    },
    "Motor Accident": {
        "Victim / Claimant (You were injured or suffered loss in the accident)": {
            "your": [
                "FIR copy",
                "Medical bills and treatment reports",
                "Insurance policy of the vehicle",
                "RC book (Registration Certificate) of the vehicle",
                "Disability certificate (if applicable)",
                "Witness statements",
            ],
            "opp": [
                "Driver's valid license",
                "Vehicle fitness certificate",
                "Insurance company's denial letter",
            ],
        },
        "Driver / Insurance Company (You are the driver or insurance company being claimed against)": {
            "your": [
                "Valid driver's license",
                "Vehicle fitness certificate",
                "Valid insurance policy",
                "Accident reconstruction report",
            ],
            "opp": [
                "FIR copy filed by the victim",
                "Medical bills of the victim",
                "Witness statements from the victim's side",
            ],
        },
    },
    "Property": {
        "Plaintiff (You are claiming ownership or rights over the property)": {
            "your": [
                "Sale deed or registry document",
                "Mutation records (change of ownership records)",
                "Property tax receipts",
                "Survey map of the property",
                "Witness affidavits",
            ],
            "opp": [
                "Sale deed claimed by the opposite party",
                "Encumbrance certificate (property loan or lien history)",
                "Possession proof of the opposite party",
            ],
        },
        "Defendant (You are defending your ownership or possession of the property)": {
            "your": [
                "Possession proof",
                "Encumbrance certificate",
                "Counter sale deed (if any)",
            ],
            "opp": [
                "Sale deed submitted by the opposite party",
                "Mutation records submitted by the opposite party",
                "Property tax receipts submitted by the opposite party",
            ],
        },
    },
    "Cheque Bounce": {
        "Payee (Someone gave you a cheque that bounced — money is owed to you)": {
            "your": [
                "Original bounced cheque",
                "Bank return memo (the bank's rejection slip)",
                "Copy of legal notice sent to the drawer",
                "Proof of delivery of the legal notice",
                "Loan or transaction agreement showing the debt",
            ],
            "opp": [
                "Drawer's reply to your legal notice",
                "Proof of payment claimed by the drawer",
                "Bank account closure proof claimed by the drawer",
            ],
        },
        "Drawer (You issued the cheque that has bounced)": {
            "your": [
                "Proof of payment already made",
                "Bank account closure proof (if applicable)",
                "Your reply to the legal notice",
                "Agreement showing you had no outstanding liability",
            ],
            "opp": [
                "Original bounced cheque",
                "Bank return memo",
                "Legal notice sent to you",
            ],
        },
    },
    "Tax": {
        "Taxpayer (You are the individual or business challenging the tax demand)": {
            "your": [
                "Income Tax Return (ITR) copies",
                "CA certificate or chartered accountant report",
                "Bank statements",
                "Previous assessment orders",
                "Tax payment challans",
            ],
            "opp": [
                "Tax department's assessment order",
                "Notice of demand issued to you",
                "Audit report (if any)",
            ],
        },
        "Tax Department (You represent the government tax authority)": {
            "your": [
                "Assessment order issued",
                "Audit report",
                "Notice of demand issued",
            ],
            "opp": [
                "ITR copies submitted by the taxpayer",
                "CA certificate submitted by the taxpayer",
                "Bank statements submitted by the taxpayer",
            ],
        },
    },
}

# Load models once
models_loaded = False
xgb = rf = explainer = encoders = label_map = metrics_data = importance = None

def load_models():
    global xgb, rf, explainer, encoders, label_map, metrics_data, importance, models_loaded
    try:
        xgb       = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
        rf        = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
        explainer = joblib.load(os.path.join(MODEL_DIR, "shap_explainer.pkl"))
        encoders  = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
        with open(os.path.join(MODEL_DIR, "label_map.json")) as f:
            label_map = json.load(f)
        with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
            metrics_data = json.load(f)
        with open(os.path.join(MODEL_DIR, "feature_importance.json")) as f:
            importance = json.load(f)
        models_loaded = True
        return True
    except Exception as e:
        print(f"Model load error: {e}")
        return False

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/config")
def get_config():
    return jsonify({
        "states": VALID_STATES,
        "roles": ROLES,
        "doc_checklist": DOC_CHECKLIST,
        "case_types": list(ROLES.keys()),
        "models_loaded": models_loaded
    })

@app.route("/api/metrics")
def get_metrics():
    if not models_loaded:
        return jsonify({"error": "Models not loaded"}), 500
    return jsonify(metrics_data)

@app.route("/api/importance")
def get_importance():
    if not models_loaded:
        return jsonify({"error": "Models not loaded"}), 500
    return jsonify(importance)

@app.route("/api/predict", methods=["POST"])
def predict():
    if not models_loaded:
        return jsonify({"error": "Models not loaded. Run train_model.py first."}), 500

    data = request.json
    try:
        state          = data["state"]
        case_type      = data["case_type"]
        court_level    = data["court_level"]
        complexity     = data["case_complexity"]
        opp_lawyer     = data["opposite_lawyer_experience"]
        legal_aid_val  = data["legal_aid"]
        settlement_val = data["settlement_attempted"]
        duration       = float(data["case_duration_years"])
        witnesses      = float(data["num_witnesses"])
        bench_size     = float(data["bench_size"])
        cited_cases    = float(data["num_cited_cases"])
        has_documents  = float(data["has_documents"])
        evidence_str   = data["evidence_strength"]

        row = {}
        for col, val in {
            "state":                      state,
            "case_type":                  case_type,
            "court_level":                court_level,
            "evidence_strength":          evidence_str,
            "opposite_lawyer_experience": opp_lawyer,
            "case_complexity":            complexity,
        }.items():
            row[col] = encoders[col].transform([val])[0]

        row["case_duration_years"]  = duration
        row["has_documents"]        = has_documents
        row["num_witnesses"]        = witnesses
        row["legal_aid"]            = 1.0 if legal_aid_val == "Yes" else 0.0
        row["settlement_attempted"] = 1.0 if settlement_val == "Yes" else 0.0
        row["bench_size"]           = bench_size
        row["num_cited_cases"]      = cited_cases

        X_input  = pd.DataFrame([row])[FEATURES]
        xgb_prob = float(xgb.predict_proba(X_input)[0][1])
        rf_prob  = float(rf.predict_proba(X_input)[0][1])
        win_prob = round(0.70 * xgb_prob + 0.30 * rf_prob, 4)
        win_pct  = round(win_prob * 100, 1)

        if win_prob >= 0.70:
            verdict = "Strong Case"
        elif win_prob >= 0.50:
            verdict = "Moderate Case"
        else:
            verdict = "Weak Case"

        # SHAP values
        shap_vals   = explainer.shap_values(X_input)[0]
        shap_dict   = {f: round(float(v), 4) for f, v in zip(FEATURES, shap_vals)}
        sorted_shap = sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)

        readable_labels = {
            "state":                      "State",
            "case_type":                  "Case Type",
            "court_level":                "Court Level",
            "evidence_strength":          "Evidence Strength",
            "opposite_lawyer_experience": "Opposite Lawyer Experience",
            "case_complexity":            "Case Complexity",
            "case_duration_years":        "Case Duration (Years)",
            "has_documents":              "Has Documents",
            "num_witnesses":              "Number of Witnesses",
            "legal_aid":                  "Legal Aid Available",
            "settlement_attempted":       "Settlement Attempted",
            "bench_size":                 "Number of Judges",
            "num_cited_cases":            "Cited Cases (Precedents)",
        }

        shap_result = [
            {"feature": readable_labels.get(f, f), "value": v}
            for f, v in sorted_shap
        ]

        pos_factors = [s for s in shap_result if s["value"] > 0][:3]
        neg_factors = [s for s in shap_result if s["value"] < 0][:3]

        return jsonify({
            "win_probability": win_pct,
            "win_prob_raw": win_prob,
            "verdict": verdict,
            "xgb_prob": round(xgb_prob * 100, 1),
            "rf_prob": round(rf_prob * 100, 1),
            "shap_values": shap_result,
            "positive_factors": pos_factors,
            "negative_factors": neg_factors,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    load_models()
    app.run(debug=True, port=5000)