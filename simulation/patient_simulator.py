# ============================================================
# PATIENT DISCHARGE SIMULATOR — Hospital Readmission ML System
# ============================================================

import requests
import random
import time
import os

# ── API URL — local dev vs deployed ──────────────────────────
# Set ENV variable for production: READMISSION_API_URL=https://your-render-url.onrender.com
API_URL = os.getenv(
    "READMISSION_API_URL",
    "http://127.0.0.1:8000"
) + "/predict"

_AGE_BRACKETS = ["[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)"]
_RACES        = ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"]


# ============================================================
# GENERATE SYNTHETIC PATIENT DISCHARGE RECORD
# ============================================================

def generate_patient(scenario: str = "random") -> dict:
    """
    Scenarios:
        random    — mixed realistic discharges
        high_risk — frequent-flyer, polypharmacy, long-stay profile
        low_risk  — first admission, short stay, few meds
    """

    base = {
        "race": random.choice(_RACES),
        "gender": random.choice(["Male", "Female"]),
        "medical_specialty": "Missing",
        "diag_1": "250.83", "diag_2": "401", "diag_3": "272",
        "max_glu_serum": "None", "A1Cresult": "None",
        "metformin": "No", "insulin": random.choice(["No", "Steady", "Up", "Down"]),
        "change": random.choice(["Ch", "No"]),
        "diabetesMed": "Yes",
    }

    if scenario == "high_risk":
        base.update({
            "age":                       random.choice(_AGE_BRACKETS[-3:]),
            "admission_type_id":         1,
            "discharge_disposition_id":  1,
            "admission_source_id":       7,
            "time_in_hospital":          random.randint(8, 14),
            "num_lab_procedures":        random.randint(50, 90),
            "num_procedures":            random.randint(0, 3),
            "num_medications":           random.randint(16, 30),
            "number_outpatient":         random.randint(0, 3),
            "number_emergency":          random.randint(1, 4),
            "number_inpatient":          random.randint(3, 6),
            "number_diagnoses":          random.randint(8, 16),
        })

    elif scenario == "low_risk":
        base.update({
            "age":                       random.choice(_AGE_BRACKETS[:3]),
            "admission_type_id":         3,
            "discharge_disposition_id":  1,
            "admission_source_id":       1,
            "time_in_hospital":          random.randint(1, 3),
            "num_lab_procedures":        random.randint(10, 35),
            "num_procedures":            random.randint(0, 1),
            "num_medications":           random.randint(1, 8),
            "number_outpatient":         0,
            "number_emergency":          0,
            "number_inpatient":          0,
            "number_diagnoses":          random.randint(2, 5),
        })

    else:  # random
        base.update({
            "age":                       random.choice(_AGE_BRACKETS),
            "admission_type_id":         random.choice([1, 2, 3]),
            "discharge_disposition_id":  random.choice([1, 1, 1, 6, 3]),
            "admission_source_id":       random.choice([1, 7, 4]),
            "time_in_hospital":          random.randint(1, 12),
            "num_lab_procedures":        random.randint(5, 100),
            "num_procedures":            random.randint(0, 6),
            "num_medications":           random.randint(1, 25),
            "number_outpatient":         random.randint(0, 3),
            "number_emergency":          random.randint(0, 2),
            "number_inpatient":          random.randint(0, 4),
            "number_diagnoses":          random.randint(1, 16),
        })

    return base


# ============================================================
# SEND TO API + PRINT RESULT
# ============================================================

def send_patient(patient: dict, idx: int):

    try:
        response = requests.post(API_URL, json=patient, timeout=10)

        if response.status_code == 200:
            result = response.json()
            print(f"[{idx+1}]  age={patient['age']:9s}  LOS={patient['time_in_hospital']:2d}d  "
                  f"prior_inpatient={patient['number_inpatient']}  meds={patient['num_medications']:2d}  "
                  f"->  prob={result['readmission_probability']:.4f}  "
                  f"band={result['risk_band']}  decision={result['decision']}")
        else:
            print(f"[{idx+1}] API error: {response.status_code}")

    except Exception as e:
        print(f"[{idx+1}] Connection error: {e}")


# ============================================================
# RUN SIMULATION
# ============================================================

def simulate_discharges(n: int = 20, scenario: str = "random"):

    print(f"\nSimulating {n} patient discharges  |  scenario={scenario}\n" + "-" * 70)

    for i in range(n):
        patient = generate_patient(scenario)
        send_patient(patient, i)
        time.sleep(0.5)

    print("-" * 70 + "\nSimulation complete")


if __name__ == "__main__":
    simulate_discharges(20, scenario="random")
