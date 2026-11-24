"""
NeuroGuard AI — Aplikasi web Flask untuk prediksi penyakit Parkinson
menggunakan model Random Forest yang dilatih di notebook.

Jalankan:  python app.py   ->  buka http://127.0.0.1:5000
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "random_forest_parkinsons.pkl")
META_PATH  = os.path.join(BASE_DIR, "model", "model_metadata.json")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load model & metadata sekali saat startup
# ---------------------------------------------------------------------------
pipeline = joblib.load(MODEL_PATH)

with open(META_PATH, encoding="utf-8") as f:
    meta = json.load(f)

FEATURES      = meta["feature_list"]           # 32 fitur, urutan sesuai training
NUM_FEATURES  = meta["features_numeric"]
CAT_FEATURES  = meta["features_categorical"]
ACCURACY      = meta["evaluation_metrics_test"]["accuracy"]
N_ESTIMATORS  = meta["best_params"]["n_estimators"]
MAX_DEPTH     = meta["best_params"]["max_depth"]
TOP10         = meta["top10_feature_importance"]

# Kelas target (binary)
CLASS_NAMES = ["Tidak Terdiagnosis Parkinson", "Terdiagnosis Parkinson"]

# ---------------------------------------------------------------------------
# Deskripsi fitur untuk form
# ---------------------------------------------------------------------------
FEATURE_META = {
    # Demografi
    "Age":                      {"label": "Usia",                        "hint": "Tahun (usia pasien)",                   "unit": "tahun",  "group": "Demografi"},
    "Gender":                   {"label": "Jenis Kelamin",               "hint": "0=Laki-laki, 1=Perempuan",              "unit": "",       "group": "Demografi",  "type": "categorical"},
    "Ethnicity":                {"label": "Etnis",                       "hint": "Latar belakang etnis",                  "unit": "",       "group": "Demografi",  "type": "categorical"},
    "EducationLevel":           {"label": "Tingkat Pendidikan",          "hint": "Tahun lama pendidikan",                 "unit": "",       "group": "Demografi",  "type": "categorical"},
    "BMI":                      {"label": "BMI",                         "hint": "Indeks Massa Tubuh",                    "unit": "kg/m²",  "group": "Demografi"},
    # Gaya Hidup
    "Smoking":                  {"label": "Merokok",                     "hint": "Riwayat merokok",                       "unit": "",       "group": "Gaya Hidup", "type": "categorical"},
    "AlcoholConsumption":       {"label": "Konsumsi Alkohol",            "hint": "Unit per minggu",                       "unit": "unit/mg","group": "Gaya Hidup"},
    "PhysicalActivity":         {"label": "Aktivitas Fisik",             "hint": "Jam per minggu",                        "unit": "jam/mg", "group": "Gaya Hidup"},
    "DietQuality":              {"label": "Kualitas Diet",               "hint": "Skor kualitas diet (skala 0–10)",       "unit": "skor",   "group": "Gaya Hidup"},
    "SleepQuality":             {"label": "Kualitas Tidur",              "hint": "Skor kualitas tidur (skala 4–10)",      "unit": "skor",   "group": "Gaya Hidup"},
    # Riwayat Kesehatan
    "FamilyHistoryParkinsons":  {"label": "Riwayat Keluarga Parkinson",  "hint": "Ada anggota keluarga dengan Parkinson", "unit": "",       "group": "Riwayat",   "type": "categorical"},
    "TraumaticBrainInjury":     {"label": "Cedera Otak Traumatis",       "hint": "Riwayat cedera kepala/otak",            "unit": "",       "group": "Riwayat",   "type": "categorical"},
    "Hypertension":             {"label": "Hipertensi",                  "hint": "Riwayat tekanan darah tinggi",          "unit": "",       "group": "Riwayat",   "type": "categorical"},
    "Diabetes":                 {"label": "Diabetes",                    "hint": "Riwayat diabetes mellitus",             "unit": "",       "group": "Riwayat",   "type": "categorical"},
    "Depression":               {"label": "Depresi",                     "hint": "Riwayat gangguan depresi",              "unit": "",       "group": "Riwayat",   "type": "categorical"},
    "Stroke":                   {"label": "Stroke",                      "hint": "Riwayat stroke sebelumnya",             "unit": "",       "group": "Riwayat",   "type": "categorical"},
    # Kardiovaskular
    "SystolicBP":               {"label": "Tekanan Darah Sistolik",      "hint": "Tekanan darah atas",                    "unit": "mmHg",   "group": "Kardiovaskular"},
    "DiastolicBP":              {"label": "Tekanan Darah Diastolik",     "hint": "Tekanan darah bawah",                   "unit": "mmHg",   "group": "Kardiovaskular"},
    "CholesterolTotal":         {"label": "Kolesterol Total",            "hint": "Kolesterol total serum",                "unit": "mg/dL",  "group": "Kardiovaskular"},
    "CholesterolLDL":           {"label": "Kolesterol LDL",              "hint": "LDL (kolesterol jahat)",                "unit": "mg/dL",  "group": "Kardiovaskular"},
    "CholesterolHDL":           {"label": "Kolesterol HDL",              "hint": "HDL (kolesterol baik)",                 "unit": "mg/dL",  "group": "Kardiovaskular"},
    "CholesterolTriglycerides": {"label": "Trigliserida",                "hint": "Kadar trigliserida darah",              "unit": "mg/dL",  "group": "Kardiovaskular"},
    # Pemeriksaan Klinis
    "UPDRS":                    {"label": "Skor UPDRS",                  "hint": "Unified Parkinson's Disease Rating Scale","unit": "skor", "group": "Klinis"},
    "MoCA":                     {"label": "Skor MoCA",                   "hint": "Montreal Cognitive Assessment (0–30)",  "unit": "skor",   "group": "Klinis"},
    "FunctionalAssessment":     {"label": "Penilaian Fungsional",        "hint": "Skor kemampuan fungsional (0–10)",      "unit": "skor",   "group": "Klinis"},
    # Gejala Motorik
    "Tremor":                   {"label": "Tremor",                      "hint": "Gemetar tidak terkontrol",              "unit": "",       "group": "Gejala",    "type": "categorical"},
    "Rigidity":                 {"label": "Rigiditas",                   "hint": "Kekakuan otot",                         "unit": "",       "group": "Gejala",    "type": "categorical"},
    "Bradykinesia":             {"label": "Bradikinesia",                "hint": "Perlambatan gerakan",                   "unit": "",       "group": "Gejala",    "type": "categorical"},
    "PosturalInstability":      {"label": "Instabilitas Postural",       "hint": "Kesulitan keseimbangan & postur",       "unit": "",       "group": "Gejala",    "type": "categorical"},
    # Gejala Non-Motorik
    "SpeechProblems":           {"label": "Gangguan Bicara",             "hint": "Kesulitan berbicara",                   "unit": "",       "group": "Non-Motorik","type": "categorical"},
    "SleepDisorders":           {"label": "Gangguan Tidur",              "hint": "Masalah tidur (insomnia, dll.)",         "unit": "",       "group": "Non-Motorik","type": "categorical"},
    "Constipation":             {"label": "Sembelit",                    "hint": "Sembelit kronis",                       "unit": "",       "group": "Non-Motorik","type": "categorical"},
}

# Urutan grup
GROUP_ORDER = ["Demografi", "Gaya Hidup", "Riwayat", "Kardiovaskular",
               "Klinis", "Gejala", "Non-Motorik"]

# Pilihan kategori per fitur (dari data training)
CATEGORICAL_CHOICES = {
    "Gender":                  ["0", "1"],
    "Ethnicity":               ["0", "1", "2", "3"],
    "EducationLevel":          ["0", "1", "2", "3"],
    "Smoking":                 ["0", "1"],
    "FamilyHistoryParkinsons": ["0", "1"],
    "TraumaticBrainInjury":    ["0", "1"],
    "Hypertension":            ["0", "1"],
    "Diabetes":                ["0", "1"],
    "Depression":              ["0", "1"],
    "Stroke":                  ["0", "1"],
    "Tremor":                  ["0", "1"],
    "Rigidity":                ["0", "1"],
    "Bradykinesia":            ["0", "1"],
    "PosturalInstability":     ["0", "1"],
    "SpeechProblems":          ["0", "1"],
    "SleepDisorders":          ["0", "1"],
    "Constipation":            ["0", "1"],
}

# Label untuk pilihan kategori biner
BINARY_LABELS = {"0": "Tidak", "1": "Ya"}
GENDER_LABELS  = {"0": "Laki-laki", "1": "Perempuan"}
ETHNICITY_LABELS = {"0": "Kaukasia", "1": "Afrika-Amerika", "2": "Asia", "3": "Lainnya"}
EDUCATION_LABELS = {"0": "Tidak Sekolah", "1": "SMA", "2": "Sarjana", "3": "Pascasarjana"}

CHOICE_LABELS = {
    "Gender":        GENDER_LABELS,
    "Ethnicity":     ETHNICITY_LABELS,
    "EducationLevel": EDUCATION_LABELS,
}

# Rentang numerik dari dataset
NUMERIC_RANGES = {
    "Age":                      {"min": 50,    "max": 90,    "median": 70,    "step": "1"},
    "BMI":                      {"min": 15,    "max": 40,    "median": 27,    "step": "0.1"},
    "AlcoholConsumption":       {"min": 0,     "max": 20,    "median": 10,    "step": "0.1"},
    "PhysicalActivity":         {"min": 0,     "max": 10,    "median": 5,     "step": "0.1"},
    "DietQuality":              {"min": 0,     "max": 10,    "median": 5,     "step": "0.1"},
    "SleepQuality":             {"min": 4,     "max": 10,    "median": 7,     "step": "0.1"},
    "SystolicBP":               {"min": 90,    "max": 180,   "median": 120,   "step": "1"},
    "DiastolicBP":              {"min": 60,    "max": 120,   "median": 80,    "step": "1"},
    "CholesterolTotal":         {"min": 150,   "max": 300,   "median": 220,   "step": "1"},
    "CholesterolLDL":           {"min": 50,    "max": 200,   "median": 120,   "step": "1"},
    "CholesterolHDL":           {"min": 20,    "max": 100,   "median": 55,    "step": "1"},
    "CholesterolTriglycerides": {"min": 50,    "max": 400,   "median": 200,   "step": "1"},
    "UPDRS":                    {"min": 0,     "max": 199,   "median": 50,    "step": "0.1"},
    "MoCA":                     {"min": 0,     "max": 30,    "median": 15,    "step": "0.1"},
    "FunctionalAssessment":     {"min": 0,     "max": 10,    "median": 5,     "step": "0.1"},
}

# Warna per kelas
CLASS_COLORS = {
    "Tidak Terdiagnosis Parkinson": {"css": "healthy", "icon": "fa-brain",         "id": 0},
    "Terdiagnosis Parkinson":       {"css": "positive", "icon": "fa-circle-xmark",  "id": 1},
}

# ---------------------------------------------------------------------------
# Build form groups
# ---------------------------------------------------------------------------
def build_form_groups():
    groups = {g: [] for g in GROUP_ORDER}
    for feat in FEATURES:
        fm   = FEATURE_META.get(feat, {})
        g    = fm.get("group", "Lainnya")
        ftype = "categorical" if feat in CAT_FEATURES else "numeric"
        item = {
            "name":  feat,
            "label": fm.get("label", feat),
            "hint":  fm.get("hint", ""),
            "unit":  fm.get("unit", ""),
            "ftype": ftype,
        }
        if ftype == "numeric":
            nr = NUMERIC_RANGES.get(feat, {"min": 0, "max": 100, "median": 50, "step": "1"})
            item.update(nr)
        else:
            choices_raw = CATEGORICAL_CHOICES.get(feat, ["0", "1"])
            lbl_map = CHOICE_LABELS.get(feat, BINARY_LABELS)
            item["choices"] = [{"val": v, "label": lbl_map.get(v, v)} for v in choices_raw]
        groups.setdefault(g, []).append(item)
    return [(g, groups[g]) for g in GROUP_ORDER if groups.get(g)]


FORM_GROUPS = build_form_groups()


def importance_list():
    # Gabungkan kolom OHE (mis. Tremor_0 + Tremor_1) menjadi satu fitur asal,
    # supaya tidak tampil ganda di daftar.
    agg = {}
    for name, val in TOP10.items():
        base = name.rsplit("_", 1)[0] if name.rsplit("_", 1)[-1] in ("0", "1") else name
        fm = FEATURE_META.get(base, FEATURE_META.get(name, {}))
        label = fm.get("label", base)
        agg[label] = agg.get(label, 0.0) + val
    items = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
    top = items[0][1] if items else 1
    return [
        {"name": k, "value": round(v * 100, 1), "rel": round(v / top * 100, 1)}
        for k, v in items
    ]


IMPORTANCE = importance_list()


# Contoh data prediksi
EXAMPLES = {
    "sehat": {
        "Age": 65, "Gender": "0", "Ethnicity": "0", "EducationLevel": "2",
        "BMI": 25.0, "Smoking": "0", "AlcoholConsumption": 5.0,
        "PhysicalActivity": 6.0, "DietQuality": 7.0, "SleepQuality": 8.0,
        "FamilyHistoryParkinsons": "0", "TraumaticBrainInjury": "0",
        "Hypertension": "0", "Diabetes": "0", "Depression": "0", "Stroke": "0",
        "SystolicBP": 118, "DiastolicBP": 76,
        "CholesterolTotal": 200, "CholesterolLDL": 110, "CholesterolHDL": 60, "CholesterolTriglycerides": 150,
        "UPDRS": 15.0, "MoCA": 26.0, "FunctionalAssessment": 8.0,
        "Tremor": "0", "Rigidity": "0", "Bradykinesia": "0",
        "PosturalInstability": "0", "SpeechProblems": "0",
        "SleepDisorders": "0", "Constipation": "0",
    },
    "positif": {
        "Age": 72, "Gender": "1", "Ethnicity": "1", "EducationLevel": "1",
        "BMI": 23.0, "Smoking": "1", "AlcoholConsumption": 8.0,
        "PhysicalActivity": 2.0, "DietQuality": 4.0, "SleepQuality": 5.0,
        "FamilyHistoryParkinsons": "1", "TraumaticBrainInjury": "1",
        "Hypertension": "1", "Diabetes": "0", "Depression": "1", "Stroke": "0",
        "SystolicBP": 145, "DiastolicBP": 92,
        "CholesterolTotal": 240, "CholesterolLDL": 160, "CholesterolHDL": 38, "CholesterolTriglycerides": 310,
        "UPDRS": 95.0, "MoCA": 18.0, "FunctionalAssessment": 3.5,
        "Tremor": "1", "Rigidity": "1", "Bradykinesia": "1",
        "PosturalInstability": "1", "SpeechProblems": "1",
        "SleepDisorders": "1", "Constipation": "1",
    },
}

# ---------------------------------------------------------------------------
# Helper: parse inputs
# ---------------------------------------------------------------------------
def parse_inputs(source):
    values, errors = {}, []
    for feat in FEATURES:
        raw = source.get(feat)
        if raw is None or str(raw).strip() == "":
            errors.append(f"'{FEATURE_META.get(feat, {}).get('label', feat)}' belum diisi.")
            continue
        if feat in CAT_FEATURES:
            values[feat] = str(raw).strip()
        else:
            try:
                val = float(str(raw).replace(",", "."))
            except ValueError:
                errors.append(f"'{FEATURE_META.get(feat, {}).get('label', feat)}' harus berupa angka.")
                continue
            values[feat] = val
    return values, errors


def predict(values):
    df_in  = pd.DataFrame([values], columns=FEATURES)
    kelas  = str(pipeline.predict(df_in)[0])
    # Map kelas: "0" -> Tidak, "1" -> Ya
    label  = CLASS_NAMES[int(kelas)]
    proba  = pipeline.predict_proba(df_in)[0]
    prob_dict = {
        CLASS_NAMES[0]: round(float(proba[0]) * 100, 1),
        CLASS_NAMES[1]: round(float(proba[1]) * 100, 1),
    }
    cc = CLASS_COLORS.get(label, {"css": "positive", "icon": "fa-circle-xmark", "id": 1})
    return {
        "label":    label,
        "css":      cc["css"],
        "icon":     cc["icon"],
        "probs":    prob_dict,
        "top_prob": round(max(prob_dict.values()), 1),
    }


def page_context(**extra):
    ctx = dict(
        form_groups=FORM_GROUPS,
        examples=EXAMPLES,
        importance=IMPORTANCE,
        accuracy=ACCURACY,
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        class_names=CLASS_NAMES,
        class_colors=CLASS_COLORS,
        result=None,
        errors=[],
        form={},
    )
    ctx.update(extra)
    return ctx


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("home.html", active="home", **page_context())


@app.route("/prediksi", methods=["GET"])
def predict_page():
    return render_template("predict.html", active="predict", **page_context())


@app.route("/predict", methods=["POST"])
def predict_form():
    values, errors = parse_inputs(request.form)
    if errors:
        return render_template("predict.html", active="predict",
                               **page_context(errors=errors, form=request.form)), 400
    return render_template("predict.html", active="predict",
                           **page_context(result=predict(values), form=request.form))


@app.route("/api/predict", methods=["POST"])
def predict_api():
    """Endpoint JSON. Body: {fitur: nilai, ...} (32 fitur)."""
    payload = request.get_json(silent=True) or {}
    values, errors = parse_inputs(payload)
    if errors:
        return jsonify({"error": errors}), 400
    hasil = predict(values)
    return jsonify({
        "prediction":    hasil["label"],
        "probabilities": hasil["probs"],
    })


if __name__ == "__main__":
    app.run(debug=True)