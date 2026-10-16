# GlucoTwin — A Digital Twin for Type 2 Diabetes

**Predicting glucose spikes 2 hours ahead, explaining why, and simulating what would prevent them.**

Submission for the **Happiest Health Digital Twin Challenge 2026** (Unstop).

> ⚠️ **Not a medical device.** GlucoTwin is a research prototype built entirely on **synthetic data**.
> It must not be used for diagnosis, treatment or any clinical decision.

| | |
|---|---|
| 🎥 **Demo video** | **[VIDEO LINK — to be added](#)** |
| 🏗️ **Architecture** | [docs/architecture.pdf](docs/architecture.pdf) |
| 📊 **Presentation** | [docs/presentation.pdf](docs/presentation.pdf) |
| 📄 **License** | [MIT](LICENSE) |

---

## Team

| | |
|---|---|
| **Team name** | _[TEAM NAME]_ |
| **College** | Birla Institute of Technology (BIT), Mesra |
| **Members** | _[Utkarsh Pal]_<br>_[Ayush Choubey]_<br>_ |
| **Repository** | https://github.com/swinter-ctrl/GlucoTwin_BITMesra |

---

## Problem statement

India has more than 100 million people living with diabetes, most of them with type 2 diabetes (T2D).
Between clinic visits, a patient's glucose swings with every meal, walk and short night of sleep. The
data to see these swings now exists (CGMs, fitness bands, phones), but it sits apart from the
health record that explains *who* the patient is. Clinicians get either a lab value every three
months or a raw CGM trace. Neither tells them **what is about to happen** or **what would change it**.

## Healthcare use case

GlucoTwin is a patient-specific **digital twin** that fuses two streams:

1. **Static EHR:** age, sex, BMI, HbA1c, T2D/prediabetes, hypertension, metformin, lipids.
2. **Dynamic wearables:** CGM every 5 min, steps, heart rate, sleep and logged meals.

For each patient, at any moment, it:

- **Predicts** the probability that glucose will exceed **180 mg/dL within the next 2 hours**.
- **Explains** the prediction in plain English (top 3 SHAP reasons), e.g. *"Ate 105 g carbs, 10 min ago — raises risk"*.
- **Simulates what-ifs** with the twin: a smaller meal, a 20-minute post-meal walk or a better night's sleep,
  shown as a new glucose curve and a new risk score.

A **doctor-facing dashboard** ranks patients by current risk, so a care team can triage who needs a nudge
now and show the patient which change helps them most.

### Demo story

Patient `be33d994`: a 51-year-old woman with T2D, on metformin, BMI 29.3, with hypertension.
She slept **5.2 h**, then ate a **105 g** rice-heavy lunch at 13:15. At 13:25 the model gives **98%** risk.

| What-if (twin re-simulation of the next 2 h) | Spike probability | Peak glucose | Minutes > 180 |
|---|---|---|---|
| As usual | 100% | 198 mg/dL | 40 |
| 20-minute post-meal walk | 36% | 173 mg/dL | 0 |
| 60 g lunch instead of 105 g | 0% | 160 mg/dL | 0 |
| Walk **and** 60 g lunch | **0%** | **137 mg/dL** | **0** |

Across all 238 patients' lunches, a 20-minute post-meal walk lowers the simulated peak by **24 mg/dL** on average
and the mean spike probability from **74% to 33%**.

![Dashboard chart: observed CGM, twin projection and what-if overlay](docs/img/dashboard_demo_chart.png)

---

## Technical stack

| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| Data and ML | pandas, NumPy, PyArrow, scikit-learn, **LightGBM**, **SHAP**, joblib |
| EHR data | [Synthea](https://synthetichealth.github.io/synthea/) pre-generated sample (CSV, 1,171 patients) |
| Wearable data | Our own physiological simulator (`src/wearable`) |
| API | **FastAPI** + uvicorn |
| Dashboard | **Streamlit** + Plotly |
| Docs and testing | Matplotlib (figures), Mermaid (architecture), pytest |

Versions are pinned in [requirements.txt](requirements.txt).

## Architecture

![Architecture](docs/img/architecture.png)

Full diagram: [docs/architecture.pdf](docs/architecture.pdf) (source: [docs/architecture.mmd](docs/architecture.mmd)).

| Module | What it does |
|---|---|
| `src/ehr` | Loads Synthea CSVs. Keeps adults with T2D or prediabetes. One row per patient → `patients.parquet` (238 patients). |
| `src/wearable` | Deterministic simulator seeded from each patient's EHR. 14 days × 5 min of CGM, HR, steps, meals and sleep → `wearables.parquet`. |
| `src/features` | Fuses both streams into 32 features plus the label → `features.parquet` (948k rows). |
| `src/model` | LightGBM, patient-level split, naive baseline, metrics, SHAP and plain-English reasons. |
| `src/twin` | What-if engine: edits meal carbs, walk or sleep, re-simulates 2 h, and re-scores. |
| `api/` | FastAPI: `GET /patients`, `GET /patients/{id}`, `POST /predict`, `POST /whatif`. |
| `dashboard/` | Streamlit: risk-sorted patient list, EHR card, 24 h CGM with risk band, top-3 reasons, what-if sliders. |

### How the simulator works

- **Baseline glucose from HbA1c:** eAG (mg/dL) = 28.7 × HbA1c − 46.7. The fasting level is set so the simulated mean matches eAG (correlation 0.945).
- **Meals:** 3 per day plus an optional snack, in Indian carb-heavy patterns. Glucose peaks 45–75 min after eating. The rise is scaled by carbs × insulin resistance (from BMI and HbA1c); higher resistance means a bigger, slower curve.
- **Activity:** steps lower glucose for 1–3 h, strongly after meals and modestly when fasting. Heart rate rises with activity.
- **Sleep:** under 6 h, or low efficiency, raises next-day insulin resistance.
- **Metformin:** blunts the rise by 15%.
- **Noise:** slow physiological drift plus CGM sensor noise, clipped to 40–400 mg/dL. The simulator is deterministic per seed.

## AI/ML model

| | |
|---|---|
| **Task** | Binary classification: will max glucose in (t, t+120 min] exceed 180 mg/dL? |
| **Unit** | (patient, 5-minute timestamp) |
| **Model** | LightGBM (500 trees, learning rate 0.05, 31 leaves) |
| **Features (32)** | Current glucose; lags at 15/30/60 min; rate of change; 60/120 min rolling stats; carbs in last 1–2 h; minutes since meal; steps in last 1–2 h; mean HR; last night's sleep hours and efficiency; time of day; static EHR fields |
| **Split** | `GroupShuffleSplit` **by patient**: 178 train / 60 test. No patient appears in both (tested in `tests/test_split.py`). |
| **Evaluated on** | Test-patient moments with glucose ≤ 180 now (predicting an *upcoming* spike): 227,025 rows, 12.5% positive |
| **Baseline** | Naive rule: current glucose > threshold (threshold swept) |
| **Explainability** | SHAP TreeExplainer. Related features are grouped into plain-English reasons. |
| **Alert threshold** | 0.629, chosen for 80% precision |

### Results (held-out patients)

| Metric | **LightGBM** | Naive baseline |
|---|---|---|
| AUROC | **0.972** | 0.554 |
| AUPRC | **0.836** | 0.162 |
| Recall at 80% precision | **0.66** | 0.00 (best achievable precision is 0.48) |
| Spike events alerted in advance | **1,219 / 1,282 (95%)** | — |
| Median alert lead time | **95 min** | — |
| AUROC, all rows (incl. glucose already > 180) | 0.960 | 0.676 |

Source: [models/metrics.json](models/metrics.json).

**Top features (mean |SHAP|):** time of day (sin/cos), HbA1c, minutes since meal, steps in last hour,
carbs in last hour, 2-hour minimum glucose, BMI, last night's sleep efficiency, carbs in last 2 h.
The directions make physiological sense: more carbs and higher HbA1c or BMI raise risk, while recent walking
and better sleep lower it.

![SHAP summary](docs/img/shap_summary.png)

---

## How to run

**Prerequisites:** Python 3.11. `make` is optional; on Windows, use the commands shown under each target.

```bash
# 1. Environment
py -3.11 -m venv .venv                       # Linux/macOS: python3.11 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Linux/macOS: .venv/bin/python

# 2. Synthea sample EHR (≈9 MB zip, 1,171 synthetic patients)
curl -L -o synthea.zip https://synthetichealth.github.io/synthea-sample-data/downloads/synthea_sample_data_csv_apr2020.zip
unzip -j synthea.zip "csv/*" -d data/raw/synthea/

# 3. Pipeline
make data        # python -m src.ehr.build && python -m src.wearable.simulate && python -m src.features.build
make train       # python -m src.model.train   -> models/model.joblib, models/metrics.json
make test        # python -m pytest -q

# 4. Apps (two terminals)
make api         # uvicorn api.main:app --port 8800      -> http://localhost:8800/docs
make dashboard   # streamlit run dashboard/app.py        -> http://localhost:8501
```

- The dashboard opens on the demo patient at the demo time. Use the sidebar to pick another moment.
- If the API is not running, the dashboard calls the same code in-process.
- **CLI twin demo:** `python -m src.twin.demo` prints the risk before and after a 20-minute walk (`--help` lists the other options).
- **Simulator plots:** `python -m src.wearable.plot` → `docs/img/sim_*_hba1c.png`.

---

## Data and privacy

- **All data is synthetic.** The EHR comes from [Synthea](https://synthetichealth.github.io/synthea/), an open-source simulator
  of realistic but fictional patients. Wearable streams come from our own simulator, seeded from those records.
- **No real person's data** is collected, stored or processed, so the project is safe under India's **DPDP Act 2023** and HIPAA.
- Raw and processed data are not committed. Running the pipeline above regenerates them deterministically.
- The patient identifiers shown are Synthea's random UUIDs and correspond to no real individual.
- In a real deployment, CGM and EHR data would need consent, encryption in transit and at rest, access control and audit logging.
  None of these are implemented in this prototype.

## Limitations

- **The model partly learns our simulator.** The physiology is hand-built and simplified, so the metrics show how well
  the model recovers our simulator's rules, not real-world performance. **Validating on real CGM data (e.g. public
  T2D CGM datasets) is future work.**
- **Time of day is an over-strong signal.** In the simulator every patient eats at similar times (±25 min), so the
  clock alone reaches AUROC 0.935. Without time-of-day features the model scores AUROC 0.927 / AUPRC 0.68 vs 0.972 / 0.836 with them.
  This also makes most patients look high-risk around lunchtime. Giving each patient their own meal schedule is the next fix.
- **Narrow HbA1c range.** Synthea's sample tops out at 7.5%. We removed implausible values (< 4%), leaving
  39 T2D and 199 prediabetes patients, fewer poorly controlled patients than an Indian T2D clinic would see.
- **Population mismatch.** Synthea models a US (Massachusetts) population. Our meal patterns are Indian, but the
  demographics and lab distributions are not.
- **Simplified what-ifs.** Each intervention is a single lever: carbs of one meal, a fixed-intensity walk, or one night's sleep.
  Meal composition (fibre, protein, glycaemic index), medication timing and stress are not modelled.
- **Two risk views.** The ML model sees only the past, so a planned walk shows up in its risk curve only once it happens.
  The twin's Monte Carlo spike probability covers the planned future.

## Disclaimer

**GlucoTwin is a research prototype and NOT a medical device.** It has not been clinically validated or reviewed by
any regulator (including CDSCO or the FDA). It must not be used to diagnose, treat or manage any medical condition,
or to change medication, diet or activity. All data shown is synthetic.

## License

[MIT](LICENSE) © 2026 swinter-ctrl and team.
Synthea sample data is © The MITRE Corporation, used under the Apache License 2.0.
