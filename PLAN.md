# Project plan — Predictive Disk-Failure Self-Healing Agent

Purpose
-------
This document outlines the research & engineering plan for the Predictive Disk-Failure
Self-Healing Agent and its Reliability Checker. It is written for the team to get
quickly productive and to reproduce the main results.

High-level goal
---------------
- Build a well-calibrated predictor of impending disk failure from Backblaze Drive Stats.
- Implement a healing policy (automatic actions like drain/replace) driven by calibrated risk
  and an explicit cost model.
- Validate the agent with a counterfactual "replay" reliability checker that simulates per-tome
  state using Backblaze topology and enforces safety invariants (parity budget, concurrency).
- Report a five-axis trust score: correctness, safety, timeliness, efficiency, stability.

Datasets
--------
- Primary: Backblaze Drive Stats Q1 2026 via Apache Iceberg on Backblaze B2 (no full download needed).
  - Compressed download: 1.3 GB; extracted CSVs ~12.02 GB.
  - We use DuckDB + Iceberg to query remotely and materialize compact parquet artifacts.
- Auxiliary: SMART-Z (OSF doi:10.17605/OSF.IO/24Y6G) for cross-fleet generalisation testing.
  - Long-format SMART file is large (195M rows); stream and pivot filtered SMART_IDs only.

Artifacts produced by notebooks (location: `notebooks/` and `DRIVE_ROOT`):
- `drive_registry.parquet`  — 1 row per drive (model, capacity, first/last seen, ever_failed).
- `cohort.parquet`         — all failed drives + stratified healthy sample (with sample_weight).
- `panel/panel_YYYY_MM.parquet` — per-drive daily telemetry for cohort, one file per month.
- `fleet_minimal/fleet_YYYY_MM.parquet` — minimal full-fleet per-day topology + failure (for replay).
- `features.parquet`       — leakage-tested modelling table with labels and engineered features.

Notebooks (Colab-ready)
-----------------------
1. `01_setup_and_data_access.ipynb`
   - Connects to Backblaze Iceberg via DuckDB. Extracts registry, cohort and per-month panel to Drive.
   - Warm-up: pulls Dec 2025 as lookback so 30d trailing windows in Jan are valid.
   - Safety: viability checks for topology population and per-drive time-series integrity.
2. `02_eda.ipynb`
   - Fleet-level analysis from registry (unbiased): AFR, bathtub curve, capacity mix.
   - SMART availability matrix by manufacturer (identifies vendor-conditional sensors).
   - Pre-failure trajectories and measured lead-time distribution (motivates horizon).
3. `03_feature_engineering.ipynb`
   - Label construction with censoring handling and distinct horizons (7/14/30d).
   - Strictly causal trailing-window features, anomaly-onset features, threshold margins.
   - Tome-context features from the full-fleet table and leakage/sanity assertions.

Workflow and responsibilities
-----------------------------
Phase A — Data & EDA (owner: Data / Backend)
- Create Drive registry and cohort via `01_setup_and_data_access.ipynb` on Colab (recommended).
- Run `02_eda.ipynb` to confirm vendor patterns, lead-time, censoring rates, and tome geometry.
- Decision point: if test-window failures are too few to give tight CIs, extend to earlier quarters.

Phase B — Feature engineering (owner: Feature)
- Run `03_feature_engineering.ipynb` to produce `features.parquet`.
- Ensure the three leakage tests pass:
  1. Core delta features have non-zero variance.
  2. Trailing features unchanged when truncating future rows (causality).
  3. Horizon labels are distinct and nested.

Phase C — Prediction & calibration (owner: ML)
- Baselines: SMART-threshold rule (any non-zero of critical counters) and logistic regression.
- Main model: LightGBM with GroupKFold by serial number inside a time-series split.
- Sequence model: 1D-CNN or GRU over 30-day windows (optional).
- Survival framing: discrete-time hazard or Cox model to handle censoring properly.
- Calibration: isotonic/Platt on the calibration fold; evaluate with Brier and ECE.

Phase D — Healing policy design (owner: Systems / ML)
- Action space: NO_OP, WATCH, MIGRATE_DATA (drain), REPLACE_DRIVE, CORDON_TOME.
- Policy = expected-cost minimisation using calibrated probabilities and an explicit cost model.
- Hard safety rules override policy: per-tome parity budget, concurrency caps, vault-level caps.
- All decisions emit structured audit records for traceability.

Phase E — Reliability checker & simulator (owner: Systems)
- Discrete-event simulator that initialises tome-state from `fleet_minimal` and replays days.
- Simulated failures use historical failure dates; agent actions change availability and consume rebuild bandwidth.
- Stress tests: sensor faults, distribution shifts, correlated cabinet events, adversarial tomes.
- Produce five-axis scorecard and per-axis breakdowns.

Phase F — Reporting and reproducibility (owner: All)
- Produce a reproducible pipeline with pinned `requirements.txt`, seed, and unit tests for leakage.
- Deliverables: notebook results, a scorecard report, dashboard (optional), and a paper-style write-up.

Resource & infra guidance
-------------------------
- Local disk is constrained (workspace had ~1.1–3.3 GB free). Do NOT download full extracted CSVs locally.
- Recommended: run notebooks in Google Colab with Drive mounted (`/content/drive/MyDrive/hpe_capstone`).
  - DuckDB + Iceberg reads directly from Backblaze B2 (read-only credentials are public).
  - Materialise compact parquet artifacts to Drive, one month per file (resumable).
- If large-scale training or faster IO is needed, use GCS + GCE / Vertex AI and store parquet in a bucket.
- Expected working sizes after materialisation (column-pruned, zstd compressed): 200–700 MB for panel per quarter depending on chosen columns; fleet minimal ~150–300 MB; features.parquet typically <500 MB.

How to run (quickstart)
-----------------------
1. Open Colab and mount Drive. Create folder `MyDrive/hpe_capstone`.
2. In Colab, open `notebooks/01_setup_and_data_access.ipynb`. Set env vars (optional) or use defaults.
3. Run the viability checks; if they pass, run the extraction cells — they are resumable per-month.
4. Run `02_eda.ipynb` and `03_feature_engineering.ipynb` in order.
5. After `features.parquet` exists, proceed to modeling (we will add `04_prediction.ipynb` next).

Key risks & mitigations
-----------------------
- Risk: accidental row-level sampling destroys time series. Mitigation: cohort by drive and unit tests assert median days/drive >> 1.
- Risk: vendor-conditional attributes mis-imputed. Mitigation: explicit missingness indicators and manufacturer feature; no fleet-mean imputation of vendor-only attributes.
- Risk: insufficient failures for tight CI. Mitigation: extend training window to previous quarters (Iceberg scan supports quarter-agnostic queries).
- Risk: topology sparsity. Mitigation: viability checks run early; simulator can fall back to conservative pseudo-tomes if needed.

Next immediate tasks
--------------------
1. Team: pull `notebooks/01_setup_and_data_access.ipynb` into Colab and run the viability checks (fast).
2. If viability OK, run the materialisation for Dec 2025–Mar 2026 (resumable; takes minutes per month).
3. Run `02_eda.ipynb` and confirm SMART availability & lead-time; decide whether to extend to 2025.
4. Run `03_feature_engineering.ipynb` and ensure all three leakage assertions pass.

Contact / owners
----------------
- Data / extraction: <your-data-person>@company (owner of notebook 01)
- Feature engineering: <your-feature-person>@company (owner of notebook 03)
- Modeling: <your-ml-person>@company (owner of notebook 04)
- Systems & reliability checker: <your-systems-person>@company

Notes
-----
- The full plan and engineering trade-offs are in `/Users/aryan/.cursor/plans/disk_failure_self-healing_agent_bfb58521.plan.md` in the agent plan store. The notebooks in `notebooks/` are the executable implementation of the plan's first three phases.

--- end of plan\n+
