# Home Credit Credit Risk | مخاطر الائتمان

**From probability of default to a business decision.** This project uses Home Credit applicant and credit-history data to rank default risk, compare classification models, and select an approval policy using an explicit financial cost matrix.

> **Portfolio headline:** the project reports ROC-AUC improving from **0.6882** to **0.7677** and a projected net portfolio value of approximately **$64.28M**. These are project-reported results; the notebook recalculates validation scores and portfolio value when run.

## Business problem

The Home Credit training set is imbalanced: approximately **8.07%** of applicants have `TARGET = 1` (payment difficulties). A standard probability cutoff of **0.50** is not necessarily the best lending decision because the costs of approving a defaulter and rejecting a reliable applicant differ.

The proposed cost matrix assigns the following value to each outcome:

| Actual outcome | Decision | Confusion matrix | Value per applicant |
|---|---|---:|---:|
| No default | Approve | TN | +$1,500 |
| No default | Reject | FP | −$3,000 |
| Default | Approve | FN | −$4,000 |
| Default | Reject | TP | $0 |

The objective is to select a threshold \(t\) that maximizes:

\[
\text{Net Value}(t) = 1500\,TN(t) - 3000\,FP(t) - 4000\,FN(t)
\]

This is an illustrative business scenario. The cost assumptions and portfolio-size scaling should be reviewed with business stakeholders; the estimate is not realized profit.

## Pipeline and methodology

1. **Data loading:** application training data, bureau history, and previous application history are read from `data/raw/`.
2. **Feature engineering:** customer-level bureau and previous-application aggregates are joined to the application table before modeling. Aggregations are grouped by applicant ID to retain one row per applicant. The current shared pipeline does **not** aggregate credit-card balance data.
3. **Preprocessing:** a stratified train/validation split is created. Median imputation is fit on the training split and then applied to validation data.
4. **Model benchmarking:** Logistic Regression, Decision Tree, Random Forest, and tuned XGBoost are compared using validation ROC-AUC.
5. **Decision optimization:** candidate probability thresholds are scored with the cost matrix. The notebook compares the 0.50 reference with the cost-optimal threshold and reports a scaled portfolio estimate.
6. **Explainability and export:** native XGBoost TreeSHAP contributions summarize global feature importance. The model, imputer, feature schema, and decision metadata are saved under `models/`.

### Reported model progression

| Model | Reported validation ROC-AUC |
|---|---:|
| Logistic Regression baseline | 0.6882 |
| Logistic Regression with engineered features | 0.6946 |
| Decision Tree | 0.7093 |
| Random Forest | 0.7439 |
| XGBoost before tuning | 0.7659 |
| Tuned XGBoost | **0.7677** |

The notebook recalculates these values on execution. Treat the table as the project's reported progression and confirm that the final published numbers come from one consistent split and run.

## Threshold and business impact

Project materials report two business-threshold values, **0.3083** and **0.1773**, across iterations. The notebook includes **0.1773** as the reference threshold and calculates the optimal threshold from the validation predictions and cost assumptions. Re-run the notebook and choose one canonical threshold before publishing headline results; threshold choice changes with the cost matrix and validation predictions.

The reported net portfolio value is approximately **$64.28M**. It is a projection based on scenario economics and a portfolio-size assumption, not a guaranteed or realized return. See the notebook's threshold simulation for the assumptions and freshly calculated value.

## Repository structure

```text
notebooks/main01.ipynb       Executive analysis and model comparison
src/data_loader.py         Loading application, bureau, and previous-application data
src/feature_engineering.py Shared feature construction and history aggregation
src/preprocessing.py       Imputation helpers
src/train.py               XGBoost training helper
src/evaluate.py            Business threshold evaluation
predict.py                  Inference entry point
models/                     Required inference artifacts for the dashboard
data/raw/                   Locally supplied dataset files (git-ignored)
```

## Installation and run

Use Python 3.12 or newer:

```bash
# Windows PowerShell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
# macOS/Linux: python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name project-1-py312 --display-name "Python 3.12 (.venv)"
```

Place `application_train.csv`, `bureau.csv`, and `previous_application.csv` in `data/raw/`, then open `notebooks/main01.ipynb` and run all cells from top to bottom. The notebook writes the trained model, imputer, feature names, and decision metadata to `models/`. The four inference artifacts in `models/` are included in Git so the deployed Streamlit app can load the model. Raw datasets remain excluded.

Run the dashboard from the project root. On Windows, call the project's Python directly so this works even when `python` is not on `PATH`:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

On macOS/Linux, activate `.venv` first and run `python -m streamlit run app.py`.

#### If port 8501 is already in use

Streamlit uses port `8501` by default. This message usually means another Streamlit process is already running. Reuse its browser tab and refresh it; closing the tab alone does not stop the server.

To restart the app on the default port, go to the terminal where Streamlit is running, press `Ctrl+C` once, then run the dashboard only once:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit prints the local URL and normally opens it in your browser. If you want a second instance while the first is still running, choose a different port, such as `8502`:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

Open `http://localhost:8502` on the computer. For phone access, use `http://<computer-LAN-IP>:8502` instead of port `8501` while that instance is running.

The dashboard supports Arabic and English and displays USD and EGP using the editable planning rate in the sidebar (it is not a live exchange quote).

### Open the dashboard from a phone

1. Keep the computer and phone on the same private local network/router. The computer can use Ethernet while the phone uses Wi-Fi.
2. On Windows, run `ipconfig` in PowerShell and note the IPv4 address for the active network adapter.
3. On the phone, open `http://<computer-LAN-IP>:8501` in a browser. Replace `<computer-LAN-IP>` with the address from `ipconfig` (for example, `http://192.168.1.20:8501`). If you started Streamlit on another port, use that port in the phone address too.
4. If Windows Firewall asks, allow access on **Private networks**. If the connection is still blocked, allow inbound TCP on the port used by Streamlit (`8501` by default) on the Private profile.

The computer's LAN address can change, so check `ipconfig` again if the link stops working. Keep the app on your local network; don't use public IP port forwarding to expose it to the internet.

For inference, call `predict_loan_default()` from `predict.py` with the application CSV. Pass `bureau_path` and `previous_path` when those histories are available so the inference feature set matches training. Credit-card histories are not currently supported by the shared feature pipeline.

## Responsible use and data

This repository is a portfolio project, not a validated lending system. ROC-AUC does not establish probability calibration, fairness, compliance, or suitability for real credit decisions. Before operational use, validate on out-of-time data, review costs and threshold choices with stakeholders, assess subgroup outcomes, and monitor performance.

Download the Home Credit Default Risk dataset from its official competition source and follow its terms of use. Raw data and generated model binaries are excluded from Git; do not commit applicant-level data or private credentials.





