from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.feature_engineering import build_features

st.set_page_config(
    page_title="Credit risk | Portfolio decisions",
    page_icon=":material/account_balance:",
    layout="wide",
)


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "credit_risk_xgb_model.pkl"
IMPUTER_PATH = MODEL_DIR / "imputer.pkl"
FEATURES_PATH = MODEL_DIR / "feature_names.pkl"
METADATA_PATH = MODEL_DIR / "decision_metadata.json"
DEFAULT_RATE = 50.0  # Example planning rate only; it is not a live exchange quote.
DEFAULT_METADATA = {
    "validation_roc_auc": 0.7677,
    "decision_threshold": 0.1773,
    "projected_portfolio_value": 64_280_000,
    "portfolio_size_assumption": 307_511,
    "cost_matrix_per_applicant": {"tn": 1_500, "fp": -300, "fn": -4_000, "tp": 0},
}

TEXT = {
    "العربية": {
        "brand": "مخاطر الائتمان",
        "sidebar_subtitle": "HOME CREDIT · قرارات المحفظة",
        "app_title": "مخاطر الائتمان | لوحة قرارات المحفظة",
        "tagline": "رؤية أوضح للمخاطر، وقرارات ائتمانية مبنية على البيانات.",
        "language": "اللغة / Language",
        "navigation": "التنقل",
        "overview": "نظرة عامة",
        "simulator": "محاكي العائد",
        "scoring": "تقييم العملاء",
        "fx_label": "سعر التحويل: 1 دولار = كام جنيه؟",
        "fx_help": "سعر يدوي للتخطيط فقط، وليس سعر صرف مباشرًا.",
        "fx_note": "سعر تخطيطي تجريبي وقابل للتعديل، وليس سعرًا مباشرًا.",
        "mobile_hint": "للموبايل: افتح عنوان IP المحلي للكمبيوتر على نفس الشبكة المحلية باستخدام المنفذ 8501.",
        "overview_title": "ملخص أداء المحفظة",
        "overview_intro": "نتائج النموذج والافتراضات المالية المحفوظة مع ملفات المشروع.",
        "auc": "دقة الترتيب (ROC-AUC)",
        "threshold": "حد القرار",
        "portfolio_value": "قيمة المحفظة التقديرية",
        "portfolio_size": "حجم المحفظة المفترض",
        "reported_value_note": "تقدير محفوظ في بيانات النموذج؛ لا يمثل ربحًا محققًا.",
        "cost_title": "الأثر المالي لكل حالة",
        "cost_intro": "التكاليف مأخوذة من بيانات النموذج. غيّر سعر التحويل من الشريط الجانبي.",
        "outcome": "الحالة",
        "value_usd": "القيمة لكل عميل (USD)",
        "value_egp": "القيمة لكل عميل (EGP)",
        "good_approved": "عميل ملتزم تمت الموافقة عليه",
        "good_rejected": "عميل ملتزم تم رفضه",
        "risky_approved": "عميل متعثر تمت الموافقة عليه",
        "risky_rejected": "عميل متعثر تم رفضه",
        "usd_chart": "الأثر بالدولار الأمريكي",
        "egp_chart": "الأثر بالجنيه المصري",
        "sim_title": "احسب أثر قرارات المحفظة",
        "sim_intro": "أدخل أعداد نتائج التحقق لديك لحساب قيمة السيناريو وفق مصفوفة التكلفة الحالية.",
        "count_tn": "ملتزمون تمت الموافقة عليهم (TN)",
        "count_fp": "ملتزمون تم رفضهم (FP)",
        "count_fn": "متعثرون تمت الموافقة عليهم (FN)",
        "count_tp": "متعثرون تم رفضهم (TP)",
        "calculate": "احسب قيمة السيناريو",
        "total_cases": "إجمالي الحالات",
        "net_value": "صافي القيمة بالدولار",
        "converted_value": "ما يعادل بالجنيه المصري",
        "scenario_note": "الحساب تقديري حسب الافتراضات المدخلة، ولا يمثل ربحًا فعليًا.",
        "score_title": "تقييم مخاطر العملاء",
        "score_intro": "ارفع بيانات طلبات العملاء بصيغة CSV لتقدير احتمال التعثر باستخدام النموذج المحفوظ.",
        "application_csv": "ملف طلبات العملاء (CSV)",
        "bureau_csv": "ملف السجل الائتماني السابق (اختياري)",
        "previous_csv": "ملف الطلبات السابقة (اختياري)",
        "score_button": "حلّل الطلبات",
        "threshold_control": "حد الرفض حسب درجة المخاطر",
        "threshold_help": "أي احتمال تعثر يساوي الحد أو يزيد عليه يصنف عالي المخاطر.",
        "missing_artifact": "ملفات النموذج غير مكتملة. شغّل الـnotebook لتدريب النموذج وحفظ ملفاته داخل models/.",
        "score_success": "اكتمل تقييم الطلبات.",
        "score_error": "تعذر تقييم الملف. راجع الأعمدة المطلوبة ونوع البيانات.",
        "applicant_id": "رقم العميل",
        "probability": "احتمال التعثر",
        "decision": "القرار",
        "approve": "موافقة مبدئية",
        "reject": "مراجعة / رفض",
        "high_risk": "طلبات عالية المخاطر",
        "lower_risk": "طلبات أقل مخاطرة",
        "download": "نزّل نتائج التقييم CSV",
        "preview_note": "عرض أول 5,000 صف كحد أقصى؛ ملف التنزيل يحتوي على كل النتائج.",
        "model_note": "أضف ملفي السجل السابق للحصول على ميزات التاريخ الائتماني عند توفرهما.",
        "formula_title": "طريقة الحساب",
        "formula_text": "صافي القيمة = (TN × قيمة الموافقة الصحيحة) + (FP × تكلفة الرفض) + (FN × تكلفة التعثر المقبول) + (TP × قيمة الرفض الصحيح).",
        "no_rows": "لم يتم رفع ملف بعد.",
    },
    "English": {
        "brand": "CREDIT RISK",
        "sidebar_subtitle": "HOME CREDIT · PORTFOLIO DECISIONS",
        "app_title": "Credit risk | Portfolio decision dashboard",
        "tagline": "A clearer view of risk, with data-informed credit decisions.",
        "language": "اللغة / Language",
        "navigation": "Navigation",
        "overview": "Overview",
        "simulator": "Value simulator",
        "scoring": "Applicant scoring",
        "fx_label": "Exchange rate: EGP per 1 USD",
        "fx_help": "Manual planning rate; this is not a live exchange quote.",
        "fx_note": "Editable example planning rate, not a live quote.",
        "mobile_hint": "Phone access: open the computer's LAN IP on the same local network at port 8501.",
        "overview_title": "Portfolio performance snapshot",
        "overview_intro": "Model results and financial assumptions saved with the project artifacts.",
        "auc": "Ranking quality (ROC-AUC)",
        "threshold": "Decision threshold",
        "portfolio_value": "Estimated portfolio value",
        "portfolio_size": "Assumed portfolio size",
        "reported_value_note": "Estimate saved with the model; this is not realized profit.",
        "cost_title": "Financial impact per outcome",
        "cost_intro": "Costs come from the model metadata. Change the exchange rate in the sidebar.",
        "outcome": "Outcome",
        "value_usd": "Value per applicant (USD)",
        "value_egp": "Value per applicant (EGP)",
        "good_approved": "Reliable applicant approved",
        "good_rejected": "Reliable applicant rejected",
        "risky_approved": "Defaulting applicant approved",
        "risky_rejected": "Defaulting applicant rejected",
        "usd_chart": "Impact in US dollars",
        "egp_chart": "Impact in Egyptian pounds",
        "sim_title": "Estimate portfolio decision value",
        "sim_intro": "Enter your validation outcome counts to calculate scenario value using the current cost matrix.",
        "count_tn": "Reliable applicants approved (TN)",
        "count_fp": "Reliable applicants rejected (FP)",
        "count_fn": "Defaulting applicants approved (FN)",
        "count_tp": "Defaulting applicants rejected (TP)",
        "calculate": "Calculate scenario value",
        "total_cases": "Total cases",
        "net_value": "Net value in US dollars",
        "converted_value": "Equivalent in Egyptian pounds",
        "scenario_note": "This is an estimate from the entered assumptions, not realized profit.",
        "score_title": "Applicant risk scoring",
        "score_intro": "Upload applicant applications as a CSV to estimate default risk using the saved model.",
        "application_csv": "Applicant applications (CSV)",
        "bureau_csv": "Previous credit history (optional)",
        "previous_csv": "Previous applications (optional)",
        "score_button": "Score applicants",
        "threshold_control": "High-risk rejection threshold",
        "threshold_help": "Applicants at or above this estimated default probability are marked high risk.",
        "missing_artifact": "Model artifacts are incomplete. Run the notebook to train the model and save its files under models/.",
        "score_success": "Applicant scoring is complete.",
        "score_error": "Could not score this file. Check the required columns and data types.",
        "applicant_id": "Applicant ID",
        "probability": "Default probability",
        "decision": "Decision",
        "approve": "Provisional approval",
        "reject": "Review / reject",
        "high_risk": "High-risk applications",
        "lower_risk": "Lower-risk applications",
        "download": "Download scoring results CSV",
        "preview_note": "Showing up to 5,000 rows; the download contains all results.",
        "model_note": "Add previous credit-history files when available to include those engineered features.",
        "formula_title": "Calculation",
        "formula_text": "Net value = (TN × good approval value) + (FP × rejection cost) + (FN × accepted-default cost) + (TP × correct rejection value).",
        "no_rows": "No file has been uploaded yet.",
    },
}


def format_usd(value: float) -> str:
    return f"-${abs(value):,.0f}" if value < 0 else f"${value:,.0f}"


def format_egp(value: float) -> str:
    return f"EGP -{abs(value):,.0f}" if value < 0 else f"EGP {value:,.0f}"


def compact_money(value: float, currency: str) -> str:
    for scale, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if abs(value) >= scale:
            amount = abs(value) / scale
            sign = "-" if value < 0 else ""
            if currency == "USD":
                return f"{sign}${amount:,.2f}{suffix}"
            return f"EGP {sign}{amount:,.2f}{suffix}"
    return format_usd(value) if currency == "USD" else format_egp(value)


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    if METADATA_PATH.is_file():
        try:
            return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return DEFAULT_METADATA.copy()


@st.cache_resource(show_spinner=False)
def load_model_bundle():
    return (
        joblib.load(MODEL_PATH),
        joblib.load(IMPUTER_PATH),
        joblib.load(FEATURES_PATH),
    )


def build_cost_table(costs: dict, exchange_rate: float, labels: dict) -> pd.DataFrame:
    rows = [
        (labels["good_approved"], float(costs.get("tn", 1_500))),
        (labels["good_rejected"], float(costs.get("fp", -300))),
        (labels["risky_approved"], float(costs.get("fn", -4_000))),
        (labels["risky_rejected"], float(costs.get("tp", 0))),
    ]
    return pd.DataFrame(
        {
            labels["outcome"]: [row[0] for row in rows],
            labels["value_usd"]: [row[1] for row in rows],
            labels["value_egp"]: [row[1] * exchange_rate for row in rows],
        }
    )


def score_applications(
    application_df: pd.DataFrame,
    bureau_df: pd.DataFrame | None,
    previous_df: pd.DataFrame | None,
    model,
    imputer,
    feature_names: list[str],
    threshold: float,
) -> pd.DataFrame:
    identifiers = (
        application_df["SK_ID_CURR"].reset_index(drop=True)
        if "SK_ID_CURR" in application_df.columns
        else pd.Series(application_df.index, name="SK_ID_CURR")
    )
    engineered = build_features(application_df, bureau_df=bureau_df, prev_df=previous_df)
    engineered = engineered.drop(columns=["TARGET"], errors="ignore")
    model_input = engineered.reindex(columns=feature_names, fill_value=0)
    model_input = model_input.apply(pd.to_numeric, errors="coerce")
    imputed = imputer.transform(model_input)

    probabilities = []
    batch_size = 10_000
    for start in range(0, len(imputed), batch_size):
        probabilities.extend(model.predict_proba(imputed[start : start + batch_size])[:, 1])
    probabilities = np.asarray(probabilities)
    high_risk = probabilities >= threshold
    return pd.DataFrame(
        {
            "SK_ID_CURR": identifiers,
            "Default probability": np.round(probabilities, 4),
            "Decision threshold": threshold,
            "High risk": high_risk,
        }
    )


metadata = load_metadata()
text_language = st.sidebar.selectbox(
    "Language / اللغة", ["العربية", "English"], index=0, key="language"
)
t = TEXT[text_language]
alignment = "right" if text_language == "العربية" else "left"

with st.sidebar:
    st.markdown(f":material/account_balance:  **{t['brand']}**")
    st.caption(t["sidebar_subtitle"])
    page_key = st.selectbox(
        t["navigation"],
        options=["overview", "simulator", "scoring"],
        format_func=lambda key: t[key],
        key="navigation",
    )
    exchange_rate = st.number_input(
        t["fx_label"],
        min_value=0.01,
        max_value=100_000.0,
        value=DEFAULT_RATE,
        step=0.25,
        format="%.2f",
        help=t["fx_help"],
        key="usd_egp_rate",
    )
    st.caption(f"1 USD = {exchange_rate:,.2f} EGP")
    st.caption(t["fx_note"])
    st.caption(t["mobile_hint"])

st.title(t["app_title"], text_alignment=alignment)
st.caption(t["tagline"])

costs = metadata.get("cost_matrix_per_applicant", DEFAULT_METADATA["cost_matrix_per_applicant"])
auc = float(metadata.get("validation_roc_auc", DEFAULT_METADATA["validation_roc_auc"]))
default_threshold = float(metadata.get("decision_threshold", DEFAULT_METADATA["decision_threshold"]))
portfolio_value = float(metadata.get("projected_portfolio_value", DEFAULT_METADATA["projected_portfolio_value"]))
portfolio_size = int(metadata.get("portfolio_size_assumption", DEFAULT_METADATA["portfolio_size_assumption"]))

if page_key == "overview":
    st.header(t["overview_title"], text_alignment=alignment)
    st.write(t["overview_intro"])

    auc_col, threshold_col, value_col, size_col = st.columns(4)
    auc_col.metric(t["auc"], f"{auc:.4f}", border=True)
    threshold_col.metric(t["threshold"], f"{default_threshold:.4f}", border=True)
    value_col.metric(t["portfolio_value"], compact_money(portfolio_value, "USD"), border=True)
    value_col.caption(f"≈ {compact_money(portfolio_value * exchange_rate, 'EGP')}")
    size_col.metric(t["portfolio_size"], f"{portfolio_size:,}", border=True)

    st.caption(t["reported_value_note"])
    st.subheader(t["cost_title"])
    st.caption(t["cost_intro"])
    cost_table = build_cost_table(costs, exchange_rate, t)
    st.dataframe(cost_table, hide_index=True, width="stretch", alt=t["cost_title"])

    usd_chart_data = cost_table[[t["outcome"], t["value_usd"]]]
    with st.container(border=True):
        st.subheader(t["usd_chart"], text_alignment=alignment)
        st.bar_chart(
            usd_chart_data,
            x=t["outcome"],
            y=t["value_usd"],
            horizontal=True,
            width="stretch",
            height=380,
            alt=t["usd_chart"],
        )

    egp_chart_data = cost_table[[t["outcome"], t["value_egp"]]]
    with st.container(border=True):
        st.subheader(t["egp_chart"], text_alignment=alignment)
        st.bar_chart(
            egp_chart_data,
            x=t["outcome"],
            y=t["value_egp"],
            horizontal=True,
            width="stretch",
            height=380,
            alt=t["egp_chart"],
        )

elif page_key == "simulator":
    st.header(t["sim_title"], text_alignment=alignment)
    st.write(t["sim_intro"])
    tn_cost = float(costs.get("tn", 1_500))
    fp_cost = float(costs.get("fp", -300))
    fn_cost = float(costs.get("fn", -4_000))
    tp_cost = float(costs.get("tp", 0))

    with st.form("portfolio_value_simulator"):
        left, right = st.columns(2)
        with left:
            tn_count = st.number_input(t["count_tn"], min_value=0, value=0, step=1)
            fp_count = st.number_input(t["count_fp"], min_value=0, value=0, step=1)
        with right:
            fn_count = st.number_input(t["count_fn"], min_value=0, value=0, step=1)
            tp_count = st.number_input(t["count_tp"], min_value=0, value=0, step=1)
        submitted = st.form_submit_button(t["calculate"], type="primary", width="stretch")

    if submitted:
        total_value = tn_count * tn_cost + fp_count * fp_cost + fn_count * fn_cost + tp_count * tp_cost
        total_cases = tn_count + fp_count + fn_count + tp_count
        usd_col, egp_col, cases_col = st.columns(3)
        usd_col.metric(t["net_value"], format_usd(total_value), border=True)
        egp_col.metric(t["converted_value"], format_egp(total_value * exchange_rate), border=True)
        cases_col.metric(t["total_cases"], f"{total_cases:,}", border=True)
        st.caption(t["scenario_note"])

    st.subheader(t["cost_title"])
    st.dataframe(
        build_cost_table(costs, exchange_rate, t),
        hide_index=True,
        width="stretch",
        alt=t["cost_title"],
    )
    with st.expander(t["formula_title"]):
        st.write(t["formula_text"])

else:
    st.header(t["score_title"], text_alignment=alignment)
    st.write(t["score_intro"])
    threshold = st.slider(
        t["threshold_control"],
        min_value=0.0,
        max_value=1.0,
        value=min(max(default_threshold, 0.0), 1.0),
        step=0.01,
        help=t["threshold_help"],
        format="%.2f",
    )
    application_file = st.file_uploader(t["application_csv"], type=["csv"], key="application_csv")
    bureau_col, previous_col = st.columns(2)
    with bureau_col:
        bureau_file = st.file_uploader(t["bureau_csv"], type=["csv"], key="bureau_csv")
    with previous_col:
        previous_file = st.file_uploader(t["previous_csv"], type=["csv"], key="previous_csv")
    st.caption(t["model_note"])

    artifacts_ready = all(path.is_file() for path in (MODEL_PATH, IMPUTER_PATH, FEATURES_PATH))
    if not artifacts_ready:
        st.warning(t["missing_artifact"])

    upload_fingerprint = (
        (application_file.name, application_file.size) if application_file else None,
        (bureau_file.name, bureau_file.size) if bureau_file else None,
        (previous_file.name, previous_file.size) if previous_file else None,
        threshold,
        text_language,
    )
    if st.session_state.get("scored_input_fingerprint") != upload_fingerprint:
        st.session_state.pop("scored_results", None)
        st.session_state.pop("scored_input_fingerprint", None)

    if st.button(t["score_button"], type="primary", width="stretch", disabled=not (application_file and artifacts_ready)):
        try:
            with st.spinner(t["score_intro"]):
                application_df = pd.read_csv(io.BytesIO(application_file.getvalue()))
                bureau_df = pd.read_csv(io.BytesIO(bureau_file.getvalue())) if bureau_file else None
                previous_df = pd.read_csv(io.BytesIO(previous_file.getvalue())) if previous_file else None
                model, imputer, feature_names = load_model_bundle()
                results = score_applications(
                    application_df,
                    bureau_df,
                    previous_df,
                    model,
                    imputer,
                    feature_names,
                    threshold,
                )
                results = results.rename(
                    columns={
                        "SK_ID_CURR": t["applicant_id"],
                        "Default probability": t["probability"],
                        "Decision threshold": t["threshold"],
                        "High risk": t["decision"],
                    }
                )
                results[t["decision"]] = np.where(
                    results[t["decision"]], t["reject"], t["approve"]
                )
                st.session_state["scored_results"] = results
                st.session_state["scored_input_fingerprint"] = upload_fingerprint
        except Exception as error:
            st.error(f"{t['score_error']} ({error})")

    scored_results = st.session_state.get("scored_results")
    if scored_results is not None:
        risky_count = int((scored_results[t["decision"]] == t["reject"]).sum())
        lower_risk_count = len(scored_results) - risky_count
        scored_col, high_col, lower_col = st.columns(3)
        scored_col.metric(t["total_cases"], f"{len(scored_results):,}", border=True)
        high_col.metric(t["high_risk"], f"{risky_count:,}", border=True)
        lower_col.metric(t["lower_risk"], f"{lower_risk_count:,}", border=True)
        st.success(t["score_success"])
        st.caption(t["preview_note"])
        st.dataframe(scored_results.head(5_000), hide_index=True, width="stretch", alt=t["score_title"])
        st.download_button(
            t["download"],
            data=scored_results.to_csv(index=False).encode("utf-8-sig"),
            file_name="credit_risk_predictions.csv",
            mime="text/csv",
            width="stretch",
        )
    elif application_file is None:
        st.caption(t["no_rows"])

