# app.py — SME Credit Scoring Streamlit App
# Run with: streamlit run app.py

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SME Credit Scoring",
    page_icon="🏦",
    layout="wide",
)

# ── Load saved artefacts ───────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    model      = joblib.load("sme_credit_model.pkl")
    calibrator = joblib.load("calibrator.pkl")
    scaler     = joblib.load("scaler.pkl")
    columns    = joblib.load("feature_columns.pkl")   # saved in notebook — see below
    return model, calibrator, scaler, columns

model, calibrator, scaler, FEATURE_COLUMNS = load_artefacts()


# ── Helper functions (mirror the notebook exactly) ─────────────────────────────

def engineer_features(d: dict) -> pd.DataFrame:
    """Apply the same feature engineering used during training."""
    row = pd.DataFrame([d])

    row["cashflow_net"]            = row["mpesa_avg_monthly_inflow"] - row["mpesa_avg_monthly_outflow"]
    row["cashflow_ratio"]          = row["mpesa_avg_monthly_inflow"] / (row["mpesa_avg_monthly_outflow"] + 1)
    row["loan_to_sales_ratio"]     = row["loan_amount_kes"] / (row["pos_monthly_sales"] + 1)
    row["credit_pressure"]         = row["loan_amount_kes"] / (row["mpesa_avg_monthly_inflow"] + 1)
    row["business_activity_score"] = row["num_employees"] * row["mpesa_txn_count_monthly"]
    row["revenue_per_employee"]    = row["pos_monthly_sales"] / (row["num_employees"] + 1)
    row["business_maturity"]       = row["years_in_operation"] * row["num_employees"]
    row["digital_engagement"]      = row["social_media_active"] + (row["online_ads_spend_kes"] > 0).astype(int)
    row["utility_risk_score"]      = row["electricity_payment_delays"]
    row["high_utility_risk"]       = (row["electricity_payment_delays"] > 5).astype(int)
    row["expense_pressure"]        = (
        (row["electricity_bill_avg"] + row["water_bill_avg"]) /
        (row["mpesa_avg_monthly_inflow"] + 1)
    )

    drop_cols = ["electricity_payment_delays", "social_media_active"]
    row.drop(columns=[c for c in drop_cols if c in row.columns], inplace=True)

    row = pd.get_dummies(row, columns=["sector"], drop_first=True)
    row = row.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return row


def compute_credit_score(prob, min_score=300, max_score=850):
    """Convert default probability to a 300–850 credit score."""
    return max_score - (prob * (max_score - min_score))


def risk_category(score):
    if score >= 750: return "Very Low Risk"
    elif score >= 650: return "Low Risk"
    elif score >= 550: return "Medium Risk"
    elif score >= 450: return "High Risk"
    else: return "Very High Risk"


def lending_decision(score):
    if score >= 720: return "APPROVE — Premium Loan"
    elif score >= 640: return "APPROVE — Standard Loan"
    elif score >= 560: return "REVIEW — Conditional Approval"
    else: return "REJECT — High Risk"


def flag_rules(row: dict) -> list:
    """Apply business-rule flags."""
    flags = []
    if row["credit_score"] < 450:
        flags.append("VERY_LOW_SCORE")
    if row["electricity_payment_delays"] >= 6:
        flags.append("HIGH_UTILITY_RISK")
    if row["loan_amount_kes"] / (row["pos_monthly_sales"] + 1) > 50:
        flags.append("EXCESSIVE_LEVERAGE")
    if row["mpesa_txn_count_monthly"] < 50:
        flags.append("LOW_ACTIVITY")
    if row["interest_rate_pct"] > 25:
        flags.append("HIGH_INTEREST_RISK")
    return flags


def final_decision(score: float, flags: list) -> str:
    """Combine model score + flags into a final lending decision."""
    if "VERY_LOW_SCORE"    in flags: return "REJECT — Credit Score Too Low"
    if "EXCESSIVE_LEVERAGE" in flags: return "REJECT — Overleveraged Business"
    if "HIGH_UTILITY_RISK" in flags: return "REJECT — High Utility Risk"
    if score < 600: return "REVIEW — Manual Assessment Required"
    if score < 700: return "APPROVE — Standard Loan"
    return "APPROVE — Premium Loan"


def score_application(inputs: dict) -> dict:
    """Full end-to-end scoring pipeline."""
    row        = engineer_features(inputs)
    row_scaled = scaler.transform(row)

    raw_prob = model.predict_proba(row_scaled)[:, 1][0]
    cal_prob = calibrator.predict_proba([[raw_prob]])[:, 1][0]
    score    = compute_credit_score(cal_prob)
    flags    = flag_rules({**inputs, "credit_score": score})
    final    = final_decision(score, flags)

    return {
        "raw_probability"       : round(float(raw_prob), 4),
        "calibrated_probability": round(float(cal_prob), 4),
        "credit_score"          : round(float(score), 1),
        "risk_category"         : risk_category(score),
        "lending_decision"      : lending_decision(score),
        "flags"                 : flags,
        "final_decision"        : final,
    }


# ── Colour helpers ─────────────────────────────────────────────────────────────

DECISION_COLOURS = {
    "APPROVE — Premium Loan"        : "#2ca02c",
    "APPROVE — Standard Loan"       : "#98df8a",
    "REVIEW — Conditional Approval" : "#ff7f0e",
    "REJECT — High Risk"            : "#d62728",
    "REJECT — Credit Score Too Low" : "#d62728",
    "REJECT — Overleveraged Business": "#d62728",
    "REJECT — High Utility Risk"    : "#d62728",
    "REVIEW — Manual Assessment Required": "#ff7f0e",
    "REVIEW — Conditional Approval" : "#ff7f0e",
}

RISK_COLOURS = {
    "Very Low Risk": "#2ca02c",
    "Low Risk"     : "#98df8a",
    "Medium Risk"  : "#ff7f0e",
    "High Risk"    : "#d62728",
    "Very High Risk": "#8B0000",
}

FLAG_DESCRIPTIONS = {
    "VERY_LOW_SCORE"     : "⚠️ Credit score below minimum threshold (< 450)",
    "HIGH_UTILITY_RISK"  : "⚠️ 6+ electricity payment delays detected",
    "EXCESSIVE_LEVERAGE" : "⚠️ Loan is more than 50× monthly POS sales",
    "LOW_ACTIVITY"       : "⚠️ Fewer than 50 M-Pesa transactions per month",
    "HIGH_INTEREST_RISK" : "⚠️ Interest rate above 25% — high cost of debt",
}


# ── UI ─────────────────────────────────────────────────────────────────────────

st.title("🏦 SME Credit Scoring System")
st.markdown(
    "Enter the SME's details below and click **Score Application** "
    "to get an instant credit assessment."
)
st.divider()

# ── Input form ─────────────────────────────────────────────────────────────────
with st.form("sme_form"):

    st.subheader("📋 Business Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        sector = st.selectbox(
            "Sector",
            ["Manufacturing", "Retail", "Transport", "Wholesale",
             "Construction", "Hospitality", "Services", "ICT",
             "Informal", "Agribusiness"]
        )
        years_in_operation = st.number_input(
            "Years in Operation", min_value=0.5, max_value=50.0,
            value=5.0, step=0.5
        )
        num_employees = st.number_input(
            "Number of Employees", min_value=1, max_value=500,
            value=8, step=1
        )

    with col2:
        loan_amount_kes = st.number_input(
            "Loan Amount (KES)", min_value=5_000, max_value=25_000_000,
            value=500_000, step=10_000
        )
        loan_term_months = st.selectbox(
            "Loan Term (months)", [6, 12, 18, 24, 36, 48, 60], index=3
        )
        interest_rate_pct = st.number_input(
            "Interest Rate (%)", min_value=6.0, max_value=36.6,
            value=18.5, step=0.5
        )

    with col3:
        pos_monthly_sales = st.number_input(
            "Monthly POS Sales (KES)", min_value=0, max_value=5_000_000,
            value=200_000, step=5_000
        )
        pos_txn_volume = st.number_input(
            "Monthly POS Transaction Volume", min_value=0, max_value=5_000,
            value=350, step=10
        )

    st.divider()
    st.subheader("📱 M-Pesa Activity")
    col4, col5, col6 = st.columns(3)

    with col4:
        mpesa_avg_monthly_inflow = st.number_input(
            "Avg Monthly Inflow (KES)", min_value=1_000, max_value=3_000_000,
            value=150_000, step=5_000
        )
        mpesa_avg_monthly_outflow = st.number_input(
            "Avg Monthly Outflow (KES)", min_value=0, max_value=3_000_000,
            value=90_000, step=5_000
        )

    with col5:
        mpesa_txn_count_monthly = st.number_input(
            "Monthly Transaction Count", min_value=0, max_value=2_000,
            value=200, step=10
        )
        mpesa_savings_ratio = st.slider(
            "M-Pesa Savings Ratio", min_value=0.0, max_value=1.0,
            value=0.35, step=0.01
        )

    with col6:
        avg_monthly_data_spend = st.number_input(
            "Avg Monthly Data Spend (KES)", min_value=0, max_value=50_000,
            value=1_500, step=100
        )
        avg_monthly_airtime_spend = st.number_input(
            "Avg Monthly Airtime Spend (KES)", min_value=0, max_value=20_000,
            value=800, step=50
        )

    st.divider()
    st.subheader("🔌 Utility Bills & Digital Presence")
    col7, col8, col9 = st.columns(3)

    with col7:
        electricity_bill_avg = st.number_input(
            "Avg Electricity Bill (KES)", min_value=0, max_value=200_000,
            value=5_000, step=500
        )
        water_bill_avg = st.number_input(
            "Avg Water Bill (KES)", min_value=0, max_value=50_000,
            value=2_000, step=200
        )

    with col8:
        electricity_payment_delays = st.number_input(
            "Electricity Payment Delays (#)", min_value=0, max_value=30,
            value=1, step=1
        )
        online_ads_spend_kes = st.number_input(
            "Monthly Online Ads Spend (KES)", min_value=0, max_value=200_000,
            value=3_000, step=500
        )

    with col9:
        social_media_active = st.selectbox(
            "Active on Social Media?", [1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    st.divider()
    submitted = st.form_submit_button("🔍 Score Application", use_container_width=True)


# ── Results ────────────────────────────────────────────────────────────────────
if submitted:
    inputs = {
        "sector"                    : sector,
        "years_in_operation"        : years_in_operation,
        "num_employees"             : num_employees,
        "loan_amount_kes"           : loan_amount_kes,
        "loan_term_months"          : loan_term_months,
        "interest_rate_pct"         : interest_rate_pct,
        "mpesa_avg_monthly_inflow"  : mpesa_avg_monthly_inflow,
        "mpesa_avg_monthly_outflow" : mpesa_avg_monthly_outflow,
        "mpesa_txn_count_monthly"   : mpesa_txn_count_monthly,
        "mpesa_savings_ratio"       : mpesa_savings_ratio,
        "electricity_payment_delays": electricity_payment_delays,
        "electricity_bill_avg"      : electricity_bill_avg,
        "water_bill_avg"            : water_bill_avg,
        "social_media_active"       : social_media_active,
        "online_ads_spend_kes"      : online_ads_spend_kes,
        "avg_monthly_data_spend"    : avg_monthly_data_spend,
        "avg_monthly_airtime_spend" : avg_monthly_airtime_spend,
        "pos_monthly_sales"         : pos_monthly_sales,
        "pos_txn_volume"            : pos_txn_volume,
    }

    with st.spinner("Scoring application..."):
        result = score_application(inputs)

    st.divider()
    st.subheader("📊 Credit Assessment Results")

    # ── Top metrics row ──
    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(
            label="Credit Score",
            value=f"{result['credit_score']:.0f} / 850",
        )

    with m2:
        st.metric(
            label="Default Probability",
            value=f"{result['calibrated_probability']:.1%}",
        )

    with m3:
        st.metric(
            label="Risk Category",
            value=result["risk_category"],
        )

    # ── Credit score gauge bar ──
    score_pct = (result["credit_score"] - 300) / 550   # normalise to 0–1
    bar_colour = RISK_COLOURS.get(result["risk_category"], "#aaa")

    st.markdown("**Credit Score Gauge**")
    st.progress(float(score_pct))
    st.markdown(
        f"<p style='color:{bar_colour}; font-weight:bold; font-size:14px;'>"
        f"Score {result['credit_score']:.0f} — {result['risk_category']}</p>",
        unsafe_allow_html=True
    )

    # ── Final decision banner ──
    decision_colour = DECISION_COLOURS.get(result["final_decision"], "#aaa")
    st.markdown(
        f"""
        <div style='
            background-color:{decision_colour}22;
            border-left: 6px solid {decision_colour};
            padding: 16px 20px;
            border-radius: 6px;
            margin-top: 12px;
        '>
            <h3 style='color:{decision_colour}; margin:0;'>
                {result["final_decision"]}
            </h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── Flags ──
    if result["flags"]:
        st.markdown("")
        st.markdown("**🚩 Risk Flags Triggered**")
        for flag in result["flags"]:
            st.warning(FLAG_DESCRIPTIONS.get(flag, flag))
    else:
        st.success("✅ No risk flags triggered.")

    # ── Detail expander ──
    with st.expander("🔎 Full Scoring Detail"):
        detail_df = pd.DataFrame([{
            "Raw Default Probability"       : result["raw_probability"],
            "Calibrated Default Probability": result["calibrated_probability"],
            "Credit Score"                  : result["credit_score"],
            "Risk Category"                 : result["risk_category"],
            "Lending Decision (model)"      : result["lending_decision"],
            "Final Decision (model + rules)": result["final_decision"],
            "Flags"                         : ", ".join(result["flags"]) if result["flags"] else "None",
        }]).T
        detail_df.columns = ["Value"]
        st.dataframe(detail_df, use_container_width=True)

    # ── Derived features expander ──
    with st.expander("⚙️ Computed Financial Ratios"):
        cashflow_net   = mpesa_avg_monthly_inflow - mpesa_avg_monthly_outflow
        cashflow_ratio = mpesa_avg_monthly_inflow / (mpesa_avg_monthly_outflow + 1)
        loan_to_sales  = loan_amount_kes / (pos_monthly_sales + 1)
        credit_pressure = loan_amount_kes / (mpesa_avg_monthly_inflow + 1)
        expense_pressure = (electricity_bill_avg + water_bill_avg) / (mpesa_avg_monthly_inflow + 1)

        ratios_df = pd.DataFrame({
            "Ratio"      : [
                "Net Cash Flow (KES)", "Cash Flow Ratio",
                "Loan-to-Sales Ratio", "Credit Pressure",
                "Expense Pressure"
            ],
            "Value"      : [
                f"KES {cashflow_net:,.0f}",
                f"{cashflow_ratio:.2f}",
                f"{loan_to_sales:.2f}×",
                f"{credit_pressure:.2f}×",
                f"{expense_pressure:.2%}",
            ],
            "Interpretation": [
                "Positive = healthy cash buffer" if cashflow_net > 0 else "Negative = outflows exceed inflows",
                "Above 1.0 = more inflow than outflow",
                "Below 10× is generally safer",
                "Below 5× is generally safer",
                "Below 10% of income on utilities is healthy",
            ]
        })
        st.dataframe(ratios_df, use_container_width=True, hide_index=True)
