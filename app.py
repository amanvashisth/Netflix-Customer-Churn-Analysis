import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Netflix Churn Predictor",
    page_icon="🎬",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = "Models/XGBoost_model.pkl"

    st.write("Model exists:", os.path.exists(model_path))

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    return model


try:
    model = load_model()

except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("🎬 Netflix Customer Churn Predictor")

st.markdown(
    """
    Predict whether a Netflix customer is likely to churn based on
    their demographic information, account details, and engagement behavior.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("About the Model")

st.sidebar.info(
    """
    This application uses a machine learning model trained on
    Netflix customer behavior data.

    The model analyzes customer engagement, account information,
    viewing activity, ratings, and login activity to estimate
    churn probability.
    """
)

st.sidebar.markdown("### Model Pipeline")

st.sidebar.write(
    """
    • Data preprocessing  
    • Feature encoding  
    • Feature scaling  
    • Machine learning model  
    • Churn probability
    """
)


# ============================================================
# CUSTOMER INFORMATION
# ============================================================

st.header("👤 Customer Information")

col1, col2, col3 = st.columns(3)


# -------------------- COLUMN 1 --------------------

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    region = st.selectbox(
        "Region",
        [
            "North",
            "South",
            "East",
            "West",
            "Central",
            "Northeast"
        ]
    )

    subscription_type = st.selectbox(
        "Subscription Type",
        [
            "Basic",
            "Standard",
            "Premium"
        ]
    )


# -------------------- COLUMN 2 --------------------

with col2:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Credit Card",
            "Debit Card",
            "PayPal",
            "UPI"
        ]
    )

    primary_device = st.selectbox(
        "Primary Device",
        [
            "Mobile",
            "Tablet",
            "Laptop",
            "Smart TV"
        ]
    )

    account_age_months = st.number_input(
        "Account Age (Months)",
        min_value=1,
        max_value=200,
        value=24,
        step=1
    )

    favorite_genre = st.selectbox(
        "Favorite Genre",
        [
            "Action",
            "Comedy",
            "Drama",
            "Horror",
            "Romance",
            "Sci-Fi",
            "Documentary"
        ]
    )


# -------------------- COLUMN 3 --------------------

with col3:

    time_of_day = st.selectbox(
        "Preferred Viewing Time",
        [
            "Morning",
            "Afternoon",
            "Evening",
            "Night"
        ]
    )

    recommendation_source = st.selectbox(
        "Recommendation Source",
        [
            "Algorithm",
            "Friends",
            "Search",
            "Trending",
            "External"
        ]
    )

    session_count = st.number_input(
        "Session Count",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )

    watch_sessions_per_week = st.number_input(
        "Watch Sessions per Week",
        min_value=1,
        max_value=30,
        value=5,
        step=1
    )


st.divider()


# ============================================================
# ENGAGEMENT INFORMATION
# ============================================================

st.header("📊 Customer Engagement")

col1, col2, col3 = st.columns(3)


with col1:

    avg_watch_time_minutes_per_week = st.number_input(
        "Average Watch Time (Minutes/Week)",
        min_value=0,
        max_value=2000,
        value=250,
        step=5
    )

    completion_rate = st.slider(
        "Completion Rate (%)",
        min_value=0,
        max_value=100,
        value=75
    )


with col2:

    avg_rating_given = st.slider(
        "Average Rating Given",
        min_value=1.0,
        max_value=5.0,
        value=4.0,
        step=1.0
    )

    app_rating = st.slider(
        "App Rating",
        min_value=1.0,
        max_value=5.0,
        value=4.0,
        step=1.0
    )


with col3:

    recommendation_click_rate = st.slider(
        "Recommendation Click Rate (%)",
        min_value=0,
        max_value=100,
        value=40
    )

    days_since_last_login = st.number_input(
        "Days Since Last Login",
        min_value=0,
        max_value=100,
        value=5,
        step=1
    )


st.divider()


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.header("🔮 Churn Prediction")

predict_button = st.button(
    "Predict Customer Churn",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # Create input DataFrame
    # --------------------------------------------------------

    input_data = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "region": [region],
        "subscription_type": [subscription_type],
        "payment_method": [payment_method],
        "primary_device": [primary_device],
        "account_age_months": [account_age_months],
        "favorite_genre": [favorite_genre],
        "time_of_day": [time_of_day],
        "recommendation_source": [recommendation_source],
        "session_count": [session_count],
        "avg_watch_time_minutes_per_week": [
            avg_watch_time_minutes_per_week
        ],
        "watch_sessions_per_week": [
            watch_sessions_per_week
        ],
        "completion_rate": [completion_rate],
        "avg_rating_given": [avg_rating_given],
        "app_rating": [app_rating],
        "recommendation_click_rate": [
            recommendation_click_rate
        ],
        "days_since_last_login": [
            days_since_last_login
        ]
    })


    # --------------------------------------------------------
    # Make prediction
    # --------------------------------------------------------

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(
        input_data
    )[0][1]

    probability_percentage = probability * 100


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns(2)


    # --------------------------------------------------------
    # Churn Probability
    # --------------------------------------------------------

    with result_col1:

        st.metric(
            label="Churn Probability",
            value=f"{probability_percentage:.2f}%"
        )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with result_col2:

        if prediction == 1:

            st.error(
                "⚠️ Customer is predicted to CHURN"
            )

        else:

            st.success(
                "✅ Customer is predicted to STAY"
            )


    # ========================================================
    # RISK LEVEL
    # ========================================================

    st.subheader("Risk Assessment")

    if probability < 0.30:

        risk_level = "Low Risk"

        st.success(
            f"🟢 {risk_level} — "
            f"The model estimates a {probability_percentage:.2f}% "
            "probability of churn."
        )

    elif probability < 0.60:

        risk_level = "Medium Risk"

        st.warning(
            f"🟡 {risk_level} — "
            f"The model estimates a {probability_percentage:.2f}% "
            "probability of churn."
        )

    else:

        risk_level = "High Risk"

        st.error(
            f"🔴 {risk_level} — "
            f"The model estimates a {probability_percentage:.2f}% "
            "probability of churn."
        )


    # ========================================================
    # PROBABILITY BAR
    # ========================================================

    st.subheader("Churn Probability")

    st.progress(
        int(probability_percentage)
    )


    # ========================================================
    # CUSTOMER SUMMARY
    # ========================================================

    st.subheader("Customer Summary")

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

    with summary_col1:
        st.metric(
            "Watch Time",
            f"{avg_watch_time_minutes_per_week} min/week"
        )

    with summary_col2:
        st.metric(
            "Sessions/Week",
            watch_sessions_per_week
        )

    with summary_col3:
        st.metric(
            "Completion Rate",
            f"{completion_rate}%"
        )

    with summary_col4:
        st.metric(
            "Days Since Login",
            days_since_last_login
        )


    # ========================================================
    # RETENTION SUGGESTION
    # ========================================================

    if probability >= 0.60:

        st.subheader("💡 Suggested Action")

        st.info(
            """
            This customer has a relatively high predicted churn risk.
            Consider reviewing their engagement behavior and using
            personalized content recommendations or re-engagement
            communication as potential retention strategies.
            """
        )

    elif probability >= 0.30:

        st.subheader("💡 Suggested Action")

        st.info(
            """
            This customer has moderate predicted churn risk.
            Monitoring engagement and maintaining personalized
            recommendations may be appropriate.
            """
        )

    else:

        st.subheader("💡 Suggested Action")

        st.info(
            """
            This customer currently has a lower predicted churn risk.
            Continue providing relevant content and maintaining
            engagement.
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Netflix Customer Churn Prediction • Machine Learning Project"
)