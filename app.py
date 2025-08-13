import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
from scipy.optimize import minimize


# Set page title and layout
st.set_page_config(page_title="Revenue Optimization of in app purchases", layout="wide")

# Application title
st.title("Game Purchase Price Prediction and Optimization")
st.write(
    "This app predicts and optimizes the price for in-game purchases based on user and behavioral features."
)

# Define paths
models_path = os.path.join(os.getcwd(), "models")
default_models_path = (
    "/Users/sanatwalia/Desktop/Assignments_applications/Revenue_Optimization-/models"
)


# Load saved models
@st.cache_resource
def load_models():
    try:
        with open(os.path.join(default_models_path, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        with open(os.path.join(default_models_path, "label_encoder.pkl"), "rb") as f:
            label_encoder = pickle.load(f)
        with open(
            os.path.join(default_models_path, "random_forest_model.pkl"), "rb"
        ) as f:
            model = pickle.load(f)
        st.sidebar.success("Models loaded from default path")
        return scaler, label_encoder, model
    except FileNotFoundError:
        try:
            with open(os.path.join(models_path, "scaler.pkl"), "rb") as f:
                scaler = pickle.load(f)
            with open(os.path.join(models_path, "label_encoder.pkl"), "rb") as f:
                label_encoder = pickle.load(f)
            with open(os.path.join(models_path, "random_forest_model.pkl"), "rb") as f:
                model = pickle.load(f)
            st.sidebar.success("Models loaded from local path")
            return scaler, label_encoder, model
        except FileNotFoundError as e:
            st.error(f"Error loading models: {e}")
            return None, None, None


# Load models
scaler, label_encoder, model = load_models()

# Define columns
categorical_columns = [
    "User Segment",
    "Purchase Type",
    "Promo Applied",
    "Item-Purchased",
]
numerical_columns = [
    "Session Time",
    "Level Reached",
    "Prior Purchases",
    "Days Installed Before Last Purchase",
]

# Debug section
with st.sidebar.expander("Debug Info", expanded=False):
    if scaler is not None and hasattr(scaler, "n_features_in_"):
        st.write(f"Scaler expects {scaler.n_features_in_} features")
    if model is not None and hasattr(model, "n_features_in_"):
        st.write(f"Model expects {model.n_features_in_} features")
    debug_mode = st.checkbox("Enable Debug Mode", value=False)

# Input sidebar
st.sidebar.header("Input Features")

user_segment = st.sidebar.selectbox("User Segment", ["Casual", "Hardcore", "Whales"])
item_purchased = st.sidebar.selectbox(
    "Item Purchased",
    [
        "Season Pass",
        "Premium Weapon",
        "Limited-Time Offer Pack",
        "Ultra Bundle",
        "Power-Up Bundle",
        "Character Skin",
        "Lifetime Membership",
        "Coin Pack - 1000",
        "Coin Pack - 5000",
        "Coin Pack - 10000",
    ],
)
purchase_type = st.sidebar.selectbox("Purchase Type", ["Subscription", "One-time"])
promo_applied = st.sidebar.selectbox("Promo Applied", ["No", "Yes"])
session_time = st.sidebar.slider("Session Time (minutes)", 1, 180, 30)
level_reached = st.sidebar.slider("Level Reached", 1, 100, 20)
prior_purchases = st.sidebar.slider("Prior Purchases", 0, 20, 2)
install_date = st.sidebar.date_input(
    "Date of Install", datetime.now() - timedelta(days=30)
)
purchase_date = st.sidebar.date_input("Date of Purchase", datetime.now())
days_installed = (purchase_date - install_date).days

if days_installed < 0:
    st.sidebar.error("Purchase date must be after install date!")


# Prepare input data
def prepare_input_data():
    return pd.DataFrame(
        {
            "User Segment": [user_segment],
            "Item-Purchased": [item_purchased],
            "Session Time": [session_time],
            "Level Reached": [level_reached],
            "Prior Purchases": [prior_purchases],
            "Purchase Type": [purchase_type],
            "Promo Applied": [promo_applied],
            "Days Installed Before Last Purchase": [days_installed],
            "Date of Purchase": [purchase_date.strftime("%Y-%m-%d")],
            "Date of Install": [install_date.strftime("%Y-%m-%d")],
        }
    )


# Data processing
def process_data(df):
    if scaler is None or label_encoder is None:
        st.error("Models not loaded correctly.")
        return None

    processed_df = df.copy()

    # Encode categorical features (label encoding)
    for col in categorical_columns:
        mapped_values = []
        for val in df[col]:
            try:
                mapped_values.append(label_encoder.transform([val])[0])
            except:
                mapped_values.append(0)
        processed_df[col] = mapped_values

    try:
        # Define the correct feature order (same as X_train)
        all_features = [
            "User Segment",
            "Item-Purchased",
            "Session Time",
            "Level Reached",
            "Prior Purchases",
            "Purchase Type",
            "Promo Applied",
            "Days Installed Before Last Purchase",
        ]

        # Ensure column order matches training
        data_for_scaling = processed_df[all_features].values

        # Apply the scaler (trained on all 8 features)
        scaled_data = scaler.transform(data_for_scaling)

        # Replace with scaled values
        for i, col in enumerate(all_features):
            processed_df[col] = scaled_data[:, i]

        # Final DataFrame with correct order
        return processed_df[all_features]

    except Exception as e:
        st.error(f"Error scaling data: {str(e)}")
        return None


# Prediction
def predict_price(processed_df):
    try:
        return model.predict(processed_df)[0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None


# Optimization
def optimize_price(predicted_price):
    rndf_pred = np.array([predicted_price])

    def objective(prices):
        return -np.sum(prices) + 0.05 * np.sum(np.abs(prices - rndf_pred) ** 1.2)

    def constraints(prices):
        return np.concatenate(
            [
                prices - 0.8 * rndf_pred,
                1.1 * rndf_pred - prices,
                [np.sum(prices) - 0.96 * np.sum(rndf_pred)],
            ]
        )

    result = minimize(
        objective,
        rndf_pred,
        constraints={"type": "ineq", "fun": constraints},
        bounds=[(0.8 * p, 1.1 * p) for p in rndf_pred],
    )
    return result.x[0], np.sum(rndf_pred), np.sum(result.x)


# Prediction logic
if st.sidebar.button("Predict and Optimize Price"):
    with st.spinner("Processing..."):
        input_data = prepare_input_data()
        st.subheader("Input Data")
        st.dataframe(input_data)

        processed_data = process_data(input_data)

        if processed_data is not None:
            st.subheader("Processed Data")
            st.dataframe(processed_data)

            predicted_price = predict_price(processed_data)

            if predicted_price is not None:
                optimized_price, rev_before, rev_after = optimize_price(predicted_price)
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Predicted Price", f"₹{predicted_price:.2f}")
                    st.write(f"Original Revenue: ₹{rev_before:.2f}")
                with col2:
                    st.metric(
                        "Optimized Price",
                        f"₹{optimized_price:.2f}",
                        delta=f"{((optimized_price - predicted_price) / predicted_price) * 100:.2f}%",
                    )
                    st.write(f"Optimized Revenue: ${rev_after:.2f}")

                st.bar_chart(
                    pd.DataFrame(
                        {
                            "Price Type": ["Predicted", "Optimized"],
                            "Price": [predicted_price, optimized_price],
                        }
                    ).set_index("Price Type")
                )

                if optimized_price > predicted_price:
                    st.success(
                        f"Recommendation: Increase price to ₹{optimized_price:.2f}"
                    )
                else:
                    st.success(
                        f"Recommendation: Decrease price to ₹{optimized_price:.2f}"
                    )

# Batch processing
st.sidebar.markdown("---")
st.sidebar.subheader("Batch Processing")
uploaded_file = st.sidebar.file_uploader("Upload CSV with user data", type="csv")

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)

        if "Days Installed Before Last Purchase" not in batch_df.columns:
            batch_df["Date of Purchase"] = pd.to_datetime(batch_df["Date of Purchase"])
            batch_df["Date of Install"] = pd.to_datetime(batch_df["Date of Install"])
            batch_df["Days Installed Before Last Purchase"] = (
                batch_df["Date of Purchase"] - batch_df["Date of Install"]
            ).dt.days

        results = []

        for _, row in batch_df.iterrows():
            input_df = pd.DataFrame([row])
            processed = process_data(input_df)
            if processed is not None:
                pred_price = predict_price(processed)
                if pred_price is not None:
                    opt_price, rev_before, rev_after = optimize_price(pred_price)
                    input_df["Predicted Price"] = pred_price
                    input_df["Optimized Price"] = opt_price
                    input_df["Original Revenue"] = rev_before
                    input_df["Optimized Revenue"] = rev_after
                    input_df["Price Change (%)"] = (
                        (opt_price - pred_price) / pred_price
                    ) * 100
                    results.append(input_df)

        if results:
            result_df = pd.concat(results, ignore_index=True)
            st.subheader("Batch Results")
            st.dataframe(result_df)

            # Download button
            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Results as CSV", csv, "optimized_prices.csv", "text/csv"
            )

    except Exception as e:
        st.error(f"Error processing batch file: {str(e)}")

# About section
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info(
    """
    This application uses machine learning to predict and optimize the price of in-game purchases.
    Upload user behavior data to perform bulk optimization for revenue maximization.
    """
)
