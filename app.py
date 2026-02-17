import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Telco Churn Prediction Lab",
    page_icon="📡",
    layout="wide"
)

# --- 1. DATA LOADING & PREPROCESSING (Cached) ---
@st.cache_data
def load_and_preprocess_data():
    """
    Loads data from the source, performs cleaning, encoding, and feature engineering.
    Cached to prevent reloading on every interaction.
    """
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)

    # 1. Cleaning 'TotalCharges'
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # 2. Drop ID
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # 3. Feature Engineering (The "Senior" Touch)
    # Total Services
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    # Count how many services are 'Yes'
    df['TotalServices'] = df[service_cols].apply(lambda x: x.str.contains('Yes').sum(), axis=1)

    # Price Hike Indicator
    df['Avg_Charge_Real'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['Price_Hike_Indicator'] = df['MonthlyCharges'] - df['Avg_Charge_Real']

    # 4. Encoding
    # Binary Encoding (Yes/No -> 1/0)
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    le = LabelEncoder()
    for col in binary_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
    
    # Gender (Male/Female -> 1/0)
    if 'gender' in df.columns:
        df['gender'] = le.fit_transform(df['gender'])

    # One-Hot Encoding for the rest
    df = pd.get_dummies(df, drop_first=True)

    return df

# --- 2. SIDEBAR: MODEL SELECTION ---
st.sidebar.header("🛠️ Model Configuration")

# Dropdown for Algorithm
model_choice = st.sidebar.selectbox(
    "Choose an Algorithm:",
    ("Random Forest", "XGBoost (Champion)", "CatBoost (Challenger)")
)

# Dynamic Hyperparameters based on selection
st.sidebar.subheader("Hyperparameters")
split_size = st.sidebar.slider("Test Split Size", 0.1, 0.4, 0.2)

if model_choice == "Random Forest":
    n_estimators = st.sidebar.slider("Number of Trees", 50, 500, 100)
    max_depth = st.sidebar.slider("Max Depth", 2, 20, 10)
    
elif model_choice == "XGBoost (Champion)":
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1)
    scale_pos_weight = st.sidebar.slider("Class Balance Weight", 1.0, 5.0, 2.0)
    
elif model_choice == "CatBoost (Challenger)":
    iterations = st.sidebar.slider("Iterations", 100, 1000, 500)
    depth = st.sidebar.slider("Tree Depth", 4, 10, 6)

# --- 3. MAIN APP LOGIC ---
st.title("📡 Telco Customer Churn Prediction Lab")
st.markdown("""
This application allows you to benchmark different Machine Learning algorithms on the Telco Churn dataset.
**Goal:** Maximize F1-Score (catching churners) while maintaining high Accuracy.
""")

# Load Data
with st.spinner("Loading and Engineering Features..."):
    df = load_and_preprocess_data()

st.write(f"**Data Status:** Loaded {df.shape[0]} rows with {df.shape[1]} features (after engineering).")

# Button to Train
if st.button("🚀 Train Model"):
    
    # Prepare Data
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=42)

    # Initialize Model
    if model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        
    elif model_choice == "XGBoost (Champion)":
        model = XGBClassifier(learning_rate=learning_rate, scale_pos_weight=scale_pos_weight, 
                              eval_metric='logloss', random_state=42, use_label_encoder=False)
        
    elif model_choice == "CatBoost (Challenger)":
        model = CatBoostClassifier(iterations=iterations, depth=depth, 
                                   auto_class_weights='Balanced', verbose=0, random_state=42)

    # Train
    with st.spinner(f"Training {model_choice}..."):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

    # --- 4. DISPLAY RESULTS ---
    st.divider()
    st.subheader(f"📊 Results for {model_choice}")

    # Metrics Columns
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{acc:.2%}", delta_color="normal")
    col2.metric("F1-Score (Key Metric)", f"{f1:.4f}", delta_color="inverse")
    
    # Logic Gate Status
    if f1 > 0.60:
        col3.success("✅ Ready for Production")
    else:
        col3.error("❌ Failed Logic Gate")

    # Layout for Plots
    col_chart1, col_chart2 = st.columns(2)

    # Plot 1: Confusion Matrix
    with col_chart1:
        st.write("### Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        st.pyplot(fig)

    # Plot 2: Feature Importance (if supported)
    with col_chart2:
        st.write("### Top 10 Drivers of Churn")
        try:
            if model_choice == "Random Forest":
                importances = model.feature_importances_
            elif model_choice == "XGBoost (Champion)":
                importances = model.feature_importances_
            elif model_choice == "CatBoost (Challenger)":
                importances = model.feature_importances_
            
            # Create a dataframe for plotting
            feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(10)
            
            fig2, ax2 = plt.subplots()
            sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis", ax=ax2)
            plt.xlabel("Importance Score")
            st.pyplot(fig2)
        except:
            st.warning("Feature importance not available for this configuration.")

    # Detailed Report (Expandable)
    with st.expander("See Detailed Classification Report"):
        st.text(classification_report(y_test, y_pred))

else:
    st.info("👈 Select a model from the sidebar and click 'Train Model' to start.")