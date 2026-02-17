Telco Customer Churn Prediction - MLOps Assessment
1. Implementation Overview & Design Choices
For this assessment, I chose to implement a Telco Customer Churn Prediction pipeline. I selected this project because churn prediction is a high-impact business problem where false negatives (missing a churning customer) are significantly more costly than false positives. I compared three algorithms: Random Forest, XGBoost, and CatBoost.

Why XGBoost?
I selected XGBoost as the production champion because it offered the best strategic trade-off. While Random Forest achieved higher raw accuracy (~80%), it failed to identify nearly 50% of churners (low Recall). By tuning XGBoost with a scale_pos_weight of 2.0, I sacrificed a small amount of overall accuracy (dropping to ~78%) to boost the Recall to 81% and F1-Score to 0.64. This trade-off is intentional: in a churn context, catching 81% of at-risk customers is far more valuable than maximizing accuracy on happy customers who were never going to leave.

2. How to Run the Code
This project is deployed as an interactive Streamlit web application that allows for real-time model training and benchmarking.

Steps to Run Locally:

Set up the Environment:
Ensure you have Python installed, then install the required dependencies:

Bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn xgboost catboost
Launch the App:
Navigate to the project directory in your terminal and run:

Bash
streamlit run app.py
Use the Interface:
A browser window will open automatically. You can use the sidebar to select an algorithm (e.g., "XGBoost (Champion)"), adjust hyperparameters like scale_pos_weight, and click "Train Model" to see the F1-Score and Confusion Matrix in real-time.

3. Assumptions & Limitations
Data Consistency: The pipeline assumes the input data schema (column names and types) remains consistent with the IBM Telco Dataset. Any changes to column names like tenure or MonthlyCharges would require updates to the preprocessing pipeline.

Static Training: The current deployment trains on the full dataset for demonstration purposes. In a real-world production scenario, the model would be trained on historical data and validated on a separate "future" slice to prevent data leakage.

Feature Engineering: I assumed that TotalServices (a sum of active add-ons) is a proxy for customer "stickiness." This holds true for this dataset but would need validation against domain experts for a different telecom provider.
