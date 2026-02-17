**📡 Telco Customer Churn Prediction - MLOps Assessment**
1. Implementation Overview & Design ChoicesFor this assessment, I implemented an end-to-end Telco Customer Churn Prediction pipeline. I selected this project because churn prediction is a high-impact business problem where the cost of errors is asymmetric: false negatives (failing to identify a churning customer) are significantly more costly to a business than false positives (incorrectly flagging a happy customer).

To address this, I benchmarked three distinct algorithms:
Random ForestXGBoost (Selected Champion)
CatBoostWhy XGBoost?
I selected XGBoost as the production champion because it offered the best strategic trade-off between precision and recall.


**2. How to Run the Code**
 This project is deployed as an interactive Streamlit web application that allows for real-time model training, hyperparameter tuning, and benchmarking.PrerequisitesEnsure you have Python installed.
You can check this by running python --version in your terminal.

Step 1: Set up the Environment Install the required dependencies using pip: pip install streamlit pandas numpy matplotlib seaborn scikit-learn xgboost catboost

Step 2: Launch the AppNavigate to the project directory in your terminal and run the Streamlit application: streamlit run app.py


Step 3: Use the InterfaceA browser window will open automatically (usually at http://localhost:8501).
Use the Sidebar to select an algorithm (e.g., "XGBoost (Champion)").Adjust hyperparameters such as scale_pos_weight to see the impact on model performance.Click "Train Model" to view the F1-Score and Confusion Matrix in real-time.3. Assumptions & LimitationsData ConsistencyThe pipeline currently assumes the input data schema (column names and data types) remains consistent with the IBM Telco Dataset.Constraint: Any upstream changes to column names (e.g., tenure, MonthlyCharges) would require updates to the preprocessing pipeline.
Static TrainingThe current deployment trains on the full dataset for demonstration purposes.
Production Reality: In a real-world scenario, the model would be trained on historical data and validated on a strictly separated "future" time slice to prevent data leakage.Feature EngineeringI introduced a feature named TotalServices (a sum of active add-ons) as a proxy for customer "stickiness."
Validation: While this assumption holds true for this specific dataset, it would require validation against domain experts before being applied to a different telecom provider's data.
