
# **video showcase the different model's accuracy and F-1 score**

https://github.com/user-attachments/assets/0d0a21b8-41b7-4c75-8588-e95d3f0d4dcf


**figure show the streamlit dashboard for the Telecom customer churn XGBoost F-1 score and Accuracy** 
<img width="1918" height="1018" alt="image" src="https://github.com/user-attachments/assets/ab0918d1-6210-4025-8ea2-cda9ff00f853" />




# 📡 Telco Customer Churn Prediction - MLOps Assessment

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Status](https://img.shields.io/badge/Status-Complete-success)

## 1. Implementation Overview & Design Choices

For this assessment, I implemented an end-to-end **Telco Customer Churn Prediction pipeline**. I selected this project because churn prediction is a high-impact business problem where the cost of errors is asymmetric: **false negatives** (failing to identify a churning customer) are significantly more costly to a business than **false positives** (incorrectly flagging a happy customer).

To address this, I benchmarked three distinct algorithms:
* **Random Forest**
* **XGBoost** (Selected Champion)
* **CatBoost**

### Why XGBoost?
I selected **XGBoost** as the production champion because it offered the best strategic trade-off between precision and recall. In a churn context, maximizing recall (catching at-risk customers) is critical to the business model, and XGBoost provided the most robust performance after hyperparameter tuning.

---

## 2. How to Run the Code

This project is deployed as an interactive **Streamlit** web application that allows for real-time model training, hyperparameter tuning, and benchmarking.

### Prerequisites
Ensure you have Python installed. You can check this by running `python --version` in your terminal.

### Step 1: Set up the Environment
Install the required dependencies using `pip`:

```bash```
pip install streamlit pandas numpy matplotlib seaborn scikit-learn xgboost catboost

Step 2: Launch the App
Navigate to the project directory in your terminal and run the Streamlit application:

```Bash```
streamlit run app.py

Step 3: Use the Interface
A browser window will open automatically (usually at http://localhost:8501).

Use the Sidebar to select an algorithm (e.g., "XGBoost (Champion)").

Adjust hyperparameters such as scale_pos_weight to see the impact on model performance.

Click "Train Model" to view the F1-Score and Confusion Matrix in real-time.

### 4. Reflection
Did the coding assistant help you move faster?
Absolutely. I utilized Gemini throughout the process, and it worked well for me. It significantly accelerated the development of the boilerplate code and the Streamlit interface.

Did it generate incorrect or surprising suggestions?
No. I provided information and requirements in modular "chunks," which allowed the LLM to understand the specific context clearly. As a result, it answered my questions accurately without hallucinations.

Where was it most or least useful?
Most Useful: It was invaluable for the coding implementation, specifically for writing the Streamlit app logic and the Scikit-Learn/XGBoost pipeline syntax.

Least Useful: I performed the domain analysis and dataset understanding myself. The assistant was less involved in the conceptual understanding of the problem and the data, as I preferred to rely on my own analysis to ensure business alignment.
pip install streamlit pandas numpy matplotlib seaborn scikit-learn xgboost catboost
