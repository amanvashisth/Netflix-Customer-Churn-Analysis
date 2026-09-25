# Netflix Customer Churn Prediction

## Deployment

**Live application:** https://netflix-customer-churn-analysi-p7kzvjsykmvwtzmyk7yzjf.streamlit.app/

## Project Overview

Customer churn is a critical business problem for subscription platforms. Knowing which customers are likely to leave helps retention teams prioritize outreach, investigate engagement patterns, and design more targeted interventions.

This project develops an end-to-end customer churn prediction workflow using a dataset of 50,000 synthetic Netflix user records. It includes exploratory data analysis, model comparison and tuning, a persisted preprocessing-and-model pipeline, and a Streamlit interface where a user can enter customer information and receive:

- A predicted outcome: **STAY** or **CHURN**
- An estimated churn probability

The repository also establishes a foundation for customer segmentation and retention analysis. The deployed application currently focuses on churn prediction; segmentation is identified as a follow-up analytical component rather than presented as an implemented application feature.

## Key Capabilities

- Interactive churn prediction through a Streamlit web application
- Support for demographic, subscription, device, content, engagement, and recency attributes
- Numerical feature standardization and categorical one-hot encoding inside the saved pipeline
- Comparison of Logistic Regression, Random Forest, and XGBoost classifiers
- Model evaluation using accuracy, precision, recall, F1 score, ROC-AUC, PR-AUC, classification reports, and confusion matrices
- Randomized hyperparameter search with stratified cross-validation for tree-based models
- A reusable serialized model artifact in `Models/XGBoost_model.pkl`

## Data

The source file is `Data/netflix_user_behavior_churn_50000v2.csv`.

| Dataset characteristic | Description |
| --- | --- |
| Records | 50,000 customer records |
| Columns | 20 total columns |
| Target | `churned` (`0` = not churned, `1` = churned) |
| Churn distribution | 39,536 not churned and 10,464 churned records, approximately 20.93% churn |
| Identifier | `user_id`, excluded from model training |
| Missing values | None identified during EDA |
| Duplicate rows | None identified during EDA |

### Model Features

The model uses the following 18 predictors:

- **Customer and account:** `age`, `gender`, `region`, `subscription_type`, `payment_method`, `account_age_months`
- **Device and content preferences:** `primary_device`, `favorite_genre`, `time_of_day`, `recommendation_source`
- **Usage and engagement:** `session_count`, `avg_watch_time_minutes_per_week`, `watch_sessions_per_week`, `completion_rate`
- **Ratings and recency:** `avg_rating_given`, `app_rating`, `recommendation_click_rate`, `days_since_last_login`

## Modeling Approach

1. Load the customer behavior dataset.
2. Remove `user_id` from the predictors and use `churned` as the binary target.
3. Split the data into training and test sets using an 80/20 stratified split with `random_state=42`.
4. Standardize numerical variables with `StandardScaler`.
5. Encode categorical variables with `OneHotEncoder(handle_unknown="ignore")`.
6. Train and compare Logistic Regression, Random Forest, and XGBoost pipelines.
7. Tune Random Forest and XGBoost with `RandomizedSearchCV` and `StratifiedKFold` cross-validation.
8. Persist the selected XGBoost pipeline with preprocessing included.
9. Load the artifact in Streamlit and generate a class prediction plus churn probability for a new customer profile.

### Saved Model

The deployed app loads `Models/XGBoost_model.pkl`. The artifact is a scikit-learn `Pipeline` containing:

- A `ColumnTransformer`
- `StandardScaler` for 10 numerical features
- `OneHotEncoder(handle_unknown="ignore")` for 8 categorical features
- A tuned `XGBClassifier`

Keeping preprocessing and prediction together helps ensure that inputs submitted through the application are transformed consistently with the training workflow.

## Streamlit Application

The application is implemented in `app.py` and provides controls for:

- Age, gender, and region
- Subscription type and payment method
- Primary device and favorite genre
- Preferred viewing time and recommendation source
- Session count and weekly watch sessions
- Weekly watch time and completion rate
- Average rating given and app rating
- Recommendation click rate and days since last login

After selecting **Predict Customer Churn**, the app displays the estimated churn probability and a clear predicted outcome. The sidebar summarizes the model pipeline and intended use.

## Repository Structure

```text
.
├── app.py                                      # Streamlit prediction application
├── requirements.txt                            # Python dependencies
├── README.md                                   # Project documentation
├── Data/
│   └── netflix_user_behavior_churn_50000v2.csv # Training and analysis data
├── Models/
│   └── XGBoost_model.pkl                       # Saved preprocessing/model pipeline
└── Notebooks/
	├── 01_business_understanding.ipynb         # Business understanding workspace
	├── 02_Data_Exploration_EDA.ipynb            # Data quality and exploratory analysis
	└── Customer_churn_model.ipynb               # Training, evaluation, tuning, and export
```

## Run Locally

### Requirements

- Python 3.9 or newer recommended
- `pip`
- A browser for the Streamlit interface

### Installation

From the project root:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Start the Application

```bash
streamlit run app.py
```

Streamlit will print a local URL, typically `http://localhost:8501`. The model file must remain at `Models/XGBoost_model.pkl` relative to the project root because that is the path used by the application.

## Reproduce the Analysis

Open the notebooks in the following order:

1. `Notebooks/01_business_understanding.ipynb`
2. `Notebooks/02_Data_Exploration_EDA.ipynb`
3. `Notebooks/Customer_churn_model.ipynb`

The modeling notebook documents data preparation, feature selection, preprocessing, baseline models, model comparison, hyperparameter tuning, evaluation visualizations, final model selection, and model serialization.

When running notebooks on another machine, update any machine-specific dataset paths to a project-relative path such as:

```python
pd.read_csv("Data/netflix_user_behavior_churn_50000v2.csv")
```

## Business Use Case

The prediction output can support a retention workflow by helping teams:

- Prioritize high-risk customers for proactive engagement
- Identify behavioral signals associated with inactivity or reduced engagement
- Compare churn risk across subscription, device, and usage profiles
- Inform experiments around recommendations, content discovery, and customer communication

The probability should be treated as a prioritization signal, not as a definitive statement about a customer's future behavior.

## Limitations and Responsible Use

- The dataset is a prepared behavioral dataset and should not be treated as a production Netflix data source.
- A churn prediction is probabilistic and should be validated against current business outcomes before operational use.
- The application does not currently provide explanations such as SHAP values or feature-level reasons for an individual prediction.
- The interface supports a single customer profile at a time; batch scoring and history tracking are not implemented.
- Threshold selection should reflect the business cost of false positives versus missed churners rather than relying only on the default classifier threshold.
- Model performance metrics should be revalidated when the data distribution, product experience, or customer population changes.
- Customer data should be handled according to applicable privacy, security, and data-governance requirements.

## Future Enhancements

- Implement behavioral customer segmentation with a documented clustering workflow
- Add model explainability and feature-level prediction insights
- Add batch CSV scoring for retention teams
- Add monitoring for data drift, calibration, and production performance
- Add automated tests for input schema, model loading, and prediction behavior
- Add a deployment workflow with dependency pinning and environment configuration

## Technology Stack

- **Python** for data science and application development
- **Pandas / NumPy** for data preparation
- **scikit-learn** for preprocessing, pipelines, evaluation, and model selection
- **XGBoost** for the final classifier
- **Matplotlib / Seaborn** for notebook-based analysis and visualizations
- **Streamlit** for the interactive web application

## License

No license file is currently included. Add an appropriate license before distributing this project publicly.
