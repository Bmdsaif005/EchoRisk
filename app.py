import streamlit as st
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# --- 2. Caching and Model Loading (FinBERT - Optional, but kept for now) ---
# If you want to allow users to paste news text for sentiment analysis, keep this.
# For manual slider input, this is not strictly needed for current implementation.
# @st.cache_resource
# def load_sentiment_model():
#     """Loads the FinBERT model and tokenizer for sentiment analysis."""
#     tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
#     model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
#     return pipeline("text-classification", model=model, tokenizer=tokenizer)

# --- 3. Data Ingestion & Feature Engineering Module (MSME Specific) ---

def get_msme_financial_input():
    """Provides Streamlit forms for users to input MSME financial data."""
    st.sidebar.header("MSME Financial Data Input (Latest Year)")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Balance Sheet Items")
    current_assets = st.sidebar.number_input("Current Assets", min_value=0.0, value=100000.0, help="Total current assets (cash, receivables, inventory)")
    current_liabilities = st.sidebar.number_input("Current Liabilities", min_value=0.0, value=50000.0, help="Total current liabilities (payables, short-term debt)")
    total_liabilities = st.sidebar.number_input("Total Liabilities", min_value=0.0, value=150000.0, help="Total liabilities (current + non-current)")
    total_equity = st.sidebar.number_input("Total Equity", min_value=0.0, value=100000.0, help="Total owner's equity")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Income Statement Items")
    total_revenue = st.sidebar.number_input("Total Revenue", min_value=0.0, value=200000.0, help="Total sales/revenue for the period")
    net_income = st.sidebar.number_input("Net Income (Profit/Loss)", value=20000.0, help="Net profit or loss after all expenses and taxes")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Additional Information")
    business_age_years = st.sidebar.number_input("Business Age (Years)", min_value=1.0, value=1.0, step=0.5, help="How long the business has been operating in years")
    sentiment_score = st.sidebar.slider("Market Sentiment (Manual Score)", min_value=-0.5, max_value=0.5, value=0.0, step=0.05,
                                      help="A subjective score: -0.5 (Very Negative) to 0.5 (Very Positive)")

    # Validate essential inputs
    validation_passed = True
    if current_liabilities <= 0:
        st.sidebar.warning("Current Liabilities must be greater than zero for ratio calculation.")
        validation_passed = False
    if total_equity <= 0:
        st.sidebar.warning("Total Equity must be greater than zero for ratio calculation.")
        validation_passed = False
    if total_revenue <= 0:
        st.sidebar.warning("Total Revenue must be greater than zero for ratio calculation.")
        validation_passed = False
    
    if not validation_passed:
        return None

    return {
        'current_assets': current_assets,
        'current_liabilities': current_liabilities,
        'total_liabilities': total_liabilities,
        'total_equity': total_equity,
        'total_revenue': total_revenue,
        'net_income': net_income,
        'business_age_years': business_age_years,
        'sentiment_score': sentiment_score
    }


def calculate_msme_financial_ratios(data_input):
    ratios = {}
    try:
        # Liquidity: Current Ratio
        if data_input['current_liabilities'] > 0:
            ratios['current_ratio'] = data_input['current_assets'] / data_input['current_liabilities']
        else:
            ratios['current_ratio'] = np.nan # Or a very high number to indicate extreme liquidity

        # Leverage: Debt-to-Equity Ratio
        if data_input['total_equity'] > 0:
            ratios['debt_to_equity'] = data_input['total_liabilities'] / data_input['total_equity']
        else:
            ratios['debt_to_equity'] = 99.0 # Placeholder for negative or zero equity

        # Profitability: Profit Margin
        if data_input['total_revenue'] > 0:
            ratios['profit_margin'] = data_input['net_income'] / data_input['total_revenue']
        else:
            ratios['profit_margin'] = np.nan

        # Include Business Age and Sentiment Score directly as features
        ratios['business_age_years'] = data_input['business_age_years']
        ratios['sentiment_score'] = data_input['sentiment_score']

    except Exception as e:
        st.error(f"Error calculating ratios: {e}")
        return None
    return ratios

# --- 4. Modeling Module ---

def create_synthetic_training_data(size=500):
    np.random.seed(42)
    data = {
        'current_ratio': np.random.uniform(0.1, 4.0, size),
        'debt_to_equity': np.random.uniform(0.5, 10.0, size),
        'profit_margin': np.random.uniform(-0.5, 0.3, size),
        'business_age_years': np.random.uniform(1.0, 10.0, size),
        'sentiment_score': np.random.uniform(-0.5, 0.5, size)
    }
    df = pd.DataFrame(data)

    bankruptcy_propensity_score = (
        -3 * df['current_ratio']
        + 2 * df['debt_to_equity']
        - 4 * df['profit_margin']
        - 1 * df['business_age_years']
        - 2 * df['sentiment_score']
    )

    probability_not_bankrupt = 1 / (1 + np.exp(bankruptcy_propensity_score))
    df['is_bankrupt'] = (probability_not_bankrupt < 0.3).astype(int)

    return df.drop('is_bankrupt', axis=1), df['is_bankrupt']


def train_model(X_train, y_train, model_type='logistic'):
    if model_type == 'logistic':
        model = LogisticRegression(random_state=42, solver='liblinear')
    else:
        model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model


def predict_bankruptcy(model, features):
    prediction = model.predict(features)[0]
    probability_of_bankruptcy = model.predict_proba(features)[0][1]
    return prediction, probability_of_bankruptcy


def evaluate_model(model_instance_type, X, y): # Changed 'model' to 'model_instance_type' for clarity
    """
    Evaluates the model using Stratified K-Fold Cross-Validation and calculates metrics.
    `model_instance_type` should be the class of the model (e.g., LogisticRegression).
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    accuracies, precisions, recalls, f1_scores, roc_aucs = [], [], [], [], []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Create a new instance of the model for each fold
        fold_model = model_instance_type() # Instantiate the model type
        if isinstance(fold_model, LogisticRegression):
            fold_model.set_params(random_state=42, solver='liblinear')
        elif isinstance(fold_model, xgb.XGBClassifier):
            fold_model.set_params(random_state=42, use_label_encoder=False, eval_metric='logloss')

        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_test)
        y_proba = fold_model.predict_proba(X_test)[:, 1]

        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
        
        if len(np.unique(y_test)) > 1:
            roc_aucs.append(roc_auc_score(y_test, y_proba))
        else:
            roc_aucs.append(np.nan)

    metrics = {
        'accuracy': np.mean(accuracies),
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'f1_score': np.mean(f1_scores),
        'roc_auc': np.nanmean(roc_aucs)
    }
    return metrics

# --- 5. Recommendation Module ---

def generate_bankruptcy_recommendations(features, prediction_result):
    recs = []
    if prediction_result == 1:
        recs.append("🚨 **High Bankruptcy Risk Detected!** Immediate action required.")
    else:
        recs.append("✅ **Low Bankruptcy Risk.** Continue to monitor financial health closely.")

    if features['current_ratio'] < 1.0:
        recs.append("🔴 **Liquidity Alert:** Current ratio is low. Focus on improving short-term cash flow, managing receivables, and optimizing inventory.")
    elif features['current_ratio'] > 3.0:
        recs.append("🟡 **Liquidity Note:** Current ratio is quite high. While good for solvency, ensure assets are efficiently utilized and not idle.")

    if features['debt_to_equity'] > 2.0:
        recs.append("🔴 **High Leverage:** Debt-to-Equity is elevated. Consider strategies to reduce debt, such as equity injection or improving profitability to pay down liabilities.")
    elif features['debt_to_equity'] > 5.0:
        recs.append("🔥 **Extreme Leverage:** Debt-to-Equity is very high. This indicates significant financial risk. Seek professional financial advice immediately.")

    if features['profit_margin'] < 0.05 and features['profit_margin'] >= 0:
        recs.append("🟡 **Tight Margins:** Profit margin is very low. Review pricing strategy, control operating costs, and identify areas for efficiency improvement.")
    elif features['profit_margin'] < 0.0:
        recs.append("🔴 **Unprofitable:** Business is currently operating at a loss. Urgent review of business model, cost structure, and revenue generation is critical.")

    if features['business_age_years'] < 3.0 and prediction_result == 1:
        recs.append("💡 **Early Stage Vulnerability:** As a young business, cash flow management and securing stable revenue streams are paramount to overcome initial hurdles.")

    if features['sentiment_score'] < -0.1:
        recs.append("🔴 **Negative Market Perception:** Address any negative publicity or market sentiment. Proactive communication and customer satisfaction efforts can help.")

    if len(recs) == 1:
        recs.append("Overall financial indicators appear healthy. Maintain good financial practices.")
    return recs

# --- New Feature: What-If Analysis Visualization ---
def plot_what_if_analysis(model, base_features, feature_to_vary, label_x, min_val, max_val, current_value):
    """
    Plots how bankruptcy probability changes when a selected feature is varied.
    """
    vary_range = np.linspace(min_val, max_val, 50)
    probabilities = []

    for val in vary_range:
        temp_features = base_features.copy()
        temp_features[feature_to_vary] = val
        proba = model.predict_proba(pd.DataFrame([temp_features]))[0][1]
        probabilities.append(proba)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=vary_range, y=probabilities, ax=ax, color='skyblue', linewidth=2)
    ax.axvline(x=current_value, color='red', linestyle='--', label=f'Current {label_x}: {current_value:.2f}')
    
    ax.set_title(f'Impact of {label_x} on Bankruptcy Probability')
    ax.set_xlabel(label_x)
    ax.set_ylabel('Probability of Bankruptcy')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    st.pyplot(fig)

# --- New: Global Feasibility Input ---
def get_global_feasibility_input():
    st.subheader("🌍 Global Feasibility Study")
    domain = st.selectbox("Business Domain *", ["Retail", "Technology", "Healthcare", "Manufacturing", "Other"])
    product_service = st.text_input("Product/Service *", placeholder="Enter your product/service")
    city = st.text_input("Target City *", placeholder="Enter target city")

    if st.button("Run Feasibility Study"):
        st.success(f"Running Feasibility Study for {product_service} in {city} ({domain})...")

        # Mock competition insights (can be replaced with real dataset APIs)
        competitors = np.random.randint(3, 15)
        avg_price = np.random.randint(100, 500)
        demand_index = np.random.uniform(0.2, 0.9)

        st.metric("Estimated Competitors", competitors)
        st.metric("Avg Price Point", f"${avg_price}")
        st.metric("Demand Index", f"{demand_index:.2f}")

        # Visuals
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=demand_index * 100,
            title={'text': "Market Demand (%)"},
            gauge={'axis': {'range': [0, 100]}}
        ))
        st.plotly_chart(fig, use_container_width=True)

        # Recommendations
        st.subheader("Strategic Recommendations")
        st.markdown("- 📊 Focus on differentiating through **unique value propositions**.")
        st.markdown("- 🤝 Build **local partnerships** to penetrate city markets faster.")
        st.markdown("- 📈 Consider **pricing between** current average and +10% premium.")

# --- 6. Streamlit Dashboard ---

def main():
    st.set_page_config(page_title="MSME Bankruptcy Analysis DSS", layout="wide")
    st.title("📉 MSME Bankruptcy Risk Analysis Decision Support System")
    st.markdown("---")

    st.sidebar.header("Configuration")
    msme_name = st.sidebar.text_input("MSME Name (e.g., 'Bharat Enterprises')", "MSME XYZ")
    model_choice = st.sidebar.selectbox("Choose Prediction Model", ['Logistic Regression', 'XGBoost'])

    msme_inputs = get_msme_financial_input() # This call is now after the function definition

    if msme_inputs is None:
        st.warning("Please correct the input errors in the sidebar to proceed with analysis.")
        # Ensure session state for model/features is cleared if inputs are invalid
        if 'model' in st.session_state:
            del st.session_state['model']
        if 'features' in st.session_state:
            del st.session_state['features']
        return

    st.sidebar.markdown("---")
    
    # Initialize session state variables if they don't exist
    if 'model' not in st.session_state:
        st.session_state['model'] = None
    if 'features' not in st.session_state:
        st.session_state['features'] = None
    if 'prediction' not in st.session_state:
        st.session_state['prediction'] = None
    if 'probability_of_bankruptcy' not in st.session_state:
        st.session_state['probability_of_bankruptcy'] = None
    if 'recommendations' not in st.session_state:
        st.session_state['recommendations'] = []
    if 'model_metrics' not in st.session_state:
        st.session_state['model_metrics'] = {}

    if st.sidebar.button("Analyze MSME"):
        with st.spinner(f"Analyzing {msme_name}..."):
            financial_ratios = calculate_msme_financial_ratios(msme_inputs)
            if financial_ratios is None:
                st.error("Could not calculate financial ratios. Please check inputs.")
                # Clear relevant session state on error
                st.session_state['model'] = None
                st.session_state['features'] = None
                return

            features = {
                'current_ratio': financial_ratios.get('current_ratio', 0),
                'debt_to_equity': financial_ratios.get('debt_to_equity', 0),
                'profit_margin': financial_ratios.get('profit_margin', 0),
                'business_age_years': financial_ratios.get('business_age_years', 1),
                'sentiment_score': financial_ratios.get('sentiment_score', 0)
            }

            features = {k: 0 if pd.isna(v) else v for k, v in features.items()}
            features_df = pd.DataFrame([features])

            X_synth, y_synth = create_synthetic_training_data()
            
            # Get the model class based on user choice
            if model_choice == 'Logistic Regression':
                model_class = LogisticRegression
            else:
                model_class = xgb.XGBClassifier

            # Train the final model on all synthetic data for prediction
            model = train_model(X_synth, y_synth, 'logistic' if model_choice == 'Logistic Regression' else 'xgboost')
            
            # Evaluate the model using cross-validation on the synthetic data
            # Pass the model *class* to evaluate_model
            model_metrics = evaluate_model(model_class, X_synth, y_synth) 

            prediction, probability_of_bankruptcy = predict_bankruptcy(model, features_df)
            recommendations = generate_bankruptcy_recommendations(features, prediction)

            # Store results in session state
            st.session_state['model'] = model
            st.session_state['features'] = features
            st.session_state['prediction'] = prediction
            st.session_state['probability_of_bankruptcy'] = probability_of_bankruptcy
            st.session_state['recommendations'] = recommendations
            st.session_state['model_metrics'] = model_metrics
            
    # Display results only if analysis has been run (i.e., model and features are in session_state)
    if st.session_state['model'] is not None and st.session_state['features'] is not None:
        st.header(f"Bankruptcy Analysis for {msme_name}")
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state['prediction'] == 1:
                st.error("Prediction: High Bankruptcy Risk")
            else:
                st.success("Prediction: Low Bankruptcy Risk")
        with col2:
            st.metric("Probability of Bankruptcy", f"{st.session_state['probability_of_bankruptcy']:.2%}")

        st.subheader("Key Financial & Business Indicators")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Current Ratio", f"{st.session_state['features']['current_ratio']:.2f}")
        c2.metric("Debt-to-Equity", f"{st.session_state['features']['debt_to_equity']:.2f}")
        c3.metric("Profit Margin", f"{st.session_state['features']['profit_margin']:.2%}")
        c4.metric("Business Age (Yrs)", f"{st.session_state['features']['business_age_years']:.1f}")
        c5.metric("Market Sentiment", f"{st.session_state['features']['sentiment_score']:.2f}")

        st.subheader("Actionable Recommendations")
        for rec in st.session_state['recommendations']:
            st.markdown(f"- {rec}")

        st.markdown("---")
        st.subheader("Interactive 'What-If' Analysis")
        st.write("See how changes in a key financial indicator could affect the probability of bankruptcy.")
        
        # What-If Analysis controls - moved outside the button block
        feature_for_what_if = st.selectbox(
            "Select a feature to vary:",
            ['current_ratio', 'debt_to_equity', 'profit_margin', 'business_age_years', 'sentiment_score'],
            format_func=lambda x: x.replace('_', ' ').title(),
            key='what_if_feature_selector' # Add a key to prevent potential issues
        )

        # Define reasonable ranges for the selected features for the plot
        min_val, max_val = 0.0, 1.0 # Default values
        current_val = st.session_state['features'][feature_for_what_if]

        if feature_for_what_if == 'current_ratio':
            min_val, max_val = 0.1, 5.0
        elif feature_for_what_if == 'debt_to_equity':
            min_val, max_val = 0.1, 15.0
        elif feature_for_what_if == 'profit_margin':
            min_val, max_val = -0.5, 0.5
        elif feature_for_what_if == 'business_age_years':
            min_val, max_val = 0.5, 20.0
        elif feature_for_what_if == 'sentiment_score':
            min_val, max_val = -0.5, 0.5
            
        plot_what_if_analysis(
            st.session_state['model'],
            st.session_state['features'], # Pass the dictionary of current features
            feature_for_what_if,
            feature_for_what_if.replace('_', ' ').title(), # Label for X-axis
            min_val,
            max_val,
            current_val
        )

        st.markdown("---")
        st.subheader(f"Model Performance ({model_choice} on Synthetic Data via 5-Fold Cross-Validation)")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Accuracy", f"{st.session_state['model_metrics']['accuracy']:.2f}")
        col_m2.metric("Precision", f"{st.session_state['model_metrics']['precision']:.2f}")
        col_m3.metric("Recall", f"{st.session_state['model_metrics']['recall']:.2f}")

        col_m4, col_m5, _ = st.columns(3)
        col_m4.metric("F1-Score", f"{st.session_state['model_metrics']['f1_score']:.2f}")
        col_m5.metric("ROC AUC", f"{st.session_state['model_metrics']['roc_auc']:.2f}")
        st.info("These metrics reflect the model's performance on the *synthetic training data* and indicate its general learning capability. Actual performance on real-world MSME data may vary.")

if __name__ == "__main__":
    main()
