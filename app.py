# Streamlit app for Student Performance Index Predictor
# This app loads trained models and preprocessor from joblib files

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the preprocessor
@st.cache_resource
def load_preprocessor():
    try:
        return joblib.load('preprocessor.joblib')
    except Exception as e:
        st.error(f"Error loading preprocessor: {str(e)}")
        return None

# Load all available models
@st.cache_resource
def load_models():
    models = {}
    model_files = [f for f in os.listdir('.') if f.endswith('.joblib') and f != 'preprocessor.joblib']
    
    for model_file in model_files:
        model_name = model_file.replace('.joblib', '')
        try:
            models[model_name] = joblib.load(model_file)
        except Exception as e:
            st.error(f"Error loading {model_name}: {str(e)}")
    
    return models

# Get model performance data (from your training results)
def get_model_performance():
    return {
        'LinearRegression': {'MAE': 1.6470, 'MSE': 4.3059, 'RMSE': 2.0751, 'R2': 0.9884},
        'Ridge': {'MAE': 1.6470, 'MSE': 4.3060, 'RMSE': 2.0751, 'R2': 0.9884},
        'Lasso': {'MAE': 2.2132, 'MSE': 7.7267, 'RMSE': 2.7797, 'R2': 0.9792},
        'ElasticNet': {'MAE': 6.0225, 'MSE': 52.4413, 'RMSE': 7.2416, 'R2': 0.8591},
        'BayesianRidge': {'MAE': 1.6470, 'MSE': 4.3059, 'RMSE': 2.0751, 'R2': 0.9884},
        'SGDRegressor': {'MAE': 1.6483, 'MSE': 4.3164, 'RMSE': 2.0776, 'R2': 0.9884},
        'HuberRegressor': {'MAE': 1.6473, 'MSE': 4.3069, 'RMSE': 2.0753, 'R2': 0.9884},
        'TheilSenRegressor': {'MAE': 1.6480, 'MSE': 4.3141, 'RMSE': 2.0771, 'R2': 0.9884},
        'RANSACRegressor': {'MAE': 1.6470, 'MSE': 4.3059, 'RMSE': 2.0751, 'R2': 0.9884},
        'DecisionTreeRegressor': {'MAE': 2.4332, 'MSE': 9.3061, 'RMSE': 3.0506, 'R2': 0.9750},
        'RandomForestRegressor': {'MAE': 1.8950, 'MSE': 5.6151, 'RMSE': 2.3696, 'R2': 0.9849},
        'GradientBoostingRegressor': {'MAE': 1.6958, 'MSE': 4.5759, 'RMSE': 2.1391, 'R2': 0.9877},
        'AdaBoostRegressor': {'MAE': 2.2543, 'MSE': 8.0578, 'RMSE': 2.8386, 'R2': 0.9783},
        'KNeighborsRegressor': {'MAE': 2.3614, 'MSE': 8.7567, 'RMSE': 2.9592, 'R2': 0.9765},
        'SVR_rbf': {'MAE': 1.8179, 'MSE': 5.4183, 'RMSE': 2.3277, 'R2': 0.9854},
        'SVR_linear': {'MAE': 1.8179, 'MSE': 5.4183, 'RMSE': 2.3277, 'R2': 0.9854}
    }

# Data for column names and types (should match your training data)
num_features = ['hours_studied', 'previous_score', 'sleep_hours', 'simple_ques_papers_prac']
cat_features = ['extra_curr_activities']

# Streamlit UI
st.set_page_config(
    page_title="Student Performance Index Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title('🎓 Student Performance Index Predictor')
st.markdown(
    """
    <div style='text-align: center; margin-bottom: 30px;'>
        <p style='font-size: 18px; color: #666;'>
            Predict student academic performance using advanced machine learning models
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

# Load models and preprocessor
try:
    preprocessor = load_preprocessor()
    models = load_models()
    model_performance = get_model_performance()
    
    if not models:
        st.error("No models found! Please ensure model .joblib files are in the current directory.")
        st.stop()
        
    if preprocessor is None:
        st.error("Preprocessor not found! Please ensure preprocessor.joblib is in the current directory.")
        st.stop()
        
except Exception as e:
    st.error(f"Error loading models or preprocessor: {str(e)}")
    st.stop()

# Sidebar for model selection and performance
st.sidebar.header("📊 Model Selection & Performance")
st.sidebar.markdown("---")

# Model selection
selected_model = st.sidebar.selectbox(
    "Choose a Model:",
    options=list(models.keys()),
    help="Select which machine learning model to use for prediction"
)

# Display model performance
if selected_model in model_performance:
    st.sidebar.subheader(f"📈 {selected_model} Performance")
    perf = model_performance[selected_model]
    
    # Create metrics in a more organized way
    col1_sidebar, col2_sidebar = st.sidebar.columns(2)
    with col1_sidebar:
        st.metric("R² Score", f"{perf['R2']:.4f}")
        st.metric("MAE", f"{perf['MAE']:.4f}")
    with col2_sidebar:
        st.metric("RMSE", f"{perf['RMSE']:.4f}")
        st.metric("MSE", f"{perf['MSE']:.4f}")

st.sidebar.markdown("---")

# Model comparison
with st.sidebar.expander("🔍 Compare All Models"):
    df_performance = pd.DataFrame(model_performance).T
    df_performance = df_performance.sort_values('R2', ascending=False)
    st.dataframe(df_performance.round(4))

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 Input Student Details")
    
    # Create input form
    with st.form("prediction_form"):
        col1_form, col2_form = st.columns(2)
        
        with col1_form:
            hours_studied = st.number_input(
                'Hours Studied per Day',
                min_value=0.0,
                max_value=24.0,
                value=5.0,
                step=0.5,
                help="Number of hours the student studies per day"
            )
            
            previous_score = st.number_input(
                'Previous Score (%)',
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=1.0,
                help="Student's previous academic score"
            )
            
            sleep_hours = st.number_input(
                'Sleep Hours per Day',
                min_value=0.0,
                max_value=24.0,
                value=7.0,
                step=0.5,
                help="Number of hours the student sleeps per day"
            )
        
        with col2_form:
            extra_curr_activities = st.selectbox(
                'Extra Curricular Activities',
                options=['Yes', 'No'],
                help="Does the student participate in extra curricular activities?"
            )
            
            simple_ques_papers_prac = st.number_input(
                'Sample Question Papers Practiced',
                min_value=0.0,
                max_value=20.0,
                value=5.0,
                step=1.0,
                help="Number of sample question papers practiced"
            )
        
        predict_button = st.form_submit_button("🔮 Predict Performance", use_container_width=True)

with col2:
    st.header("🎯 Prediction Result")
    
    if predict_button:
        try:
            # Prepare input DataFrame
            input_df = pd.DataFrame({
                'hours_studied': [hours_studied],
                'previous_score': [previous_score],
                'extra_curr_activities': [extra_curr_activities],
                'sleep_hours': [sleep_hours],
                'simple_ques_papers_prac': [simple_ques_papers_prac]
            })
            
            # Preprocess the input
            input_transformed = preprocessor.transform(input_df)
            
            # Make prediction
            model = models[selected_model]
            prediction = model.predict(input_transformed)[0]
            
            # Display prediction
            st.success(f"**Predicted Performance Index:** {prediction:.2f}")
            
            # Performance interpretation
            if prediction >= 90:
                st.balloons()
                st.success("🌟 Excellent Performance Expected!")
            elif prediction >= 80:
                st.info("🎯 Very Good Performance Expected!")
            elif prediction >= 70:
                st.info("👍 Good Performance Expected!")
            elif prediction >= 60:
                st.warning("⚠️ Average Performance Expected")
            else:
                st.error("📚 Need More Study Time!")
            
            # Show input summary
            with st.expander("📋 Input Summary"):
                st.json({
                    "Hours Studied": hours_studied,
                    "Previous Score": previous_score,
                    "Extra Curricular Activities": extra_curr_activities,
                    "Sleep Hours": sleep_hours,
                    "Sample Papers Practiced": simple_ques_papers_prac,
                    "Model Used": selected_model
                })
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    else:
        st.info("👆 Fill in the form and click 'Predict Performance' to get results")

# Additional features
st.markdown("---")
st.header("🚀 Advanced Features")

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["📊 Batch Prediction", "🧠 Model Insights", "💡 Performance Tips"])

with tab1:
    st.subheader("Upload CSV for Batch Predictions")
    st.write("Upload a CSV file with columns: hours_studied, previous_score, extra_curr_activities, sleep_hours, simple_ques_papers_prac")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            
            required_cols = ['hours_studied', 'previous_score', 'extra_curr_activities', 'sleep_hours', 'simple_ques_papers_prac']
            if all(col in batch_df.columns for col in required_cols):
                st.success("✅ CSV file format is correct!")
                
                col1_batch, col2_batch = st.columns([1, 1])
                with col1_batch:
                    st.write("**Sample Data Preview:**")
                    st.dataframe(batch_df.head())
                
                with col2_batch:
                    if st.button("🔮 Generate Predictions", type="primary"):
                        # Preprocess and predict
                        batch_transformed = preprocessor.transform(batch_df)
                        batch_predictions = models[selected_model].predict(batch_transformed)
                        
                        # Add predictions to dataframe
                        batch_df['predicted_performance_index'] = batch_predictions
                        
                        st.success(f"✅ Predictions completed using {selected_model}!")
                        st.dataframe(batch_df)
                        
                        # Download button
                        csv = batch_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions as CSV",
                            data=csv,
                            file_name=f"predictions_{selected_model}.csv",
                            mime="text/csv",
                            type="primary"
                        )
            else:
                st.error("❌ CSV must contain columns: hours_studied, previous_score, extra_curr_activities, sleep_hours, simple_ques_papers_prac")
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

with tab2:
    st.subheader("📈 Model Performance Comparison")
    df_performance = pd.DataFrame(model_performance).T
    df_performance = df_performance.sort_values('R2', ascending=False)
    
    # Create a more detailed comparison
    col1_insights, col2_insights = st.columns([2, 1])
    
    with col1_insights:
        st.dataframe(
            df_performance.round(4),
            use_container_width=True,
            height=400
        )
    
    with col2_insights:
        st.markdown("### 🏆 Top 3 Models")
        top_3 = df_performance.head(3)
        for i, (model, perf) in enumerate(top_3.iterrows(), 1):
            if i == 1:
                st.success(f"🥇 **{model}**\nR² = {perf['R2']:.4f}")
            elif i == 2:
                st.info(f"🥈 **{model}**\nR² = {perf['R2']:.4f}")
            else:
                st.warning(f"🥉 **{model}**\nR² = {perf['R2']:.4f}")

with tab3:
    col1_tips, col2_tips = st.columns(2)
    
    with col1_tips:
        st.markdown("""
        ### 📚 Study Optimization Tips
        - **Consistency is Key**: 4-6 hours of daily study is more effective than irregular long sessions
        - **Active Learning**: Practice with sample papers regularly
        - **Quality over Quantity**: Focus on understanding concepts deeply
        - **Time Management**: Break study sessions into focused blocks
        """)
    
    with col2_tips:
        st.markdown("""
        ### 🌟 Performance Enhancement
        - **Sleep**: 7-8 hours optimizes cognitive performance
        - **Balance**: Extra curricular activities enhance overall development
        - **Review**: Regular revision strengthens memory retention
        - **Health**: Physical activity improves mental clarity
        """)
    
    st.info("💡 **Pro Tip**: The best performing models (Linear Regression, Ridge, BayesianRidge) all show R² scores of 0.9884, indicating excellent predictive accuracy!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 20px;'>
        <p style='color: #666;'>
            🎓 <strong>Student Performance Index Predictor</strong> - Powered by Machine Learning<br>
            Built with Streamlit • Using your trained models from .joblib files
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
