import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="💰 Salary Prediction System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .css-1d391kg {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .prediction-result {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_and_data():
    """Load trained model and preprocessing objects"""
    try:
        model = joblib.load('best_salary_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        model_results = joblib.load('model_results.pkl')
        
        # Load sample data for reference
        df = pd.read_csv('clean_salary.csv')
        
        return model, scaler, feature_names, model_results, df
    except FileNotFoundError as e:
        st.error(f"Model files not found! Please run training.py first. Error: {e}")
        return None, None, None, None, None

def create_feature_importance_plot(model, feature_names):
    """Create feature importance plot"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        fig = px.bar(
            feature_imp.head(10), 
            x='importance', 
            y='feature',
            orientation='h',
            title="🔍 Top 10 Feature Importance",
            color='importance',
            color_continuous_scale='viridis'
        )
        fig.update_layout(
            height=400,
            xaxis_title="Importance Score",
            yaxis_title="Features"
        )
        return fig
    return None

def create_model_comparison_chart(model_results):
    """Create interactive model comparison chart"""
    models = list(model_results.keys())
    r2_scores = [model_results[model]['r2_score'] * 100 for model in models]
    
    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=r2_scores,
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'],
            text=[f'{score:.1f}%' for score in r2_scores],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="🏆 Model Performance Comparison",
        xaxis_title="Models",
        yaxis_title="Accuracy (%)",
        height=400,
        showlegend=False
    )
    
    return fig

def get_user_input(df, feature_names):
    """Get user input for prediction"""
    st.sidebar.header("🎯 Enter Your Details")
    
    user_input = {}
    
    # Common salary prediction features
    common_features = {
        'age': {'min': 18, 'max': 70, 'default': 30},
        'years_experience': {'min': 0, 'max': 50, 'default': 5},
        'education_level': {'options': ['High School', 'Bachelor', 'Master', 'PhD']},
        'job_level': {'options': ['Entry', 'Mid', 'Senior', 'Executive']},
        'company_size': {'options': ['Small', 'Medium', 'Large']},
        'location': {'options': ['Urban', 'Suburban', 'Rural']},
    }
    
    # Generate inputs based on available features
    for feature in feature_names:
        if 'age' in feature.lower():
            user_input[feature] = st.sidebar.slider(
                "👤 Age", 
                min_value=18, 
                max_value=70, 
                value=30
            )
        elif 'experience' in feature.lower() or 'year' in feature.lower():
            user_input[feature] = st.sidebar.slider(
                "💼 Years of Experience", 
                min_value=0, 
                max_value=50, 
                value=5
            )
        elif 'education' in feature.lower():
            user_input[feature] = st.sidebar.selectbox(
                "🎓 Education Level",
                options=[0, 1, 2, 3],
                format_func=lambda x: ['High School', 'Bachelor', 'Master', 'PhD'][x]
            )
        elif any(word in feature.lower() for word in ['level', 'position', 'rank']):
            user_input[feature] = st.sidebar.selectbox(
                "📊 Job Level",
                options=[0, 1, 2, 3],
                format_func=lambda x: ['Entry', 'Mid', 'Senior', 'Executive'][x]
            )
        elif 'size' in feature.lower() or 'company' in feature.lower():
            user_input[feature] = st.sidebar.selectbox(
                "🏢 Company Size",
                options=[0, 1, 2],
                format_func=lambda x: ['Small (<100)', 'Medium (100-1000)', 'Large (>1000)'][x]
            )
        else:
            # For other numeric features, use the data range
            if feature in df.columns:
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                mean_val = float(df[feature].mean())
                user_input[feature] = st.sidebar.slider(
                    f"📈 {feature.replace('_', ' ').title()}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val
                )
            else:
                user_input[feature] = st.sidebar.number_input(
                    f"📊 {feature.replace('_', ' ').title()}",
                    value=0.0
                )
    
    return user_input

def main():
    # Header
    st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <h1 style="color: white; font-size: 3em; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
                💰 Salary Prediction System
            </h1>
            <p style="color: white; font-size: 1.2em; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">
                Predict your expected salary using advanced machine learning models
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load model and data
    model, scaler, feature_names, model_results, df = load_model_and_data()
    
    if model is None:
        st.error("❌ Please run the training script first to generate the required model files!")
        return
    
    # Sidebar for user input
    with st.sidebar:
        st.image("https://via.placeholder.com/300x150/667eea/white?text=AI+Salary+Predictor", 
                caption="AI-Powered Salary Prediction")
        
        user_input = get_user_input(df, feature_names)
        
        predict_button = st.button("🚀 Predict Salary", type="primary")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if predict_button:
            # Prepare input for prediction
            input_df = pd.DataFrame([user_input])
            if scaler is not None:
                input_scaled = scaler.transform(input_df)
            else:
                st.error("Scaler not found. Please ensure 'scaler.pkl' exists and is loaded correctly.")
                return
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            
            # Display prediction
            st.markdown(f"""
                <div class="prediction-result">
                    <h2>💰 Predicted Salary</h2>
                    <h1>${prediction:,.0f}</h1>
                    <p>Based on your profile and our AI model</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Confidence interval (rough estimate)
            confidence_range = prediction * 0.1  # ±10%
            st.success(f"📊 Salary Range: ${prediction-confidence_range:,.0f} - ${prediction+confidence_range:,.0f}")
            
            # Show input summary
            st.subheader("📋 Your Input Summary")
            input_summary = pd.DataFrame.from_dict(user_input, orient='index', columns=['Value'])
            input_summary.index.name = 'Feature'
            st.dataframe(input_summary, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Model Performance")
        
        if model_results:
            best_model = max(model_results.items(), key=lambda x: x[1]['r2_score'])
            st.markdown(f"""
                <div class="metric-card">
                    <h3>Best Model</h3>
                    <h2>{best_model[0]}</h2>
                    <p>Accuracy: {best_model[1]['r2_score']*100:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Model metrics
            metrics = best_model[1]
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("R² Score", f"{metrics['r2_score']:.3f}")
                st.metric("RMSE", f"{metrics['rmse']:.0f}")
            with col_b:
                st.metric("MAE", f"{metrics['mae']:.0f}")
                st.metric("Accuracy", f"{metrics['accuracy']:.1f}%")
    
    # Charts section
    st.markdown("---")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        if model_results:
            fig_comparison = create_model_comparison_chart(model_results)
            st.plotly_chart(fig_comparison, use_container_width=True)
    
    with chart_col2:
        fig_importance = create_feature_importance_plot(model, feature_names)
        if fig_importance:
            st.plotly_chart(fig_importance, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type")
    
    # Data insights section
    st.markdown("---")
    st.subheader("📊 Dataset Insights")
    
    if df is not None:
        insight_col1, insight_col2, insight_col3 = st.columns(3)
        
        with insight_col1:
            st.metric("Total Records", f"{len(df):,}")
            st.metric("Features", f"{len(feature_names) if feature_names is not None else 0}")
        
        with insight_col2:
            # Find salary column
            salary_col = None
            for col in df.columns:
                if 'salary' in col.lower() and '_encoded' not in col:
                    salary_col = col
                    break
            
            if salary_col:
                st.metric("Average Salary", f"${df[salary_col].mean():,.0f}")
                st.metric("Salary Range", f"${df[salary_col].min():,.0f} - ${df[salary_col].max():,.0f}")
        
        with insight_col3:
            st.metric("Data Quality", "✅ Cleaned")
            st.metric("Model Status", "🎯 Ready")
        
        # Salary distribution chart
        if salary_col:
            st.subheader("💹 Salary Distribution")
            fig_dist = px.histogram(
                df, 
                x=salary_col, 
                nbins=30,
                title="Salary Distribution in Dataset",
                color_discrete_sequence=['#667eea']
            )
            fig_dist.update_layout(
                xaxis_title="Salary ($)",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig_dist, use_container_width=True)
    
    # Instructions and tips
    st.markdown("---")
    st.subheader("📖 How to Use")
    
    col_inst1, col_inst2 = st.columns(2)
    
    with col_inst1:
        st.markdown("""
        **🎯 Step-by-Step Guide:**
        1. 📝 Fill in your details in the sidebar
        2. 🎚️ Adjust the sliders and dropdowns
        3. 🚀 Click "Predict Salary" button
        4. 📊 View your predicted salary and insights
        """)
    
    with col_inst2:
        st.markdown("""
        **💡 Tips for Better Predictions:**
        - ✅ Provide accurate information
        - 🎓 Education level affects salary significantly
        - 💼 Experience is a key factor
        - 🏢 Company size matters
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; padding: 20px; color: white;">
            <p>🤖 Powered by Advanced Machine Learning | 📊 Built with Streamlit</p>
            <p>💡 For best results, ensure all input values are accurate</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()