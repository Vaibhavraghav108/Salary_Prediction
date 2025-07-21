import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

def clean_salary_data():
    """
    Clean and preprocess the Salary_Data.csv file
    """
    print("🧹 Starting data cleaning process...")
    
    # Load the dataset
    try:
        df = pd.read_csv('Salary_Data.csv')
        print(f"✅ Dataset loaded successfully! Shape: {df.shape}")
    except FileNotFoundError:
        print("❌ Error: Salary_Data.csv not found!")
        return
    
    print(f"📊 Original dataset info:")
    print(f"Columns: {list(df.columns)}")
    print(f"Null values:\n{df.isnull().sum()}")
    
    # Create a copy for cleaning
    clean_df = df.copy()
    
    # Handle missing values
    if clean_df.isnull().sum().sum() > 0:
        print("🔧 Handling missing values...")
        # Fill numeric columns with median
        numeric_cols = clean_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            clean_df[col].fillna(clean_df[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = clean_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            clean_df[col].fillna(clean_df[col].mode()[0] if len(clean_df[col].mode()) > 0 else 'Unknown', inplace=True)
    
    # Remove duplicates
    initial_rows = len(clean_df)
    clean_df.drop_duplicates(inplace=True)
    removed_duplicates = initial_rows - len(clean_df)
    if removed_duplicates > 0:
        print(f"🗑️  Removed {removed_duplicates} duplicate rows")
    
    # Standardize column names
    clean_df.columns = clean_df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Handle potential outliers (using IQR method for salary)
    if 'salary' in clean_df.columns:
        Q1 = clean_df['salary'].quantile(0.25)
        Q3 = clean_df['salary'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_before = len(clean_df)
        clean_df = clean_df[(clean_df['salary'] >= lower_bound) & (clean_df['salary'] <= upper_bound)]
        outliers_removed = outliers_before - len(clean_df)
        if outliers_removed > 0:
            print(f"🎯 Removed {outliers_removed} salary outliers")
    
    # Encode categorical variables if needed
    label_encoders = {}
    categorical_columns = clean_df.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        if col != 'salary':  # Don't encode target if it's categorical
            le = LabelEncoder()
            clean_df[col + '_encoded'] = le.fit_transform(clean_df[col])
            label_encoders[col] = le
    
    # Save the cleaned dataset
    clean_df.to_csv('clean_salary.csv', index=False)
    print(f"💾 Cleaned dataset saved as 'clean_salary.csv'")
    print(f"✨ Final dataset shape: {clean_df.shape}")
    
    # Save label encoders for later use
    import joblib
    if label_encoders:
        joblib.dump(label_encoders, 'label_encoders.pkl')
        print("🔧 Label encoders saved as 'label_encoders.pkl'")
    
    print("🎉 Data cleaning completed successfully!")
    
    # Display basic statistics
    print(f"\n📈 Dataset Summary:")
    print(clean_df.describe())
    
    return clean_df

if __name__ == "__main__":
    cleaned_data = clean_salary_data()