import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class SalaryPredictor:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_and_prepare_data(self):
        """Load and prepare the cleaned dataset"""
        print("📊 Loading cleaned dataset...")
        
        try:
            df = pd.read_csv('clean_salary.csv')
            print(f"✅ Dataset loaded! Shape: {df.shape}")
        except FileNotFoundError:
            print("❌ Error: clean_salary.csv not found! Please run cleaning.py first.")
            return None, None, None, None
        
        # Identify target column (salary-related)
        target_col = None
        for col in df.columns:
            if 'salary' in col.lower() and '_encoded' not in col:
                target_col = col
                break
        
        if target_col is None:
            print("❌ Error: No salary column found!")
            return None, None, None, None
        
        print(f"🎯 Target column: {target_col}")
        
        # Prepare features and target
        X = df.drop([target_col], axis=1)
        y = df[target_col]
        
        # Select numeric features and encoded categorical features
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        X = X[numeric_features]
        
        self.feature_names = X.columns.tolist()
        print(f"🔢 Features selected: {self.feature_names}")
        
        return X, y, df, target_col
    
    def train_models(self, X, y):
        """Train multiple models and find the best one"""
        print("🚀 Starting model training...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models with hyperparameters
        model_configs = {
            'Random Forest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7]
                }
            },
            'Linear Regression': {
                'model': LinearRegression(),
                'params': {}
            },
            'Ridge Regression': {
                'model': Ridge(),
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                }
            },
            'Lasso Regression': {
                'model': Lasso(random_state=42),
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                }
            },
            'Decision Tree': {
                'model': DecisionTreeRegressor(random_state=42),
                'params': {
                    'max_depth': [5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'SVR': {
                'model': SVR(),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01]
                }
            },
            'KNN': {
                'model': KNeighborsRegressor(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance']
                }
            }
        }
        
        results = {}
        
        for name, config in model_configs.items():
            print(f"🔧 Training {name}...")
            
            try:
                if config['params']:
                    # Use GridSearchCV for hyperparameter tuning
                    grid_search = GridSearchCV(
                        config['model'], 
                        config['params'], 
                        cv=5, 
                        scoring='r2',
                        n_jobs=-1
                    )
                    
                    # Use scaled data for models that benefit from it
                    if name in ['SVR', 'KNN', 'Ridge Regression', 'Lasso Regression', 'Linear Regression']:
                        grid_search.fit(X_train_scaled, y_train)
                        y_pred = grid_search.predict(X_test_scaled)
                    else:
                        grid_search.fit(X_train, y_train)
                        y_pred = grid_search.predict(X_test)
                    
                    best_model = grid_search.best_estimator_
                else:
                    # For models without hyperparameters to tune
                    if name in ['Linear Regression']:
                        config['model'].fit(X_train_scaled, y_train)
                        y_pred = config['model'].predict(X_test_scaled)
                    else:
                        config['model'].fit(X_train, y_train)
                        y_pred = config['model'].predict(X_test)
                    
                    best_model = config['model']
                
                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                
                results[name] = {
                    'model': best_model,
                    'r2_score': r2,
                    'rmse': rmse,
                    'mae': mae,
                    'accuracy': r2 * 100
                }
                
                print(f"✅ {name} - R² Score: {r2:.4f} ({r2*100:.2f}%)")
                
                # Update best model if this one is better
                if r2 > self.best_score:
                    self.best_score = r2
                    self.best_model = best_model
                    self.best_model_name = name
                
            except Exception as e:
                print(f"❌ Error training {name}: {str(e)}")
                continue
        
        self.models = results
        return results
    
    def save_models(self):
        """Save the best model and preprocessing objects"""
        print(f"💾 Saving best model: {self.best_model_name}")
        print(f"🎯 Best R² Score: {self.best_score:.4f} ({self.best_score*100:.2f}%)")
        
        # Save the best model
        joblib.dump(self.best_model, 'best_salary_model.pkl')
        
        # Save the scaler
        joblib.dump(self.scaler, 'scaler.pkl')
        
        # Save feature names
        joblib.dump(self.feature_names, 'feature_names.pkl')
        
        # Save model results
        joblib.dump(self.models, 'model_results.pkl')
        
        print("✅ All models and preprocessing objects saved!")
    
    def create_visualization(self):
        """Create and save model comparison visualization"""
        if not self.models:
            return
        
        plt.style.use('seaborn-v0_8')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        model_names = list(self.models.keys())
        r2_scores = [self.models[name]['r2_score'] for name in model_names]
        rmse_scores = [self.models[name]['rmse'] for name in model_names]
        mae_scores = [self.models[name]['mae'] for name in model_names]
        accuracy_scores = [self.models[name]['accuracy'] for name in model_names]
        
        # R² Score comparison
        bars1 = ax1.bar(model_names, r2_scores, color='skyblue', alpha=0.8)
        ax1.set_title('R² Score Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('R² Score')
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # RMSE comparison
        bars2 = ax2.bar(model_names, rmse_scores, color='lightcoral', alpha=0.8)
        ax2.set_title('RMSE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('RMSE')
        ax2.tick_params(axis='x', rotation=45)
        
        # MAE comparison
        bars3 = ax3.bar(model_names, mae_scores, color='lightgreen', alpha=0.8)
        ax3.set_title('MAE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Mean Absolute Error')
        ax3.tick_params(axis='x', rotation=45)
        
        # Accuracy percentage
        bars4 = ax4.bar(model_names, accuracy_scores, color='gold', alpha=0.8)
        ax4.set_title('Model Accuracy (%)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Accuracy (%)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.set_ylim(0, 100)
        
        # Add value labels on accuracy bars
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig('model_comparison.jpg', dpi=300, bbox_inches='tight')
        print("📊 Model comparison plots saved!")
        
    def print_summary(self):
        """Print training summary"""
        print("\n" + "="*60)
        print("🎉 TRAINING SUMMARY")
        print("="*60)
        print(f"Best Model: {self.best_model_name}")
        print(f"Best R² Score: {self.best_score:.4f} ({self.best_score*100:.2f}%)")
        
        print(f"\n📊 All Model Results:")
        for name, results in self.models.items():
            print(f"{name:<20} | R²: {results['r2_score']:.4f} | Accuracy: {results['accuracy']:.2f}%")
        
        if self.best_score >= 0.95:
            print(f"\n🎯 SUCCESS! Achieved target accuracy of ≥95%!")
        else:
            print(f"\n⚠️  Best accuracy: {self.best_score*100:.2f}% (Target: 95%)")
            print("   Consider feature engineering or more data for better performance.")

def main():
    predictor = SalaryPredictor()
    
    # Load and prepare data
    X, y, df, target_col = predictor.load_and_prepare_data()
    
    if X is not None:
        # Train models
        results = predictor.train_models(X, y)
        
        # Save models
        predictor.save_models()
        
        # Create visualizations
        predictor.create_visualization()
        
        # Print summary
        predictor.print_summary()

if __name__ == "__main__":
    main()