import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
MODEL_OUTPUT_PATH = 'pcos_early_detection_model.joblib'
RESULTS_PATH = 'model_evaluation_results.txt'
FEATURE_IMPORTANCE_PATH = 'feature_importance.png'
COMPARISON_GRAPH_PATH = 'model_comparison_graph.png'
CONFUSION_MATRIX_DIR = 'confusion_matrix_plots'

def load_and_prepare_data():
    """
    Load and prepare the PCOS dataset for model training
    """
    print("Loading and preparing data...")
    
    try:
        # Load the CSV file
        df = pd.read_csv('../data/PCOS_infertility.csv')
        print("Loaded PCOS_infertility.csv")
        
        # Rename target column for consistency
        if 'PCOS (Y/N)' in df.columns:
            df.rename(columns={'PCOS (Y/N)': 'PCOS'}, inplace=True)
        
        # Drop ID columns
        df = df.drop(columns=['Sl. No', 'Patient File No.'], errors='ignore')
        
        # Handle missing values in hormone measurements
        # 1.99 is a placeholder for missing values in this dataset
        hormone_cols = ['  I   beta-HCG(mIU/mL)', 'II    beta-HCG(mIU/mL)', 'AMH(ng/mL)']
        for col in hormone_cols:
            df[col] = df[col].replace(1.99, np.nan)
        
        # Fill missing values with median
        for col in df.columns:
            if col != 'PCOS' and df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
        
        # Separate features and target
        X = df.drop(columns=['PCOS'])
        y = df['PCOS']
        
        print(f"Data loaded successfully. Features: {X.shape}, Target: {y.shape}")
        return X, y
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def build_model_pipeline(X):
    """
    Build a machine learning pipeline for PCOS early detection
    """
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
    # Preprocessing for numerical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create the full pipeline with Random Forest classifier
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    return pipeline

def plot_confusion_matrix(cm, model_name, output_dir):
    """
    Plots and saves the confusion matrix as a heatmap.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No PCOS', 'PCOS'], 
                yticklabels=['No PCOS', 'PCOS'])
    plt.title(f'Confusion Matrix for {model_name}', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    file_path = os.path.join(output_dir, f'confusion_matrix_{model_name.replace(" ", "_")}.png')
    plt.savefig(file_path)
    plt.close()
    print(f"Confusion matrix plot for {model_name} saved to {file_path}")

def train_and_evaluate_model(X, y):
    """
    Train and evaluate the PCOS early detection model
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}\n")
    
    # Calculate scale_pos_weight for XGBoost to handle class imbalance
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    print(f"Calculated scale_pos_weight for XGBoost: {scale_pos_weight:.2f}")

    # Define models and their hyperparameter grids
    models = {
        # Use class_weight='balanced' to handle class imbalance
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42), # GB doesn't have a simple 'balanced' option, will rely on scoring
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
    }
    
    param_grids = {
        'Random Forest': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [10, 20],
            'classifier__min_samples_split': [2, 5]
        },
        'Gradient Boosting': {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.05, 0.1],
            'classifier__max_depth': [3, 5]
        },
        'XGBoost': {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.05, 0.1],
            'classifier__max_depth': [3, 5]
        }
    }
    
    best_model = None
    best_f1_score = 0.0
    all_results = {}
    
    # Loop through each model to train and evaluate
    for model_name, model in models.items():
        print(f"--- Training and Evaluating {model_name} ---")
        
        # Create the full pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', build_model_pipeline(X).named_steps['preprocessor']),
            ('classifier', model)
        ])
        
        # Perform grid search
        grid_search = GridSearchCV(pipeline, param_grids[model_name], cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Get the best model from grid search
        current_best_model = grid_search.best_estimator_
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        
        # Make predictions
        y_pred = current_best_model.predict(X_test)
        y_prob = current_best_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, model_name, CONFUSION_MATRIX_DIR)
        
        all_results[model_name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1, 
            'ROC AUC': roc_auc,
            'Confusion Matrix': cm,
            'Best Params': grid_search.best_params_
        }
        
        # Check if this is the best model so far
        if f1 > best_f1_score:
            best_f1_score = f1
            best_model = current_best_model
            print(f"New best model found: {model_name} with F1-Score: {f1:.4f}\n")

    # --- Save all results to a file ---
    with open(RESULTS_PATH, 'w') as f:
        f.write("PCOS Early Detection - Model Comparison Results\n")
        f.write("===============================================\n\n")
        for model_name, metrics in all_results.items():
            f.write(f"--- {model_name} ---\n")
            f.write(f"Accuracy: {metrics['Accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['Precision']:.4f}\n")
            f.write(f"Recall: {metrics['Recall']:.4f}\n")
            f.write(f"F1 Score: {metrics['F1 Score']:.4f}\n")
            f.write(f"ROC AUC: {metrics['ROC AUC']:.4f}\n")
            f.write(f"Confusion Matrix:\n{metrics['Confusion Matrix']}\n")
            f.write(f"Best Parameters: {metrics['Best Params']}\n\n")
        
        f.write(f"\n--- Best Performing Model ---\n")
        f.write(f"Model: {best_model.named_steps['classifier'].__class__.__name__}\n")
        f.write(f"F1-Score: {best_f1_score:.4f}\n")
    print(f"Detailed model evaluation results saved to {RESULTS_PATH}")

    # --- Create and save comparison graph ---
    model_names = list(all_results.keys())
    f1_scores = [res['F1 Score'] for res in all_results.values()]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=model_names, y=f1_scores, palette='viridis')
    plt.title('Model Comparison by F1-Score', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.ylim(0, 1)
    for i, score in enumerate(f1_scores):
        plt.text(i, score + 0.01, f'{score:.3f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(COMPARISON_GRAPH_PATH)
    print(f"Model comparison graph saved to {COMPARISON_GRAPH_PATH}")
    
    # Extract feature importance if using Random Forest
    if isinstance(best_model.named_steps['classifier'], RandomForestClassifier):
        # Get feature names after preprocessing
        feature_names = []
        
        # For numeric features
        if hasattr(best_model.named_steps['preprocessor'].transformers_[0][1], 'get_feature_names_out'):
            feature_names.extend(
                best_model.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out(
                    X.select_dtypes(include=['int64', 'float64']).columns
                )
            )
        else:
            feature_names.extend(X.select_dtypes(include=['int64', 'float64']).columns)
        
        # For categorical features (if any)
        if len(best_model.named_steps['preprocessor'].transformers_) > 1 and len(X.select_dtypes(include=['object', 'category']).columns) > 0:
            if hasattr(best_model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'], 'get_feature_names_out'):
                feature_names.extend(
                    best_model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(
                        X.select_dtypes(include=['object', 'category']).columns
                    )
                )
        
        # Get feature importances
        importances = best_model.named_steps['classifier'].feature_importances_
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        if len(feature_names) == len(importances):
            indices = np.argsort(importances)[-20:]  # Get indices of top 20 features
            plt.title('Top 20 Feature Importances for PCOS Early Detection')
            plt.barh(range(len(indices)), importances[indices], align='center')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Relative Importance')
            plt.tight_layout()
            plt.savefig(FEATURE_IMPORTANCE_PATH)
            print(f"Feature importance plot saved to {FEATURE_IMPORTANCE_PATH}")
        else:
            print(f"Warning: Feature names ({len(feature_names)}) and importances ({len(importances)}) length mismatch")
    
    return best_model

def save_model(model):
    """
    Save the trained model to disk
    """
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"Model saved to {MODEL_OUTPUT_PATH}")

def main():
    """
    Main function to execute the PCOS early detection model pipeline
    """
    print("Starting PCOS Early Detection Model Development")
    
    # Load and prepare data
    X, y = load_and_prepare_data()
    if X is None or y is None:
        print("Failed to load or prepare data. Exiting.")
        return
    
    # Train and evaluate model
    model = train_and_evaluate_model(X, y)
    
    # Save the model
    save_model(model)
    
    print("PCOS Early Detection Model Development Complete")

if __name__ == "__main__":
    main()