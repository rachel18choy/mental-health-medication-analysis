"""
Supervised learning: Classification model for antipsychotic attitude prediction
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import yaml
import os

class SupervisedPsychosisClassifier:
    def __init__(self):
        self.config = self.load_config()
        self.models = {}
        self.results = {}
        
    def load_config(self):
        with open('../config/config.yaml', 'r') as file:
            return yaml.safe_load(file)
    
    def load_processed_data(self):
        """Load preprocessed data for supervised learning"""
        data_path = '../data/processed/supervised_data.pkl'
        if os.path.exists(data_path):
            data = joblib.load(data_path)
            print("Supervised learning data loaded successfully")
            print(f"Training samples: {data['X_train'].shape[0]}")
            print(f"Test samples: {data['X_test'].shape[0]}")
            print(f"Number of features: {len(data['feature_names'])}")
            return data
        else:
            raise FileNotFoundError("Supervised data not found. Run preprocessing first.")
    
    def train_classification_models(self):
        """Train multiple classification models"""
        data = self.load_processed_data()
        
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        feature_names = data['feature_names']
        
        print("\n" + "="*60)
        print("SUPERVISED LEARNING: TRAINING CLASSIFICATION MODELS")
        print("="*60)
        
        # Define models
        model_definitions = {
            'Logistic Regression': LogisticRegression(
                random_state=self.config['model']['random_state'],
                max_iter=1000,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.config['model']['random_state'],
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.config['model']['random_state']
            ),
            'Support Vector Machine': SVC(
                random_state=self.config['model']['random_state'],
                probability=True,
                class_weight='balanced'
            )
        }
        
        # Train and evaluate each model
        for model_name, model in model_definitions.items():
            print(f"\nTraining {model_name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                self.models[model_name] = model
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1': f1_score(y_test, y_pred, zero_division=0)
                }
                
                if y_pred_proba is not None:
                    try:
                        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                    except:
                        metrics['roc_auc'] = 0.5
                
                # Cross-validation scores
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
                metrics['cv_f1_mean'] = cv_scores.mean()
                metrics['cv_f1_std'] = cv_scores.std()
                
                self.results[model_name] = metrics
                
                print(f"  Accuracy:    {metrics['accuracy']:.4f}")
                print(f"  Precision:   {metrics['precision']:.4f}")
                print(f"  Recall:      {metrics['recall']:.4f}")
                print(f"  F1-Score:    {metrics['f1']:.4f}")
                if 'roc_auc' in metrics:
                    print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
                print(f"  CV F1-Score: {metrics['cv_f1_mean']:.4f} (±{metrics['cv_f1_std']:.4f})")
                
            except Exception as e:
                print(f"  Error: {e}")
                self.results[model_name] = {'error': str(e)}
        
        return self.results
    
    def analyze_feature_importance(self):
        """Analyze feature importance for tree-based models"""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        data = self.load_processed_data()
        feature_names = data['feature_names']
        
        # Create results directory
        os.makedirs('../results/supervised', exist_ok=True)
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                print(f"\n{model_name} Feature Importance:")
                
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                # Print top 10 features
                print("Top 10 most important features:")
                for i in range(min(10, len(feature_names))):
                    print(f"  {i+1:2d}. {feature_names[indices[i]]:50s} {importances[indices[i]]:.4f}")
                
                # Plot feature importance
                plt.figure(figsize=(10, 6))
                plt.title(f'{model_name} - Top 10 Feature Importances')
                plt.barh(range(10), importances[indices][:10][::-1])
                plt.yticks(range(10), [feature_names[i] for i in indices[:10][::-1]])
                plt.xlabel('Relative Importance')
                plt.tight_layout()
                plt.savefig(f'../results/supervised/{model_name.lower().replace(" ", "_")}_feature_importance.png')
                plt.close()
                
                print(f"  Plot saved to: ../results/supervised/{model_name.lower().replace(' ', '_')}_feature_importance.png")
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        print("\nGenerating confusion matrices...")
        
        data = self.load_processed_data()
        X_test = data['X_test']
        y_test = data['y_test']
        
        for model_name, model in self.models.items():
            try:
                y_pred = model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Not Helpful', 'Helpful'],
                           yticklabels=['Not Helpful', 'Helpful'])
                plt.title(f'{model_name} - Confusion Matrix')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                plt.savefig(f'../results/supervised/{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
                plt.close()
                
                print(f"  Confusion matrix for {model_name} saved")
                
            except Exception as e:
                print(f"  Error creating confusion matrix for {model_name}: {e}")
    
    def plot_model_comparison(self):
        """Create comparison plot of all models"""
        print("\nCreating model comparison visualization...")
        
        # Filter out models with errors
        valid_results = {}
        for model_name, metrics in self.results.items():
            if 'error' not in metrics:
                valid_results[model_name] = metrics
        
        if not valid_results:
            print("No valid models to compare")
            return
        
        results_df = pd.DataFrame(valid_results).T
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in results_df.columns:
                plt.subplot(2, 2, i + 1)
                results_df[metric].plot(kind='bar', color=colors[i])
                plt.title(f'{metric.capitalize()} Comparison')
                plt.xticks(rotation=45)
                plt.ylim(0, 1)
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../results/supervised/model_comparison.png')
        plt.close()
        
        print("Model comparison plot saved to: ../results/supervised/model_comparison.png")
    
    def save_models_and_results(self):
        """Save trained models and results"""
        print("\n" + "="*60)
        print("SAVING MODELS AND RESULTS")
        print("="*60)
        
        # Create directories
        os.makedirs('../models/saved_models/supervised', exist_ok=True)
        os.makedirs('../results/supervised', exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            model_filename = f"../models/saved_models/supervised/{model_name.lower().replace(' ', '_')}.pkl"
            joblib.dump(model, model_filename)
            print(f"Saved {model_name} model to: {model_filename}")
        
        # Save results
        results_df = pd.DataFrame(self.results).T
        results_df.to_csv('../results/supervised/model_results.csv')
        
        # Create summary report
        with open('../results/supervised/summary_report.txt', 'w') as f:
            f.write("SUPERVISED LEARNING - CLASSIFICATION RESULTS\n")
            f.write("="*70 + "\n\n")
            
            f.write("Model Performance Summary:\n")
            for model_name, metrics in self.results.items():
                if 'error' not in metrics:
                    f.write(f"\n{model_name}:\n")
                    f.write(f"  Accuracy:  {metrics.get('accuracy', 'N/A'):.4f}\n")
                    f.write(f"  Precision: {metrics.get('precision', 'N/A'):.4f}\n")
                    f.write(f"  Recall:    {metrics.get('recall', 'N/A'):.4f}\n")
                    f.write(f"  F1-Score:  {metrics.get('f1', 'N/A'):.4f}\n")
                    if 'roc_auc' in metrics:
                        f.write(f"  ROC-AUC:   {metrics['roc_auc']:.4f}\n")
        
        print("\nResults saved to: ../results/supervised/")
        print("Models saved to: ../models/saved_models/supervised/")
    
    def run(self):
        """Main execution method"""
        print("="*70)
        print("SUPERVISED LEARNING: ANTIPSYCHOTIC ATTITUDE CLASSIFIER")
        print("="*70)
        
        # Create directories
        os.makedirs('../results/supervised', exist_ok=True)
        
        # Train models
        results = self.train_classification_models()
        
        # Display results
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        
        # Filter out models with errors
        valid_results = {}
        for model_name, metrics in results.items():
            if 'error' not in metrics:
                valid_results[model_name] = metrics
        
        if valid_results:
            results_df = pd.DataFrame(valid_results).T
            
            # Display key metrics
            display_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'cv_f1_mean']
            available_metrics = [m for m in display_metrics if m in results_df.columns]
            
            if available_metrics:
                print(results_df[available_metrics].round(4))
                
                # Find best model by F1-Score
                if 'f1' in results_df.columns:
                    best_model = results_df['f1'].idxmax()
                    print(f"\nBest model (by F1-Score): {best_model}")
                    print(f"F1-Score: {results_df.loc[best_model, 'f1']:.4f}")
        
        # Generate analyses and visualizations
        if self.models:
            self.analyze_feature_importance()
            self.plot_confusion_matrices()
            self.plot_model_comparison()
        
        # Save everything
        self.save_models_and_results()
        
        print("\n" + "="*70)
        print("SUPERVISED LEARNING COMPLETE")
        print("="*70)

if __name__ == "__main__":
    classifier = SupervisedPsychosisClassifier()
    classifier.run()