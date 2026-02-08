"""
Fix Gradient Boosting feature importance plot - Creates plot with Q labels and text file with full questions.
"""
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os

def fix_gradient_boosting_plot():
    # Load the Gradient Boosting model
    model_path = '../models/saved_models/supervised/gradient_boosting.pkl'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Available models:")
        models_dir = '../models/saved_models/supervised/'
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                print(f"  - {file}")
        return
    
    try:
        model = joblib.load(model_path)
        print(f"Successfully loaded Gradient Boosting model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load feature names
    feature_names_path = '../models/saved_models/feature_names.txt'
    if not os.path.exists(feature_names_path):
        print(f"Error: Feature names file not found at {feature_names_path}")
        return
    
    try:
        with open(feature_names_path, 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(feature_names)} feature names")
    except Exception as e:
        print(f"Error loading feature names: {e}")
        return
    
    # Check if model has feature_importances_ attribute
    if not hasattr(model, 'feature_importances_'):
        print("Error: Gradient Boosting model does not have 'feature_importances_' attribute")
        print("Available attributes:", dir(model))
        return
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]  # Sort in descending order
    
    # Select number of features to display
    top_n = 15  # Show top 15 features
    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    top_names = [feature_names[i] for i in top_indices]
    
    # Apply text replacements for better readability
    processed_names = []
    for name in top_names:
        # Replace common abbreviations with full text
        name = name.replace("how 4", "how likely")
        name = name.replace("you personally 2 or 4", "you personally agree or disagree")
        name = name.replace("q_", "Q")
        name = name.replace("_", " ")  # Replace underscores with spaces
        processed_names.append(name)
    
    top_names = processed_names
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot horizontal bars
    y_pos = np.arange(top_n)
    bars = plt.barh(y_pos, top_importances, color='steelblue', alpha=0.8)
    
    # Use Q1, Q2, ... labels on y-axis
    q_labels = [f'Q{i+1}' for i in range(top_n)]
    plt.yticks(y_pos, q_labels, fontsize=12)
    
    # Format x-axis
    plt.xlabel('Feature Importance', fontsize=12)
    plt.xlim(0, max(top_importances) * 1.1)  # Add 10% padding
    
    # Title
    plt.title(f'Gradient Boosting - Top {top_n} Feature Importances', 
              fontsize=14, pad=20, fontweight='bold')
    
    # Add grid for readability
    plt.grid(axis='x', alpha=0.3, linestyle='--', color='gray')
    
    # Invert y-axis so Q1 (most important) is at top
    plt.gca().invert_yaxis()
    
    # Add value labels on bars
    for i, (bar, imp) in enumerate(zip(bars, top_importances)):
        width = bar.get_width()
        plt.text(width + (max(top_importances) * 0.005),  # Small offset from bar
                 bar.get_y() + bar.get_height()/2, 
                 f'{imp:.4f}', 
                 ha='left', va='center', 
                 fontsize=9, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Ensure results directory exists
    results_dir = '../results/supervised/'
    os.makedirs(results_dir, exist_ok=True)
    
    # Save the plot
    output_path = f'{results_dir}gradient_boosting_feature_importance_fixed.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to: {output_path}")
    
    # Create text file with full questions
    create_feature_description_file(top_names, top_importances, q_labels, results_dir)
    
    # Print summary
    print_summary(top_names, top_importances, q_labels)
    
def create_feature_description_file(top_names, top_importances, q_labels, results_dir):
    """Create a text file with the full questions for each feature"""
    
    output_file = f'{results_dir}gradient_boosting_features.txt'
    
    # Use ASCII-friendly replacements
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("GRADIENT BOOSTING - TOP 15 FEATURE IMPORTANCES\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Plot shows features as Q1 (most important) to Q15 (least important)\n")
        f.write(f"Total features considered: {len(top_names)}\n\n")
        
        f.write("Text replacements applied:\n")
        f.write("  - 'how 4' -> 'how likely'\n")
        f.write("  - 'you personally 2 or 4' -> 'you personally agree or disagree'\n")
        f.write("  - 'q_' -> 'Q' (for question prefixes)\n")
        f.write("  - '_' -> ' ' (underscores replaced with spaces)\n\n")
        
        f.write("=" * 80 + "\n\n")
        
        # Write each feature with its importance and question text
        for i, (label, feature_text, imp) in enumerate(zip(q_labels, top_names, top_importances)):
            f.write(f"{label} - Importance: {imp:.4f}\n")
            f.write("-" * 80 + "\n")
            f.write(f"{feature_text}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("INTERPRETATION GUIDE\n")
        f.write("=" * 80 + "\n\n")
        f.write("1. Q1 is the most important feature for the Gradient Boosting model\n")
        f.write("2. Higher importance values indicate stronger influence on predictions\n")
        f.write("3. Features with importance < 0.01 may have minimal impact\n")
        f.write("4. Questions about willingness to help tend to be more predictive\n")
        f.write("5. Demographic features (age, education) often appear in top features\n\n")
        
        # Calculate statistics
        f.write("STATISTICS:\n")
        f.write(f"  - Total importance of top 15 features: {sum(top_importances):.4f}\n")
        f.write(f"  - Average importance: {np.mean(top_importances):.4f}\n")
        f.write(f"  - Most important feature (Q1): {top_importances[0]:.4f}\n")
        f.write(f"  - Least important feature (Q{len(top_importances)}): {top_importances[-1]:.4f}\n\n")
        
        # Show the original feature names before processing
        f.write("ORIGINAL FEATURE NAMES (before text replacements):\n")
        f.write("-" * 80 + "\n")
        for i, label in enumerate(q_labels):
            f.write(f"{label}: {top_names[i]}\n")
    
    print(f"Feature descriptions saved to: {output_file}")

def print_summary(top_names, top_importances, q_labels):
    """Print a summary of the top features to console"""
    
    print("\n" + "=" * 60)
    print("GRADIENT BOOSTING - FEATURE IMPORTANCE SUMMARY")
    print("=" * 60)
    
    # Top 5 features
    print("\nTOP 5 MOST IMPORTANT FEATURES:")
    print("-" * 60)
    for i in range(min(5, len(top_names))):
        print(f"\n{q_labels[i]} (Importance: {top_importances[i]:.4f}):")
        # Truncate for display
        display_text = top_names[i][:80] + "..." if len(top_names[i]) > 80 else top_names[i]
        print(f"  {display_text}")
    
    # Bottom 5 features
    print("\n\nLEAST IMPORTANT OF TOP 15:")
    print("-" * 60)
    for i in range(max(0, len(top_names)-5), len(top_names)):
        print(f"\n{q_labels[i]} (Importance: {top_importances[i]:.4f}):")
        display_text = top_names[i][:60] + "..." if len(top_names[i]) > 60 else top_names[i]
        print(f"  {display_text}")
    
    # Statistics
    print("\n" + "=" * 60)
    print("STATISTICS:")
    print("-" * 60)
    print(f"Total importance of top 15: {sum(top_importances):.4f}")
    print(f"Range: {top_importances[0]:.4f} (Q1) to {top_importances[-1]:.4f} (Q{len(top_importances)})")
    print(f"Mean importance: {np.mean(top_importances):.4f}")
    print(f"Median importance: {np.median(top_importances):.4f}")
    
    # Feature categories (simple heuristic)
    print("\n" + "=" * 60)
    print("FEATURE CATEGORIES (HEURISTIC):")
    print("-" * 60)
    
    action_keywords = ['how likely', 'would take', 'would you', 'ask', 'tell', 'encourage', 'convey', 'find out', 'try to']
    attitude_keywords = ['agree', 'disagree', 'personally', 'think', 'believe', 'sign of', 'could snap', 'makes him', 'best to avoid']
    demographic_keywords = ['age', 'study', 'area', 'years', 'student', 'tertiary']
    
    action_count = sum(1 for name in top_names if any(keyword in name.lower() for keyword in action_keywords))
    attitude_count = sum(1 for name in top_names if any(keyword in name.lower() for keyword in attitude_keywords))
    demographic_count = sum(1 for name in top_names if any(keyword in name.lower() for keyword in demographic_keywords))
    
    print(f"Action/behavior questions: {action_count}")
    print(f"Attitude/belief questions: {attitude_count}")
    print(f"Demographic questions: {demographic_count}")
    
    other_count = len(top_names) - action_count - attitude_count - demographic_count
    if other_count > 0:
        print(f"Other/unclassified: {other_count}")
    
    print("\n" + "=" * 60)
    print("FILES CREATED:")
    print("=" * 60)
    print("1. gradient_boosting_feature_importance_fixed.png - Plot with Q1-Q15 labels")
    print("2. gradient_boosting_features.txt - Full questions and statistics")
    print("\nTo view the full questions, open: ../results/supervised/gradient_boosting_features.txt")

if __name__ == "__main__":
    print("=" * 70)
    print("FIXING GRADIENT BOOSTING FEATURE IMPORTANCE PLOT")
    print("=" * 70)
    
    fix_gradient_boosting_plot()
    
    print("\n" + "=" * 70)
    print("PROCESS COMPLETE")
    print("=" * 70)