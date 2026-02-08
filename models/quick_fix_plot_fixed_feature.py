import matplotlib.pyplot as plt
import numpy as np
import joblib

# Load your model
model = joblib.load('../models/saved_models/supervised/random_forest.pkl')

# Load feature names
with open('../models/saved_models/feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f.readlines()]

# Get importances
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Get top 15 features (most to least important)
top_n = 15
top_indices = indices[:top_n]
top_importances = importances[top_indices]
top_names = [feature_names[i] for i in top_indices]

# Apply text replacements
processed_names = []
for name in top_names:
    name = name.replace("how 4", "how likely")
    name = name.replace("you personally 2 or 4", "you personally agree or disagree")
    processed_names.append(name)

top_names = processed_names

# Create just the bar plot with Q labels
plt.figure(figsize=(10, 8))

# Plot horizontal bars
y_pos = np.arange(top_n)
bars = plt.barh(y_pos, top_importances)

# Use Q1, Q2, ... labels on y-axis
plt.yticks(y_pos, [f'Q{i+1}' for i in range(top_n)], fontsize=12)
plt.xlabel('Importance', fontsize=12)
plt.title(f'Top {top_n} Feature Importances', fontsize=14, pad=20)

# Add grid for readability
plt.grid(axis='x', alpha=0.3, linestyle='--')

# Invert y-axis so Q1 (most important) is at top
plt.gca().invert_yaxis()

# Add value labels on bars
for i, (bar, imp) in enumerate(zip(bars, top_importances)):
    width = bar.get_width()
    plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
             f'{imp:.4f}', ha='left', va='center', fontsize=9)

# Adjust layout to prevent cutting
plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.08)

# Save the plot
plt.savefig('../results/supervised/feature_importance_plot.png', 
           bbox_inches='tight', dpi=300)
plt.close()

print("Plot saved to: ../results/supervised/feature_importance_plot.png")

# Get the FULL questions from the dissertation (Appendix D)
full_questions = {
    # Q1: Do you think a psychiatrist/psychologist will be helpful?
    "Do you think a psychiatrist/psychologist will be helpful?": "Do you think a psychiatrist/psychologist will be helpful?",
    
    # Q2: What is your area of study? If you are not a student, please select "not a student".
    "What is your area of study? If you are not a student, please select \"not a student\".": "What is your area of study? If you are not a student, please select \"not a student\".",
    
    # Q3: The next few questions contain statements about John's problem. Please indicate how strongly YOU PERSONALLY agree or disagree with each statement. - It is best to avoid John so that you don't develop this problem yourself.
    "It is best to avoid John so that you don't develop this problem yourself.": "It is best to avoid John so that you don't develop this problem yourself.",
    
    # Q4: John's problem makes him unpredictable.
    "John's problem makes him unpredictable.": "John's problem makes him unpredictable.",
    
    # Q5: How many years of tertiary study have you had?
    "How many years of tertiary study have you had?": "How many years of tertiary study have you had?",
    
    # Q6: How likely is it that you would take the following actions with John? - Ask whether they have other supportive people they can rely on.
    "Ask whether they have other supportive people they can rely on.": "How likely is it that you would take the following actions with John? - Ask whether they have other supportive people they can rely on.",
    
    # Q7: What is your age? - Age (years)
    "What is your age? - Age (years)": "What is your age? (Age in years)",
    
    # Q8: What do you think John is experiencing? - Selected Choice
    "What do you think John is experiencing?": "What do you think John is experiencing?",
    
    # Q9: How likely is it that you would take the following actions with John? - Let them know you are listening to what they are saying by restating and summarising what they have said.
    "Let them know you are listening to what they are saying by restating and summarising what they have said.": "How likely is it that you would take the following actions with John? - Let them know you are listening to what they are saying by restating and summarising what they have said.",
    
    # Q10: The following questions ask how you would feel about spending time with John. Please indicate how likely you would perform each statement. - To go to John's house?
    "To go to John's house?": "The following questions ask how you would feel about spending time with John. Please indicate how likely you would perform each statement. - To go to John's house?",
    
    # Q11: How likely is it that you would take the following actions with John? - Convey a message of hope by telling them help is available and things can get better.
    "Convey a message of hope by telling them help is available and things can get better.": "How likely is it that you would take the following actions with John? - Convey a message of hope by telling them help is available and things can get better.",
    
    # Q12: The next few questions contain statements about John's problem. Please indicate how strongly YOU PERSONALLY agree or disagree with each statement. - John could snap out of it if he wanted.
    "John could snap out of it if he wanted.": "The next few questions contain statements about John's problem. Please indicate how strongly YOU PERSONALLY agree or disagree with each statement. - John could snap out of it if he wanted.",
    
    # Q13: How likely is it that you would take the following actions with John? - Find out if there are specific reasons why they do not want to seek professional help.
    "Find out if there are specific reasons why they do not want to seek professional help.": "How likely is it that you would take the following actions with John? - Find out if there are specific reasons why they do not want to seek professional help.",
    
    # Q14: The next few questions contain statements about John's problem. Please indicate how strongly YOU PERSONALLY agree or disagree with each statement. - John's problem is a sign of personal weakness.
    "John's problem is a sign of personal weakness.": "The next few questions contain statements about John's problem. Please indicate how strongly YOU PERSONALLY agree or disagree with each statement. - John's problem is a sign of personal weakness.",
    
    # Q15: How likely is it that you would take the following actions with John? - Try to convince them that their beliefs and perceptions are false.
    "Try to convince them that their beliefs and perceptions are false.": "How likely is it that you would take the following actions with John? - Try to convince them that their beliefs and perceptions are false."
}

# Create a new text file with the FULL questions from the dissertation
with open('../results/supervised/feature_descriptions_full.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("TOP 15 FEATURE IMPORTANCES - FULL QUESTION TEXT\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("Plot shows features as Q1 (most important) to Q15 (least important)\n")
    f.write("Text replacements applied:\n")
    f.write("  - 'how 4' replaced with 'how likely'\n")
    f.write("  - 'you personally 2 or 4' replaced with 'you personally agree or disagree'\n")
    f.write("\nNote: Full questions extracted from dissertation Appendix D\n")
    f.write("=" * 80 + "\n\n")
    
    # Try to match our features with the full questions from dissertation
    for i, (label, feature_text, imp) in enumerate(zip([f'Q{i+1}' for i in range(top_n)], 
                                                       top_names, 
                                                       top_importances)):
        f.write(f"{label} - Importance: {imp:.4f}\n")
        f.write("-" * 40 + "\n")
        
        # Try to find the matching full question
        matched = False
        for key_phrase in full_questions.keys():
            if key_phrase.lower() in feature_text.lower():
                f.write(f"{full_questions[key_phrase]}\n\n")
                matched = True
                break
        
        if not matched:
            # If no match found, use the processed feature text
            f.write(f"{feature_text}\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("QUESTIONS FROM DISSERTATION APPENDIX D\n")
    f.write("=" * 80 + "\n\n")
    f.write("Full list of questions from the survey instrument:\n\n")
    
    # List all questions from Appendix D for reference
    appendix_d_questions = [
        "1. Do you think a psychiatrist/psychologist will be helpful?",
        "2. What is your area of study? If you are not a student, please select \"not a student\".",
        "3. How many years of tertiary study have you had?",
        "4. What is your age?",
        "5. What do you think John is experiencing?",
        "6. John could snap out of it if he wanted.",
        "7. John's problem is a sign of personal weakness.",
        "8. John's problem is not a real medical illness.",
        "9. John is dangerous. By 'dangerous' we mean 'dangerous to others'.",
        "10. It is best to avoid John so that you don't develop this problem yourself.",
        "11. John's problem makes him unpredictable.",
        "12. To go out with John on the weekend?",
        "13. To work on a project with John?",
        "14. To invite John around to your house?",
        "15. To go to John's house?",
        "16. To develop a close friendship with John?",
        "17. Ask if they have been having thoughts of harming themselves or others.",
        "18. Let them know you are listening to what they are saying by restating and summarising what they have said.",
        "19. Convey a message of hope by telling them help is available and things can get better.",
        "20. Discuss their options for seeking professional help.",
        "21. Ask whether they have other supportive people they can rely on.",
        "22. Ask if they have been thinking about suicide.",
        "23. Ask if they have a plan for suicide -- for example, how, when and where they intend to die.",
        "24. Encourage them to get appropriate professional help as soon as possible -- for example, see a mental health professional or someone at a mental health service.",
        "25. Acknowledge they might be frightened by what they are experiencing.",
        "26. Try to convince them that their beliefs and perceptions are false.",
        "27. Listen to them talk about their experiences even though you know they are not based in reality.",
        "28. Find out if there are specific reasons why they do not want to seek professional help."
    ]
    
    for question in appendix_d_questions:
        f.write(f"{question}\n")

print("\nFiles created:")
print("1. feature_importance_plot.png - Visual plot with Q1-Q15 labels")
print("2. feature_descriptions_full.txt - Full questions from dissertation Appendix D")

# Print the top 3 features for verification
print("\n" + "=" * 60)
print("TOP 3 FEATURES VERIFICATION:")
print("=" * 60)
for i in range(3):
    print(f"\nQ{i+1} (Importance: {top_importances[i]:.4f}):")
    print(f"  Feature text: {top_names[i][:100]}...")
    
    # Try to find matching full question
    for key_phrase in full_questions.keys():
        if key_phrase.lower() in top_names[i].lower():
            print(f"  Full question from dissertation: {full_questions[key_phrase]}")
            break