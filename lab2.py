# lab2.py (ROC Curve Visualization)
import subprocess
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt

# Automatically update dataset
subprocess.run(["python", "updater.py"])

# Load dataset
data = pd.read_csv('student_scores.csv')

X = data[['Subject1', 'Subject2', 'Subject3', 'Subject4', 'Subject5']].values
y = data['Result'].map({'Fail': 0, 'Pass': 1}).values

# Split data (70% train, 20% validation, 10% test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=1/3, stratify=y_temp, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Neural Network model
model = Sequential([
    Input(shape=(5,)),
    Dense(10, activation='relu'),
    Dense(8, activation='relu'),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), verbose=0)

print("\nLab 2: Neural Network trained successfully.")

# Predictions with probabilities for ROC curve
y_pred_proba = model.predict(X_test).ravel()
y_pred = (y_pred_proba >= 0.5).astype(int)

# ROC Curve visualization
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

# Classification report for detailed analysis
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Fail', 'Pass'], zero_division=0))

# Enhanced Interactive Peer Analysis (unchanged)
def student_peer_analysis(student_num):
    if not (1 <= student_num <= 100):
        print("Enter Student number between [1-100].")
        return

    target_student = data.iloc[student_num - 1]
    target_avg = target_student['Average']
    target_result = target_student['Result']

    print(f"\nAnalysis for Student_{student_num}:")
    print(f" - Average Score: {target_avg:.2f}")
    print(f" - Overall Result: {target_result}")

    # Find similar peers
    data['Similarity'] = abs(data['Average'] - target_avg)
    peers = data.drop(student_num - 1).sort_values('Similarity').head(5)

    print("\nTop 5 peers with similar results:")
    print(peers[['Student', 'Average', 'Result']].to_string(index=False))

    # Visualization of subject scores comparison
    subjects = ['Subject1', 'Subject2', 'Subject3', 'Subject4', 'Subject5']
    peer_avg_scores = peers[subjects].mean()
    student_scores = target_student[subjects]

    plt.figure(figsize=(10, 5))
    plt.bar(subjects, student_scores, alpha=0.7, color='skyblue', label=f'Student_{student_num}')
    plt.plot(subjects, peer_avg_scores, color='red', linestyle='--', marker='o', linewidth=2, markersize=8, label='Peers Avg')

    plt.title(f'Subject-wise Score Comparison: Student_{student_num} vs Peers')
    plt.ylabel('Score')
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

# Interactive student peer analysis
student_num = int(input("\nEnter Student number [1-100] for peer analysis: "))
student_peer_analysis(student_num)
