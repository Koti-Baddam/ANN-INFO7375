# Necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt

# Seed for reproducibility
np.random.seed(42)

# STEP 1: Generate dataset (100 students, 5 subjects)
scores = np.random.uniform(20, 100, (100, 5))
df = pd.DataFrame(scores, columns=['Subject1', 'Subject2', 'Subject3', 'Subject4', 'Subject5'])
df.insert(0, 'Student', [f'Student_{i + 1}' for i in range(df.shape[0])])

# Calculate averages and Pass/Fail labels (threshold: 50)
df['Average'] = df.iloc[:, 1:6].mean(axis=1)
df['Result'] = df['Average'].apply(lambda x: 1 if x >= 50 else 0)  # binary (1=Pass, 0=Fail)

# Save dataset
df.to_csv('student_scores.csv', index=False)
print("Dataset saved to student_scores.csv")

# STEP 2: Data preprocessing
data = pd.read_csv('student_scores.csv')

X = data[['Subject1', 'Subject2', 'Subject3', 'Subject4', 'Subject5']].values
y = data['Result'].values  # binary classification directly as 0 or 1

# Split: 70% Train, 20% Validation, 10% Test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, stratify=y_temp, random_state=42)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# STEP 3: Multilayer neural network (10→8→8→4→1)
model = Sequential([
    Input(shape=(5,)),
    Dense(10, activation='relu'),
    Dense(8, activation='relu'),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')  # binary output
])

# Compile model (binary classification)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), verbose=0)
print("Neural network training complete")

# STEP 4: Visualizing neural network logic using a confusion matrix
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fail', 'Pass'])
disp.plot(cmap='Blues')
plt.title('Neural Network Predictions (Pass/Fail)')
plt.show()

# STEP 5: Interactive Student Query
def check_student_result(student_num, subject_num):
    if not (1 <= student_num <= 100) or not (1 <= subject_num <= 5):
        print("Invalid input! Enter Student number [1-100] and Subject number [1-5].")
        return
    student_data = data.iloc[student_num - 1]
    subject_score = student_data[f'Subject{subject_num}']
    result = 'Pass' if subject_score >= 50 else 'Fail'

    print(f"\nStudent_{student_num} Details:")
    for i in range(1, 6):
        print(f"  Subject{i} Score: {student_data[f'Subject{i}']:.2f}")
    print(f"  Average Score: {student_data['Average']:.2f}")
    print(f"  Overall Result: {'Pass' if student_data['Result'] == 1 else 'Fail'}")
    print(f"\nSubject {subject_num}: {subject_score:.2f} → {result}")

student_num = int(input("Enter Student number [1-100]: "))
subject_num = int(input("Enter Subject number [1-5]: "))
check_student_result(student_num, subject_num)
