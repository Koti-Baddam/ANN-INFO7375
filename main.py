# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Seed for reproducibility
np.random.seed(42)

# STEP 1: Generate realistic dataset (100 students, 5 subjects)
scores = np.random.uniform(20, 100, (100, 5))
df = pd.DataFrame(scores, columns=['Subject1', 'Subject2', 'Subject3', 'Subject4', 'Subject5'])

# Assign student labels explicitly
df.insert(0, 'Student', [f'Student_{i + 1}' for i in range(df.shape[0])])

# Calculate average score per student and classify Pass or Fail (threshold = 50)
df['Average'] = df.iloc[:, 1:6].mean(axis=1)
df['Result'] = df['Average'].apply(lambda x: 'Pass' if x >= 50 else 'Fail')

# Save dataset to CSV
df.to_csv('student_scores.csv', index=False)
print(" Dataset saved to student_scores.csv")

# STEP 2: Preprocessing
# Load dataset from CSV
# Load dataset from CSV
data = pd.read_csv('student_scores.csv')

# Handle missing values for numeric columns only
numeric_cols = ['Subject1', 'Subject2', 'Subject3', 'Subject4', 'Subject5', 'Average']
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())


# Extract features (subjects) and labels (Result)
X = data[['Subject1', 'Subject2', 'Subject3', 'Subject4', 'Subject5']].values
y = data['Result'].map({'Fail': 0, 'Pass': 1}).values

# Split into Train (70%), Validation (20%), Test (10%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(1 / 3), stratify=y_temp, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# STEP 3: Neural Network Model (Regularization and Dropout included)
# Convert labels to categorical
y_train_cat = to_categorical(y_train)
y_val_cat = to_categorical(y_val)
y_test_cat = to_categorical(y_test)

# Define the neural network model
model = Sequential([
    Input(shape=(5,)),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train_cat, epochs=50, validation_data=(X_val, y_val_cat), verbose=0)
print("Neural network training complete")

# Evaluate accuracy
train_acc = model.evaluate(X_train, y_train_cat, verbose=0)[1]
val_acc = model.evaluate(X_val, y_val_cat, verbose=0)[1]
test_acc = model.evaluate(X_test, y_test_cat, verbose=0)[1]

# Plot accuracy results
plt.bar(['Train', 'Validation', 'Test'], [train_acc, val_acc, test_acc], color=['blue', 'orange', 'green'])
plt.ylabel('Accuracy')
plt.title('Accuracy (Pass/Fail) with Regularization and Dropout')
plt.ylim([0, 1])
plt.show()

print(f'Training Accuracy: {train_acc:.2f}')
print(f'Validation Accuracy: {val_acc:.2f}')
print(f'Testing Accuracy: {test_acc:.2f}')


# STEP 4: Interactive Query Function
def check_student_result(student_num, subject_num):
    if not (1 <= student_num <= 100) or not (1 <= subject_num <= 5):
        print("Invalid input! Enter Student number [1-100] and Subject number [1-5].")
        return
    student_data = data.iloc[student_num - 1]
    subject_score = student_data[f'Subject{subject_num}']
    result = 'Pass' if subject_score >= 50 else 'Fail'

    # Display detailed student information
    print(f"\n Student_{student_num} Details:")
    for i in range(1, 6):
        print(f"  - Subject{i} Score: {student_data[f'Subject{i}']:.2f}")
    print(f"  - Average Score: {student_data['Average']:.2f}")
    print(f"  - Overall Result: {student_data['Result']}")
    print(f"\nResult for Subject {subject_num}: {subject_score:.2f} → {result}")


# Interactive input for checking individual student performance
student_num = int(input("Enter Student number [1-100]: "))
subject_num = int(input("Enter Subject number [1-5]: "))
check_student_result(student_num, subject_num)
