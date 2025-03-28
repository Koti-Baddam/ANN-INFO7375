# updater.py
import numpy as np
import pandas as pd

np.random.seed()  # Random seed for new data every iteration

scores = np.random.uniform(20, 100, (100, 5))
df = pd.DataFrame(scores, columns=['Subject1', 'Subject2', 'Subject3', 'Subject4', 'Subject5'])
df.insert(0, 'Student', [f'Student_{i + 1}' for i in range(100)])

df['Average'] = df.iloc[:, 1:6].mean(axis=1)
df['Result'] = df['Average'].apply(lambda x: 'Pass' if x >= 50 else 'Fail')

df.to_csv('student_scores.csv', index=False)
print("Dataset updated successfully!")
