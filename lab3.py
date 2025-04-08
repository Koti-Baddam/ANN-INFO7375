from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from lab2 import X_train, y_train
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Retrain the model
model.fit(X_train_smote, y_train_smote, epochs=50, validation_data=(X_val, y_val), verbose=0)

# Define a function to create the Keras model
def create_model(hidden_layers=2, neurons=10, activation='relu', learning_rate=0.001):
    model = Sequential()
    model.add(Input(shape=(5,)))
    for _ in range(hidden_layers):
        model.add(Dense(neurons, activation=activation))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Wrap the Keras model for GridSearchCV
model = KerasClassifier(model=create_model, verbose=0)

# Define parameter grid for tuning
param_grid = {
    'model__hidden_layers': [1, 2, 3],
    'model__neurons': [8, 16, 32],
    'model__activation': ['relu', 'tanh'],
    'epochs': [50, 100],
    'batch_size': [16, 32]
}

# Grid search
search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
search.fit(X_train, y_train)

# Print best parameters and score
print("Best parameters:", search.best_params_)
print("Best accuracy:", search.best_score_)
