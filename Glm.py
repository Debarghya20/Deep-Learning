import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

# --- Step 1: Load training data ---
print("Loading training data...")
train_data = pd.read_csv('train_data.csv')
X = train_data.drop(columns=['age'])
y = train_data['age']

# --- Step 2: Define preprocessing ---
categorical_features = [
    'geological_period',
    'surrounding_rock_type',
    'stratigraphic_position',
    'paleomagnetic_data',
    'inclusion_of_other_fossils'
]
numerical_features = [col for col in X.columns if col not in categorical_features]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
])

# --- Step 3: Split training data into train/validation ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 4: Preprocess train and val sets ---
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_val_preprocessed = preprocessor.transform(X_val)

# --- Step 5: Try different polynomial degrees ---
best_degree = 2
best_val_r2 = -np.inf
degrees_to_try = [2, 1]  # Try degree 2 first, if overfitting, fallback to degree 1

for degree in degrees_to_try:
    print(f"\n🔍 Trying polynomial degree {degree}...")
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_preprocessed)
    X_val_poly = poly.transform(X_val_preprocessed)

    # Add constant term
    X_train_poly = sm.add_constant(X_train_poly)
    X_val_poly = sm.add_constant(X_val_poly)

    # Train model
    glm_model = sm.GLM(y_train, X_train_poly, family=sm.families.Gaussian())
    glm_results = glm_model.fit()

    # Evaluate
    y_train_pred = glm_results.predict(X_train_poly)
    y_val_pred = glm_results.predict(X_val_poly)

    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    print(f"Train R²: {train_r2:.4f}")
    print(f"Validation R²: {val_r2:.4f}")

    if train_r2 - val_r2 > 0.05:
        print("Overfitting detected! Trying simpler model...")
        continue
    else:
        best_degree = degree
        best_val_r2 = val_r2
        print("Good generalization!")
        break

# --- Step 6: Retrain final model on ALL data ---
print(f"\n🎯 Retraining final model with degree {best_degree} on all training data...")

X_all_preprocessed = preprocessor.fit_transform(X)
poly_final = PolynomialFeatures(degree=best_degree, include_bias=False)
X_all_poly = poly_final.fit_transform(X_all_preprocessed)
X_all_poly = sm.add_constant(X_all_poly)

glm_final_model = sm.GLM(y, X_all_poly, family=sm.families.Gaussian())
glm_final_results = glm_final_model.fit()

print("\n✅ Final model trained!")

# --- Step 7: Load test data ---
print("\nLoading test data...")
test_data = pd.read_csv('test_data_nolabels.csv')

if 'age' in test_data.columns:
    test_data = test_data.drop(columns=['age'])

X_test_preprocessed = preprocessor.transform(test_data)
X_test_poly = poly_final.transform(X_test_preprocessed)
X_test_poly = sm.add_constant(X_test_poly)

# --- Step 8: Predict test data ---
print("Predicting fossil ages...")
test_predictions = glm_final_results.predict(X_test_poly)

# --- Step 9: Save predictions ---
submission = pd.DataFrame({
    'age': test_predictions.round(2)
})
submission.to_csv('test_predictions.csv', index=False)

print("\n🎯 Predictions saved successfully to 'test_predictions.csv'!")

# --- Optional: Plot predicted ages ---
plt.figure(figsize=(8,6))
plt.hist(test_predictions, bins=30, color='skyblue', edgecolor='black')
plt.title("Distribution of Predicted Fossil Ages")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()