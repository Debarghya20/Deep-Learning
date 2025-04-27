import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score  # <-- added this
import statsmodels.api as sm

# Step 1: Load data
print("Loading data...")
data = pd.read_csv('train_data.csv')
print(f"Data shape: {data.shape}")
print(data.head())

# Step 2: Separate features and target
X = data.drop(columns=['age'])
y = data['age']
print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Step 3: Feature engineering
categorical_features = [
    'geological_period',
    'surrounding_rock_type',
    'stratigraphic_position',
    'paleomagnetic_data',
    'inclusion_of_other_fossils'
]
numerical_features = [col for col in X.columns if col not in categorical_features]
print("\nCategorical features:", categorical_features)
print("Numerical features:", numerical_features)

# Preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
])

# Step 4: Split data
print("\nSplitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Step 5: Fit preprocessor on training data and transform both
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Step 6: Polynomial feature expansion
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_preprocessed)
X_test_poly = poly.transform(X_test_preprocessed)

# Step 7: Add constant term for intercept
X_train_poly = sm.add_constant(X_train_poly)
X_test_poly = sm.add_constant(X_test_poly)

# Step 8: Train GLM model
print("\nTraining GLM model...")
glm_model = sm.GLM(y_train, X_train_poly, family=sm.families.Gaussian())
glm_results = glm_model.fit()

# Step 9: Print GLM summary (first 10 coefficients only)
print("\nGLM Coefficients (first 10):")
print(glm_results.params.head(10))

# Step 10: Evaluate model
y_train_pred = glm_results.predict(X_train_poly)
y_test_pred = glm_results.predict(X_test_poly)

train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\n--- Training Performance ---")
print(f"Training MSE: {train_mse:.2f}")
print(f"Training R² Score: {train_r2:.4f}")

print("\n--- Test Performance ---")
print(f"Test MSE: {test_mse:.2f}")
print(f"Test R² Score: {test_r2:.4f}")

# Step 11: Overfitting warning
if train_r2 - test_r2 > 0.05:
    print("\n Warning: Possible overfitting detected (large gap between training and test performance).")
else:
    print("\n Model generalizes well! No major overfitting detected.")

# Step 12: Sample predictions
print("\nSample predictions:")
for i in range(5):
    print(f"True age: {y_test.iloc[i]:.1f}, Predicted age: {y_test_pred[i]:.1f}")

# Step 13: Plotting Predicted vs True Age (Test Set)
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_test_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("True Age")
plt.ylabel("Predicted Age")
plt.title("True Age vs Predicted Age (Test Set) - GLM")
plt.grid(True)
plt.show()
