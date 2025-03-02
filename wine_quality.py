import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=";")

# Exploratory Data Analysis (EDA)
print("Data shape:", df.shape)
print("Data info:", df.info())
print("Data description:", df.describe())

# Check for missing values
print("Missing values:", df.isnull().sum())

# Visualize the distribution of the target variable
sns.countplot(x="quality", data=df)
plt.show()

# Visualize the correlation matrix
corr = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.show()

# Data Preprocessing
# Separate features and target variable
X = df.drop("quality", axis=1)
y = df["quality"]

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle imbalanced data using SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Model Training
# Choose a machine learning algorithm
model = GradientBoostingClassifier(random_state=42)

# Hyperparameter Tuning
param_distributions = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.5, 0.7, 1.0],
    "min_samples_split": [2, 4, 6],
    "min_samples_leaf": [1, 2, 3]
}

randomized_search = RandomizedSearchCV(
    model, param_distributions, n_iter=10, cv=3, scoring="accuracy", random_state=42
)
randomized_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best hyperparameters:", randomized_search.best_params_)

# Evaluate the model
best_model = randomized_search.best_estimator_
y_pred = best_model.predict(X_test)

# Print the evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:", classification_report(y_test, y_pred, zero_division=0))

# Feature Importance
feature_importance = best_model.feature_importances_
feature_names = df.drop("quality", axis=1).columns
plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importance, y=feature_names)
plt.title("Feature Importance")
plt.show()
