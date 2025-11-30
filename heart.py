# Load necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('heart.csv')

# Preview the data
print(df.head())

# Check dataset shape
print(df.shape)

# Get info on columns and types
print(df.info())

# Check for null values
print(df.isnull().sum())

# Summary statistics
print(df.describe())

# Show all column names and datatypes
print(df.dtypes)

# Check unique values for potential categorical columns
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
for col in categorical_cols:
    print(f"{col}: {df[col].unique()}")

# One-hot encode categorical features
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print(df_encoded.head())

# Confirm there are no missing values
print(df_encoded.isnull().sum())


# Identify numerical columns (excluding target)
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

scaler = StandardScaler()
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])


# Separate features and target
X = df_encoded.drop(columns='HeartDisease')
y = df_encoded['HeartDisease']

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42, stratify=y)

# Count of heart disease presence (1) vs absence (0)
sns.countplot(x='HeartDisease', data=df)
plt.title('Target Variable Distribution: Heart Disease')
plt.xlabel('Heart Disease (1 = Yes, 0 = No)')
plt.ylabel('Count')
print(plt.show())

sns.histplot(df['Age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
print(plt.show())

sns.boxplot(x='HeartDisease', y='Cholesterol', data=df)
plt.title("Cholesterol vs Heart Disease")
plt.xlabel("Heart Disease")
plt.ylabel("Cholesterol")
print(plt.show())

sns.boxplot(x='HeartDisease', y='MaxHR', data=df)
plt.title("Max Heart Rate vs Heart Disease")
plt.xlabel("Heart Disease")
plt.ylabel("MaxHR")
print(plt.show())

# Pie chart of HeartDisease classes
labels = ['No Disease', 'Heart Disease']
sizes = df['HeartDisease'].value_counts().sort_index()
colors = ['lightblue', 'lightcoral']

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
plt.title('Heart Disease Distribution')
plt.axis('equal')
print(plt.show())

# Plot histogram for each numeric column
numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

plt.figure(figsize=(15, 8))
for i, col in enumerate(numeric_cols):
    plt.subplot(2, 3, i + 1)
    plt.hist(df[col], bins=20, color='skyblue', edgecolor='black')
    plt.title(f'{col} Distribution')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
print(plt.show())

# Bar chart of chest pain type counts by heart disease
cp_counts = df.groupby(['ChestPainType', 'HeartDisease']).size().unstack()

cp_counts.plot(kind='bar', stacked=True, color=['lightgreen', 'salmon'])
plt.title('Chest Pain Type by Heart Disease')
plt.xlabel('Chest Pain Type')
plt.ylabel('Count')
plt.legend(['No Disease', 'Heart Disease'])
plt.xticks(rotation=0)
plt.tight_layout()
print(plt.show())

# MODEL

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Model: {model.__class__.__name__}")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("="*50)

# logistic regression
log_reg = LogisticRegression(max_iter=1000)
print(train_and_evaluate(log_reg, X_train, X_test, y_train, y_test))

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
print(train_and_evaluate(dt, X_train, X_test, y_train, y_test))

# Random forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
print(train_and_evaluate(rf, X_train, X_test, y_train, y_test))

# comparison of models
def get_metrics(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "Model": model.__class__.__name__,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }
# Metrics
# Instantiate models
models = [
    LogisticRegression(max_iter=1000),
    DecisionTreeClassifier(random_state=42),
    RandomForestClassifier(n_estimators=100, random_state=42)
]

# Collect results
results = []
for model in models:
    metrics = get_metrics(model, X_train, X_test, y_train, y_test)
    results.append(metrics)

# Create DataFrame to compare
results_df = pd.DataFrame(results)
print(results_df)

# Plot Accuracy, Precision, Recall, F1 in one figure
results_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1 Score']].plot(
    kind='bar', figsize=(10, 6), colormap='viridis'
)
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.ylim(0, 1.1)
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.grid(axis='y')
plt.tight_layout()
print(plt.show())

