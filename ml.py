import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

print("Diabetes ML Model ")

# Load data
df = pd.read_csv('diabetes_prediction_.csv')

# Clean data
df = df[(df['Diabetic'] == 0) | (df['Diabetic'] == 1)]
df = df.copy()

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['BMI'] = df['BMI'].fillna(df['BMI'].median())
df['Glucose'] = df['Glucose'].fillna(df['Glucose'].median())
df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].median())
df['FamilyHistory'] = df['FamilyHistory'].fillna('No')
df['FamilyHistory'] = df['FamilyHistory'].map({'Yes': 1, 'No': 0})

print(f"Data: {len(df)} patients, Diabetic: {df['Diabetic'].sum()}")

#  PLOT 1 & 2: Basic EDA 
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
diabetes_counts = df['Diabetic'].value_counts()
plt.pie(diabetes_counts.values, labels=['Healthy', 'Diabetic'], autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
plt.title('Diabetes Distribution')

plt.subplot(1, 2, 2)
plt.boxplot([df[df['Diabetic']==0]['Glucose'], df[df['Diabetic']==1]['Glucose']], 
            tick_labels=['Healthy', 'Diabetic'])
plt.ylabel('Glucose Level')
plt.title('Glucose Levels Comparison')

plt.tight_layout()
plt.savefig('basic_eda.png')
plt.show()

#  PLOT 3: Age vs Glucose Scatter 
plt.figure(figsize=(10, 6))
plt.scatter(df[df['Diabetic']==0]['Age'], df[df['Diabetic']==0]['Glucose'], 
            alpha=0.6, label='Healthy', color='blue')
plt.scatter(df[df['Diabetic']==1]['Age'], df[df['Diabetic']==1]['Glucose'], 
            alpha=0.6, label='Diabetic', color='red')
plt.xlabel('Age')
plt.ylabel('Glucose Level')
plt.title('Age vs Glucose Level (Colored by Diabetes Status)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('age_glucose_scatter.png')
plt.show()

# PLOT 4: BMI Distribution by Diabetes 
plt.figure(figsize=(10, 6))
plt.hist([df[df['Diabetic']==0]['BMI'], df[df['Diabetic']==1]['BMI']], 
         label=['Healthy', 'Diabetic'], alpha=0.7, bins=15, color=['blue', 'red'])
plt.xlabel('BMI')
plt.ylabel('Number of Patients')
plt.title('BMI Distribution by Diabetes Status')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bmi_distribution.png')
plt.show()

# Train model
features = ['Age', 'BMI', 'Glucose', 'BloodPressure', 'FamilyHistory']
X = df[features]
y = df['Diabetic']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model Accuracy: {accuracy:.3f}")

#  PLOT 5: Feature Importance
plt.figure(figsize=(8, 4))
importance = abs(model.coef_[0])
plt.bar(features, importance, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'])
plt.title('Feature Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# Save model
with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel Coefficients for Frontend:")
print(f"Intercept: {model.intercept_[0]:.6f}")
for i, feature in enumerate(features):
    print(f"{feature}: {model.coef_[0][i]:.6f}")

print("DONE!")