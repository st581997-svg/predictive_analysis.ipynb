import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


# LOAD DATA 
df = pd.read_csv("Hospital_Data.csv")
print(df.info())
print(df.isnull().sum())

sns.heatmap(df.isnull())

# CALCULATE LENGTH OF STAY
df['Admission Date'] = pd.to_datetime(df['Admission Date'], dayfirst=True)
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'], dayfirst=True)
df['Stay_Duration'] = (df['Discharge Date'] - df['Admission Date']).dt.days

# Handle cases where stay is 0 days (same day discharge)
df['Stay_Duration'] = df['Stay_Duration'].apply(lambda x: 1 if x == 0 else x)

# 3. FEATURE SELECTION & ENCODING
# Convert text columns (Department) into numbers so the model can read them
le = LabelEncoder()
df['Dept_Encoded'] = le.fit_transform(df['Department'])

# Select features for the model
features = ['Doctors Count', 'Patients Count', 'Stay_Duration', 'Dept_Encoded']
X = df[features]
y = df['Medical Expenses']

# 4. VISUALIZATION (Correlation Heatmap)
plt.figure(figsize=(8, 6))
sns.heatmap(df[features + ['Medical Expenses']].corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation with Medical Expenses")
plt.savefig('correlation_heatmap.png') # Save for your report

# 5. MODEL TRAINING
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# 6. EVALUATION
predictions = model.predict(X_test)
print(f"R-squared Score: {r2_score(y_test, predictions):.4f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, predictions):.2f}")

# Show Actual vs Predicted
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
print("\nFirst 5 Predictions vs Actual:")
print(comparison.head())