import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Starting to read data from train.csv...")
# 1. Read data
df = pd.read_csv('train.csv')

# 2. Preprocess to create columns matching app.py
# Extract 'Title' from 'Name'
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
# Group Titles
title_mapping = {"Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master"}
df['Title'] = df['Title'].map(lambda x: title_mapping.get(x, "Other"))

# Calculate 'Family_category' from 'SibSp' and 'Parch'
df['Family_size'] = df['SibSp'] + df['Parch'] + 1
def get_family_cat(size):
    if size == 1:
        return "Single"
    elif size <= 4:
        return "Small"
    elif size <= 6:
        return "Medium"
    else:
        return "Large"
df['Family_category'] = df['Family_size'].apply(get_family_cat)

# Select required features
features = ['Age', 'Fare', 'Pclass', 'Sex', 'Embarked', 'Title', 'Family_category']
X = df[features]
y = df['Survived']

# 3. Create Pipeline
numeric_features = ['Age', 'Fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'Family_category']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

print("Training model...")
# 4. Train model
pipeline.fit(X, y)

print("Training complete. Accuracy on train set:", round(pipeline.score(X, y) * 100, 2), "%")

# 5. Save model using joblib
model_filename = 'titanic_model_pipeline.pkl'
joblib.dump(pipeline, model_filename)
print(f"Model saved successfully to: {model_filename}")
