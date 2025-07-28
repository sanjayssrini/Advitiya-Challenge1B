# train_ensemble.py
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

# Load and prepare data
print("Loading dataset...")
df = pd.read_csv("cleaned_dataset_final.csv")

# Enhanced feature engineering
def create_advanced_features(df):
    """Create advanced features for better classification"""
    # Text-based features
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    df['avg_word_length'] = df['text_length'] / df['word_count']
    df['starts_with_number'] = df['text'].str.match(r'^\d').astype(int)
    df['has_punctuation'] = df['text'].str.contains(r'[.,!?]').astype(int)
    
    # Font features
    if 'font_size' in df.columns:
        mean_font = df['font_size'].mean()
        std_font = df['font_size'].std()
        df['font_size_zscore'] = (df['font_size'] - mean_font) / std_font
    
    # Position features
    if 'y_center' in df.columns:
        df['normalized_y_pos'] = df['y_center'] / df['y_center'].max()
        df['is_top_third'] = (df['normalized_y_pos'] <= 0.33).astype(int)
    
    return df

# Apply feature engineering
df = create_advanced_features(df)

# Prepare features and labels
print("Preparing features...")
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])
joblib.dump(label_encoder, "label_encoder.pkl")

# Select features
feature_columns = [col for col in df.columns if col not in ["label", "label_encoded", "text"]]
numeric_columns = []
categorical_columns = []

for col in feature_columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        numeric_columns.append(col)
    else:
        categorical_columns.append(col)

# Encode categorical features
for col in categorical_columns:
    df[col] = df[col].astype(str)
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])

feature_columns = numeric_columns + categorical_columns
joblib.dump(feature_columns, "feature_columns.pkl")

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(df[feature_columns])
y = df["label_encoded"]
joblib.dump(scaler, "feature_scaler.pkl")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define enhanced classifiers with optimized parameters
classifiers = [
    ('lr', LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight='balanced'
    )),
    ('rf', RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        class_weight='balanced'
    )),
    ('gb', GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5
    )),
    ('knn', KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',
        metric='cosine'
    ))
]

# Create and train ensemble
print("Training ensemble model...")
ensemble = VotingClassifier(
    estimators=classifiers,
    voting='soft',
    weights=[1, 2, 2, 1]  # Give more weight to RF and GB
)

# Train the model
ensemble.fit(X_train, y_train)

# Evaluate
print("\nModel Evaluation:")
y_pred = ensemble.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save the model
joblib.dump(ensemble, "ensemble_model.pkl")
print("✓ Enhanced model trained and saved")
