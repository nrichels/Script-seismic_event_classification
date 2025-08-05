import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib
import warnings

# Load data
df = pd.read_csv("./Grenzgletscher_fk/features_labeled.csv", sep=";")

# Split features and labels
label_column = "class_name"

# Drop the event_time column - we don't want time as a feature
X = df.drop(columns=[label_column, "event_time"])
y = df[label_column]

# Analyze class distribution
print(f"\nClass distribution:")
class_counts = pd.Series(y).value_counts()
print(class_counts)
print(f"\nClass percentages:")
class_percentages = (class_counts / len(y) * 100).round(2)
print(class_percentages)

# Identify classes with very few samples (less than 2% of data or less than 10 samples)
min_samples_threshold = max(10, len(y) * 0.02)
rare_classes = class_counts[class_counts < min_samples_threshold].index.tolist()
if rare_classes:
    print(f"\nWarning: Classes with very few samples (< {min_samples_threshold:.0f}): {rare_classes}")
    print("Consider combining these with similar classes or collecting more data.")

# Encode labels if they're words
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

print(f"\nClasses found: {class_names}")

# Calculate class weights to handle imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weight_dict = dict(zip(np.unique(y_encoded), class_weights))
print(f"\nClass weights for balancing:")
for i, weight in class_weight_dict.items():
    print(f"  {class_names[i]}: {weight:.3f}")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Check class distribution in train/test sets
print(f"\nTraining set class distribution:")
train_class_counts = pd.Series(y_train).value_counts()
for i, count in train_class_counts.items():
    print(f"  {class_names[i]}: {count}")

print(f"\nTest set class distribution:")
test_class_counts = pd.Series(y_test).value_counts()
for i, count in test_class_counts.items():
    print(f"  {class_names[i]}: {count}")

# Train the Random Forest classifier with class balancing
clf = RandomForestClassifier(
    n_estimators=300,  
    max_depth=25,      
    min_samples_split=5,  
    min_samples_leaf=2,   
    class_weight='balanced',  # Handle class imbalance
    oob_score=True,
    random_state=42,
    bootstrap=True,
    n_jobs=-1  # Use all available cores
)

print("\nTraining Random Forest model with balanced class weights...")
clf.fit(X_train, y_train)

# Cross-validation to get more robust performance estimate
cv_scores = cross_val_score(clf, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='accuracy')
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"OOB Score: {clf.oob_score_:.4f}")

# Classification report with zero_division parameter to suppress warnings
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, 
    target_names=class_names, 
    zero_division=0  
))

# Detailed per-class analysis
print("\nDetailed per-class analysis:")
for i, class_name in enumerate(class_names):
    true_count = np.sum(y_test == i)
    pred_count = np.sum(y_pred == i)
    correct_count = np.sum((y_test == i) & (y_pred == i))
    
    if true_count > 0:
        recall = correct_count / true_count
    else:
        recall = 0
        
    if pred_count > 0:
        precision = correct_count / pred_count
    else:
        precision = 0
        
    print(f"  {class_name}:")
    print(f"    True samples: {true_count}, Predicted: {pred_count}, Correct: {correct_count}")
    print(f"    Precision: {precision:.3f}, Recall: {recall:.3f}")

# Confusion Matrix in percentages
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
# Replace NaN values (from division by zero) with 0
cm_percent = np.nan_to_num(cm_percent, nan=0.0)
sns.heatmap(cm_percent, annot=True, fmt='.1f', xticklabels=class_names, yticklabels=class_names, 
            cmap="Blues", cbar_kws={'label': 'Percentage (%)'})
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (in percent)")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("confusion_matrix_percent_0.2.png", dpi=300, bbox_inches='tight')
plt.show()

# Feature Importance
importances = pd.Series(clf.feature_importances_, index=X.columns)

# Most Important Features
plt.figure(figsize=(12, 8))
importances.nlargest(15).plot(kind='barh', color='teal')
plt.title("Top 15 Most Important Features")
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.show()

# Least Important Features
plt.figure(figsize=(12, 8))
importances.nsmallest(15).plot(kind='barh', color='coral')
plt.title("Top 15 Least Important Features")
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.show()

# Print top features
print(f"\nTop 10 Most Important Features:")
for i, (feature, importance) in enumerate(importances.nlargest(10).items(), 1):
    print(f"{i:2d}. {feature}: {importance:.4f}")

# Print least important features
print(f"\nTop 10 Least Important Features:")
for i, (feature, importance) in enumerate(importances.nsmallest(10).items(), 1):
    print(f"{i:2d}. {feature}: {importance:.4f}")

# Analyze prediction confidence
y_pred_proba = clf.predict_proba(X_test)
max_probabilities = np.max(y_pred_proba, axis=1)

print(f"\nPrediction confidence analysis:")
print(f"Mean prediction confidence: {max_probabilities.mean():.3f}")
print(f"Min prediction confidence: {max_probabilities.min():.3f}")
print(f"Max prediction confidence: {max_probabilities.max():.3f}")

# Find low-confidence predictions
low_confidence_threshold = 0.5
low_confidence_mask = max_probabilities < low_confidence_threshold
if np.any(low_confidence_mask):
    print(f"\nFound {np.sum(low_confidence_mask)} predictions with confidence < {low_confidence_threshold}")
    print("These might be misclassified or difficult cases.")

# Save Model and encoders
print("\nSaving model and encoders...")
joblib.dump(clf, "random_forest_model.pkl")
joblib.dump(le, "label_encoder.pkl")

# Save feature names for later use
feature_names = X.columns.tolist()
joblib.dump(feature_names, "feature_names.pkl")

print("Model training and evaluation complete!")
print(f"Files saved:")
print(f"  - Model: random_forest_model.pkl")
print(f"  - Label encoder: label_encoder.pkl")
print(f"  - Feature names: feature_names.pkl")
print(f"  - Confusion matrix: confusion_matrix_percent.png")

# Additional recommendations
if rare_classes:
    print(f"\nRecommendations:")
    print(f"Consider collecting more data for classes: {rare_classes}")