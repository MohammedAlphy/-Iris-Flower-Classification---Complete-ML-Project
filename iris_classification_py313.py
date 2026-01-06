# iris_classification_py313.py
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

print("Python 3.13 Compatible Iris Classification")
print("=" * 50)

# Load the dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target labels

# Create DataFrame with proper column names
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("\n📊 Dataset Overview:")
print(f"Total samples: {X.shape[0]}")
print(f"Features per sample: {X.shape[1]}")
print(f"Feature names: {iris.feature_names}")
print(f"Target classes: {iris.target_names}")
print(f"\nFirst 5 rows:")
print(df.head())

# Visualize with new Seaborn 0.13.2 features
plt.figure(figsize=(12, 5))

# 1. Scatter plot with new styling
plt.subplot(1, 2, 1)
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7, edgecolors='w', linewidth=0.5)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Sepal Dimensions by Species')
plt.colorbar(scatter, ticks=[0, 1, 2], label='Species')

# 2. Pair plot style visualization
plt.subplot(1, 2, 2)
for i, species in enumerate(iris.target_names):
    species_data = X[y == i]
    plt.scatter(species_data[:, 2], species_data[:, 3], label=species, alpha=0.7)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Petal Dimensions by Species')
plt.legend()

plt.tight_layout()
plt.savefig('iris_scatter.png', dpi=100, bbox_inches='tight')
plt.show()

# New in seaborn 0.13: Better boxplots
plt.figure(figsize=(10, 6))
df_melted = df.melt(id_vars=['species_name'], 
                    value_vars=iris.feature_names,
                    var_name='feature', 
                    value_name='measurement')

# Using the new seaborn features
sns.boxplot(data=df_melted, x='feature', y='measurement', hue='species_name',
            palette='Set2', linewidth=1.5)
plt.xticks(rotation=45)
plt.title('Feature Distribution by Species (Seaborn 0.13.2)')
plt.xlabel('')
plt.ylabel('Measurement (cm)')
plt.legend(title='Species')
plt.tight_layout()
plt.savefig('iris_boxplot.png', dpi=100, bbox_inches='tight')
plt.show()

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y,  # Keep class proportions
    shuffle=True  # Shuffle before splitting
)

print(f"\n🔢 Data Split Summary:")
print(f"Training samples: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Testing samples: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"Training class distribution: {np.bincount(y_train)}")
print(f"Testing class distribution: {np.bincount(y_test)}")

# Create and train Random Forest model
print("\n🤖 Training Random Forest Model...")
model = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=None,        # Let trees grow fully
    min_samples_split=2,   # Minimum samples to split
    min_samples_leaf=1,    # Minimum samples in leaf
    random_state=42,       # Reproducibility
    n_jobs=-1              # Use all CPU cores
)

model.fit(X_train, y_train)
print("✅ Model training complete!")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📈 Model Performance:")
print(f"Accuracy: {accuracy:.2%}")
print(f"Misclassified: {(y_test != y_pred).sum()} out of {len(y_test)}")

print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=iris.target_names,
                          digits=3))

# Feature importance (new in scikit-learn 1.5+)
feature_importance = model.feature_importances_
feature_names = iris.feature_names

print("\n🎯 Feature Importance:")
for name, importance in zip(feature_names, feature_importance):
    print(f"{name:20} {importance:.3f}")

# Visualize feature importance
plt.figure(figsize=(8, 4))
bars = plt.barh(feature_names, feature_importance, color='skyblue')
plt.xlabel('Importance Score')
plt.title('Feature Importance in Iris Classification')
plt.gca().invert_yaxis()

# Add value labels
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
             f'{width:.3f}', ha='left', va='center')

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=100, bbox_inches='tight')
plt.show()

# --------------------------------------------------------------------
# STEP 7: MAKE PREDICTIONS 
# --------------------------------------------------------------------
print("\n🔮 STEP 7: Making Predictions with the Model...")

def predict_new_flower(sepal_length, sepal_width, petal_length, petal_width, description=""):
    """
    Predict the species of a new flower based on its measurements
    """
    # Prepare input data
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Get prediction and probabilities
    prediction_idx = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    predicted_species = iris.target_names[prediction_idx]
    confidence = probabilities[prediction_idx]
    
    # Display results
    if description:
        print(f"\n{description}")
    print(f"📏 Input Measurements:")
    print(f"   Sepal: {sepal_length} cm long, {sepal_width} cm wide")
    print(f"   Petal: {petal_length} cm long, {petal_width} cm wide")
    print(f"🎯 Prediction: {predicted_species}")
    print(f"   Confidence: {confidence:.2%}")
    
    print(f"📊 All Probabilities:")
    for i, species in enumerate(iris.target_names):
        prob = probabilities[i]
        bar = '█' * int(prob * 20)  # Visual bar
        print(f"   {species:12} {prob:6.2%} {bar}")
    
    print("-" * 50)
    return predicted_species, confidence

# Test with example flowers
print("\n" + "="*60)
print("🧪 TESTING WITH BETTER EXAMPLE FLOWERS")
print("="*60)

# Based on REAL Iris data ranges:
# Setosa: Small petals (petal_length < 2, petal_width < 0.6)
# Versicolor: Medium petals (3 < petal_length < 5, 1 < petal_width < 1.8)  
# Virginica: Large petals (petal_length > 5, petal_width > 1.8)

# Example 1: CLEAR Setosa
predict_new_flower(
    5.0, 3.4, 1.5, 0.2,
    "🌼 Example 1: CLEAR Setosa (small petals)"
)

# Example 2: CLEAR Versicolor  
predict_new_flower(
    6.0, 2.7, 4.0, 1.3,
    "🌸 Example 2: CLEAR Versicolor (medium petals)"
)

# Example 3: CLEAR Virginica (FIXED!)
predict_new_flower(
    6.7, 3.0, 5.8, 2.1,  # Petal length > 5, width > 1.8 = Virginica
    "🌺 Example 3: CLEAR Virginica (large petals)"
)

# Example 4: Borderline Versicolor/Virginica
predict_new_flower(
    6.2, 2.9, 4.5, 1.5,  # Petal length borderline
    "❓ Example 4: Borderline Versicolor/Virginica"
)

# Example 5: Extreme Virginica
predict_new_flower(
    7.7, 3.8, 6.7, 2.2,  # Very large flower
    "🏆 Example 5: Extreme Virginica (very large)"
)

# Example 6: What if we try a confusing one?
predict_new_flower(
    5.8, 2.7, 5.1, 1.9,  # Petal length borderline, width Virginica
    "🤔 Example 6: Confusing case"
)

# Confusion matrix visualization
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=iris.target_names)

fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')
plt.show()

# Save model using joblib
import joblib
joblib.dump(model, 'iris_classifier_v2.pkl')
print("\n💾 Model saved as 'iris_classifier_v2.pkl'")

# Load and test saved model
print("\n🔍 Testing saved model...")
loaded_model = joblib.load('iris_classifier_v2.pkl')
test_pred = loaded_model.predict(X_test[:1])
print(f"Test prediction from loaded model: {iris.target_names[test_pred[0]]}")

print("\n" + "="*50)
print("✅ All done! Here's what we accomplished:")
print("1. Loaded and explored Iris dataset")
print("2. Visualized data patterns")
print("3. Trained a Random Forest classifier")
print("4. Evaluated model performance")
print("5. Made predictions with confidence scores")
print("6. Saved/Loaded model for future use")

print("="*50)
