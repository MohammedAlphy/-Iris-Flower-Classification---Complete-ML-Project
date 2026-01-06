# demo.py - Quick demonstration script
"""
Quick demo of Iris Classification
Run: python demo.py
"""

import numpy as np
import joblib

def quick_demo():
    print("🌺 Iris Classification Demo")
    print("=" * 40)
    
    try:
        # Try to load saved model
        model = joblib.load('iris_classifier_model.pkl')
        print("✅ Model loaded successfully!")
        
        # Test predictions
        examples = [
            [5.1, 3.5, 1.4, 0.2],  # Setosa
            [6.0, 2.7, 4.2, 1.3],  # Versicolor
            [6.7, 3.0, 5.8, 2.1],  # Virginica
        ]
        
        species = ['setosa', 'versicolor', 'virginica']
        
        print("\n🧪 Test Predictions:")
        for i, example in enumerate(examples):
            pred = model.predict([example])[0]
            proba = model.predict_proba([example])[0]
            print(f"\nExample {i+1}: {example}")
            print(f"  Predicted: {species[pred]}")
            print(f"  Confidence: {proba[pred]:.1%}")
            
    except FileNotFoundError:
        print("⚠️  Model file not found. Run iris_project.py first to train and save the model.")
        print("\nTo get started:")
        print("  1. Run: python iris_project.py")
        print("  2. Then run: python demo.py")
    
    print("\n" + "=" * 40)
    print("For full project: python iris_project.py")

if __name__ == "__main__":
    quick_demo()