import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# === 1) Load your CSV ===
df = pd.read_csv('behavior_dataset.csv')  # <-- use your CSV file name!

print("\n✅ First rows:")
print(df.head())

# === 2) Features & Labels ===
X = df[['hand_speed', 'eye_contact', 'head_tilt', 'pose_distance']]
y = df['label']

# === 3) Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 4) Create & Train SVM ===
clf = SVC(kernel='rbf')
clf.fit(X_train, y_train)

# === 5) Evaluate ===
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc:.2f}")

# === 6) Save model ===
joblib.dump(clf, 'behavior_model.pkl')
print("\n✅ Saved trained model as 'behavior_model.pkl'")

# === 7) Try predicting one example ===
sample = [[10, 1, 0.1, 0.4]]
result = clf.predict(sample)
print(f"\nExample Prediction for {sample}: {result[0]}")
