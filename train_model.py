import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1️⃣ Load dataset
df = pd.read_csv("commit_data.csv")

# 2️⃣ Create label (temporary synthetic logic)
df["risky"] = ((df["insertions"] + df["deletions"]) > 100).astype(int)

# 3️⃣ Select features (NO DATA LEAKAGE)
X = df[["msg_len", "files_changed"]]  
y = df["risky"]

# 4️⃣ Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5️⃣ Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# 6️⃣ Train
model.fit(X_train, y_train)

# 7️⃣ Evaluate
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8️⃣ Save model
joblib.dump(model, "risk_model.pkl")
print("\nModel saved as risk_model.pkl")