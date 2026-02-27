import joblib

# Load model
model = joblib.load("risk_model.pkl")

# Take user input
msg_len = int(input("Enter commit message length: "))
files_changed = int(input("Enter number of files changed: "))

# Predict
prediction = model.predict([[msg_len, files_changed]])

if prediction[0] == 1:
    print("⚠ This commit is predicted as RISKY")
else:
    print("✅ This commit is predicted as SAFE")