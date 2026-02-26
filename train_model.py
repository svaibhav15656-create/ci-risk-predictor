import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
#load dataset
df = pd.read_csv("commit_data.csv")
#create label
df["risky"] = ((df["insertions"]+df["deletions"])>100).astype(int)
#select features
X = df[["msg_len","files_changed"]]
y = df["risky"]
#split data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
#create model
model = RandomForestClassifier(n_estimators=100,random_state=42)
#train
model.fit(X_train,y_train)
#evaluate
y_pred= model.predict(X_test)
accuracy = accuracy_score(y_pred,y_test)
print("model_accuracy",accuracy)

print("\nconfusion matrix")
print(confusion_matrix(y_test,y_pred))
print("\nclassification report:")
print(classification_report(y_pred,y_test))

#save model
joblib.dump(model,"risk_model.pkl")
print("\n model saved as risk_model.pkl")