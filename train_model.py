import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
#load dataset
df=pd.read_csv("commit_data.csv")
#create features
df["risky"]=((df["insertions"]+df["deletions"])>100).astype(int)
#select features
X = df[["msg_len","files_changed"]]
y=df["risky"]
#split data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
#create model
model = RandomForestClassifier(n_estimators=100,random_state=42)   # trained only on the train data
model.fit(X_train,y_train)                                         # trained only on the train data
#evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print("accuracy score of the model is",accuracy)
print("confusion matrix")
print(confusion_matrix(y_test,y_pred))
print("classification report")
print(classification_report(y_test,y_pred))
joblib.dump(model,"risk_model.pkl")
print("model saved as risk_model.pkl")

