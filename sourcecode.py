import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


train.fillna({
    'Gender': 'Male',
    'Married': 'No',
    'Dependents': '0',
    'Self_Employed': 'No',
    'LoanAmount': train['LoanAmount'].median(),
    'Loan_Amount_Term': train['Loan_Amount_Term'].median(),
    'Credit_History': 1.0
}, inplace=True)


le = LabelEncoder()

cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']

for col in cols:
    train[col] = le.fit_transform(train[col])


    train['Dependents'] = train['Dependents'].replace('3+', 3).astype(int)


    train['Total_Income'] = train['ApplicantIncome'] + train['CoapplicantIncome']
train['EMI'] = train['LoanAmount'] / train['Loan_Amount_Term']
train['Income_to_Loan_Ratio'] = train['Total_Income'] / train['LoanAmount']

X = train.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = train['Loan_Status'].map({'Y': 1, 'N': 0})

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)


model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train_sm, y_train_sm)


y_pred = model.predict(X_val)

print("Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))


y_prob = model.predict_proba(X_val)[:, 1]

def risk_label(prob):
    if prob > 0.75:
        return "Low Risk"
    elif prob > 0.5:
        return "Medium Risk"
    else:
        return "High Risk"

risk_scores = [risk_label(p) for p in y_prob]