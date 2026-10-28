"""Later review implementation; not historical employer code."""
from pathlib import Path
import argparse
import ast
import json
import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def loan_local(path):
    df=pd.read_csv(path).dropna(subset=['Loan_Status'])
    X=df.drop(columns=['Loan_Status','Loan_ID']);y=df.Loan_Status.eq('Y').astype(int)
    Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
    num=list(X.select_dtypes(include='number').columns);cat=[c for c in X if c not in num]
    results={}
    for name,model in [('majority',DummyClassifier(strategy='most_frequent')),
        ('logistic',LogisticRegression(max_iter=2000)),
        ('forest',RandomForestClassifier(n_estimators=100,min_samples_leaf=5,random_state=42,n_jobs=1))]:
        pre=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),
           ('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore'))]),cat)])
        pipe=Pipeline([('preprocess',pre),('model',model)]).fit(Xtrain,ytrain)
        pred=pipe.predict(Xtest)
        results[name]={'accuracy':float(accuracy_score(ytest,pred)),
          'macro_f1':float(f1_score(ytest,pred,average='macro')),
          'confusion_true_rows_pred_columns_N_Y':confusion_matrix(ytest,pred,labels=[0,1]).tolist()}
    return {'kind':'NEW clean learning pipeline on archived dataset; NOT reproduction of historic 83.76%',
       'rows':len(df),'train':len(ytrain),'test':len(ytest),'target':'approval label, NOT default',
       'split_seed':42,'results':results,'warning':'No dated or external validation; no lending deployment claim.'}
if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--loan-csv',type=Path);a=p.parse_args()
    if a.loan_csv:
        result=loan_local(a.loan_csv)
    else:
        import tempfile
        rng=np.random.default_rng(42);n=240
        income=rng.normal(5000,1000,n);credit=rng.integers(0,2,n)
        label=(income/2000+credit+rng.normal(0,1,n)>3)
        d=pd.DataFrame({'Loan_ID':['S'+str(i) for i in range(n)],'ApplicantIncome':income,
                        'Credit_History':credit,'Education':rng.choice(['Graduate','Other'],n),
                        'Loan_Status':np.where(label,'Y','N')})
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'synthetic.csv';d.to_csv(path,index=False);result=loan_local(path)
        result['kind']='SYNTHETIC smoke test; not historical loan data'
    print(json.dumps(result,indent=2))
