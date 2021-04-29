from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings("ignore")


class ML:
    ##### Initializing Classifiers with best parameters (after research)
    def __init__(self,classifier):
        try:
            if(classifier=="lr"):
                print("Using LogisticRegression Classifier")
                self.model=LogisticRegression(penalty="l2",C=1.0,solver="lbfgs")
            elif(classifier=="svc"):
                print("Using Linear SVC Classifier")
                self.model=LinearSVC()
            elif(classifier=="rf"):
                print("Using RandomForest Classifier")
                self.model=RandomForestClassifier(n_estimators=100,criterion="gini",max_depth=None)
            elif(classifier=="xgb"):
                print("Using XGB Classifier")
                self.model=XGBClassifier()
            else:
                raise NotImplementedError("Model not implemented")
        except Exception as e:
            print("Error initializing parameters. Check parameter name and values")
            print(e)

    #### Fit model on training set and evaluate performance on test set
    #### Evaluation metric F-1 Score
    def fit_evaluate(self,X_train,y_train,X_val,y_val,X_test,y_test):
        self.model.fit(X_train,y_train)
        predictions = self.model.predict(X_val)
        print("Validation F-1 Score")
        print('f1:',f1_score(y_val, predictions,average="macro"))
        print()

        predictions = self.model.predict(X_test)
        print("Testing F-1 Score")
        print('f1:',f1_score(y_test, predictions,average="macro"))
        print()
