import argparse
from mltools import ML
from data import Data
from utils import Utils

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Twitter Analysis')
        parser.add_argument("dataset", help="Path to dataset")
        parser.add_argument("task", help="Task: (hate,offensive,sentiment)")
        args = parser.parse_args()
        data=Data("hate/","hate")
        X_train,y_train,X_val,y_val,X_test,y_test=data.preprocess()
        ml=ML(data._best_model)
        print("Model Evaluation")
        ml.fit_evaluate(X_train,y_train,X_val,y_val,X_test,y_test)
