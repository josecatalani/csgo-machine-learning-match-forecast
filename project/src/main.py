import sys

from model import random_forest
from model import linear_regression

def main():
    match sys.argv[1]:
        case "random_forest":
            random_forest.RandomForestModel().predict()
        case "linear_regression":
            linear_regression.LinearRegressionModel().predict()
        case _:
            print("Invalid model name")
            sys.exit(1)

main()