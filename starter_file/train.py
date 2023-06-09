from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Workspace, Dataset


def clean_data(data):

    # Clean data. NOTE: this dataset does not have any categorical values and thus no need to do one hot encoding
    x_df = data.dropna()
    y_df = x_df.pop("DEATH_EVENT")
    return x_df, y_df


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0,
                        help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=200, help="Maximum number of iterations to converge")

    parser.add_argument('--intercept_scaling', type=float, default=1.0, help="Scaling the ")


    args = parser.parse_args()

    run = Run.get_context()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    
    # Get data
    
    webpath = "https://raw.githubusercontent.com/Ed-Ramos/ML-with-Azure-Capstone/master/heart_failure_clinical_records_dataset.csv"
    #ds = Dataset.Tabular.from_delimited_files(path=web_path)
    ds = pd.read_csv(webpath)

    x, y = clean_data(ds)

    # Split data into train and test sets.

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.67, random_state=42)

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    # create an output folder
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/model.joblib')


if __name__ == '__main__':
    main()
