####################################################################
#Required Packages
####################################################################
import pandas as pd
import numpy as np
import matplotlib as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
import joblib

####################################################################
#File Paths
####################################################################
INPUT_PATH="diabetes.data"
OUTPUT_PATH="diabetes.csv"
MODEL_PATH="diabetes_pipeline.joblib"

####################################################################
#Headers
####################################################################
HEADERS=[
    "Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin",
    "BMI","DiabetesPedigreeFunction","Age","Outcome"
]

####################################################################
#Function name: read_data
#Description: Read the data into pandas dataframe
#Input: path of CSV file
#Output: Give the data
#Author: Samruddhi Mahesh Kadam
#Date:10/8/24
####################################################################
def read_data(path):
    #Read the data into pandas dataframe
    data=pd.read_csv(path)
    return data

####################################################################
#Function name: get_headers
#Description: datset header
#Input: dataset
#Output: Returns headers
#Author: Samruddhi Mahesh Kadam
#Date:10/8/24
####################################################################
def get_headers(dataset):
    #Returns dataset headers
    return dataset.columns.values

####################################################################
#Function name: add headers
#Description: Add the headers to the dataset
#Input: dataset
#Output: Updated dataset
#Author: Samruddhi Mahesh Kadam
#Date:10/8/24
####################################################################
def add_headers(dataset,headers):
    #Add headers to dataset
    dataset.columns=headers
    return dataset

####################################################################
#Function name: data_file_to_csv
#Input: Nothing
#Output: Write the data to CSV
#Author: Samruddhi Mahesh Kadam
#Date:10/8/24
####################################################################
def data_file_to_csv():
    #Convert raw .data file to CSV with headers
    dataset=read_data(INPUT_PATH)
    dataset=add_headers(dataset,HEADERS)
    dataset.to_csv(OUTPUT_PATH,index=False)
    print("File is saved!!")

####################################################################
#Function name: handle_missing_values
#Description: filter missing values from dataset
#Input: dataset with missing values
#Output: dataset by removing missing values
#Author: Samruddhi Mahesh Kadam
#Date:10/8/24
####################################################################
def handle_missing_values(df,feature_headers):
    """
    Convert "0" to mean of the column
    """
    df[feature_headers]=df[feature_headers].mask(df[feature_headers]==0)
    df[feature_headers]=df[feature_headers].fillna(df[feature_headers].mean())
    return df

####################################################################
#Function name: split_dataset
#Description: split the dataset with train_percentage
#Input: dataset with related information
#Output: dataset after spliting
#Author: Samruddhi Mahesh Kadam
#Date:10/8/24
####################################################################
def split_dataset(dataset,train_percentage,feature_headers,target_header,random_state=42):
    """Split dataset into train/test"""
    x_train,x_test,y_train,y_test=train_test_split(dataset[feature_headers],dataset[target_header],train_size=train_percentage,random_state=random_state,stratify=dataset[target_header])

    return x_train,x_test,y_train,y_test

####################################################################
#Function name: dataset_statistics
#Description: Display the statistics
#Author: Samruddhi Mahesh Kadam
#Date:10/8/24
####################################################################
def dataset_statistics(dataset):
    print(dataset.describe(include="all"))

####################################################################
#Function name: build_pipeline
#Description: build pipeline
#Author: Samruddhi Mahesh Kadam
#Date:10/8/24
####################################################################
def build_pipeline():
    pipe=Pipeline(steps=[
        ("lf",LogisticRegression(
            n_jobs=-1,
            random_state=24,
            max_iter=1000
        ))
    ])
    return pipe

####################################################################
#Function name: train_pipeline
#Description: train pipeline
#Author: Samruddhi Mahesh Kadam
#Date:10/8/24
####################################################################
def train_pipeline(pipeline,X_train,Y_train):
    pipeline.fit(X_train,Y_train)
    return pipeline

####################################################################
#Function name: save_model
#Description:save the model
#Author: Samruddhi Mahesh Kadam
#Date:10/8/24
####################################################################
def save_model(model,path=MODEL_PATH):
    joblib.dump(model,path)
    print(f"Model loaded from {path}")
    return model

####################################################################
#Function name: save_model
#Description:save the model
#Author: Samruddhi Mahesh Kadam
#Date:10/8/24
####################################################################
def load_model(path=MODEL_PATH):
    model=joblib.load(path)
    print(f"Model loaded from {path}")
    return model
####################################################################
#Function name: main
#Description: main funstion from where execution starts
#Author: Samruddhi Mahesh Kadam
#Date:10/8/24
####################################################################
def main():
    """Load CSV"""
    dataset=pd.read_csv(OUTPUT_PATH)

    """Basic stats"""
    dataset_statistics(dataset)

    """Prepare features/target"""
    feature_headers=HEADERS[:-1] #drop target column,keep all
    target_header=HEADERS[-1]   #Outcome

    """Handle missing values"""
    dataset=handle_missing_values(dataset,feature_headers)

    """Split"""
    x_train,x_test,y_train,y_test=split_dataset(dataset,0.7,feature_headers,target_header)

    print("Train_X shape::",x_train.shape)
    print("Train_Y shape::",y_train.shape)
    print("Test_X shape::",x_test.shape)
    print("Test_Y shape::",y_test.shape)

    """Build+Train Pipeline"""
    pipeline=build_pipeline()
    trained_model=train_pipeline(pipeline,x_train,y_train)
    print("Trained Pipeline ::",trained_model)

    """Predict"""
    pred=trained_model.predict(x_test)

    """Metrics"""
    print("Train Accuracy::",accuracy_score(y_train,trained_model.predict(x_train)))
    print("Test Accuracy::",accuracy_score(y_test,pred))
    print("Classification report::\n",classification_report(y_test,pred))
    print("Confusion matrix::\n",confusion_matrix(y_test,pred))

    """Save model(Pipeline) using joblib """
    save_model(trained_model,MODEL_PATH)

    """Load model and test sample"""
    loaded=load_model(MODEL_PATH)
    sample=x_test.iloc[[0]]
    pred_loaded=loaded.predict(sample)
    print(f"Loaded model predictions of first test sample:: {pred_loaded[0]}")

####################################################################
#Application Starter
####################################################################
if __name__=="__main__":
    main()
