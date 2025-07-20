import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

def WinePrediction(DataPath):
    Line="-"*60
    df=pd.read_csv(DataPath)
    print(Line)
    print(df.head())

    print("Missing Enteries in Dataset: ")
    print(df.isnull().sum())
    print(Line)

    print("Statistical Summary:")
    print(df.describe())
    print(Line)

    x=df.drop(columns=["Class"])
    y=df["Class"]

    print("Lenght of dataset is")
    print(x.shape)
    print(Line)

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

    model=DecisionTreeClassifier()

    model.fit(x_train,y_train)

    y_pred=model.predict(x_test)

    accuracy=metrics.accuracy_score(y_test,y_pred)

    print("Accuracy is:",accuracy)
    print(Line)


def main():
    WinePrediction("WinePredictor.csv")


if __name__=="__main__":
    main()
