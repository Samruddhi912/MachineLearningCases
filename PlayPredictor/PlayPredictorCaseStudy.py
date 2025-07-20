import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def PlayPredict(DataPath):
    Line="*"*50
    df=pd.read_csv(DataPath)
    print(Line)
    print("Dataset is:")
    print(df.head())
    print(Line)

    df.drop(columns="Unnamed: 0",inplace=True)
    print("Dataset after Cleaning is:")
    print(df.head())
    print(Line)

    print("Number of missing values in dataset are:")
    print(df.isnull().sum())

    df["Whether"]=df["Whether"].map({"Sunny":1,"Overcast":2,"Rainy":3})
    df["Temperature"]=df["Temperature"].map({"Hot":1,"Mild":2,"Cool":3})
    df["Play"]=df["Play"].map({"No":0,"Yes":1})
    print(Line)
    print("Encoded Data is:")
    print(df.head())
    print(Line)

    print("Statistical Summary of dataset is:")
    print(df.describe())

    x=df[["Whether","Temperature"]]
    y=df["Play"]

    print(Line)
    print("Dataset lenght is:",x.shape)

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

    model=KNeighborsClassifier(n_neighbors=3)

    model.fit(x_train,y_train)

    y_pred=model.predict(x_test)

    accurarcy=accuracy_score(y_test,y_pred)

    print(Line)
    print("Testing the model:")
    test=pd.DataFrame([["1","3"]],columns=["Whether","Temperature"])
    ans=model.predict(test)
    print("Prediction of Sunny and Cool is:",ans[0])
    
    print(Line)
    print("Accurarcy when k=3:",accurarcy)

    model=KNeighborsClassifier(n_neighbors=7)

    model.fit(x_train,y_train)

    y_pred=model.predict(x_test)

    accurarcy=accuracy_score(y_test,y_pred)

    print(Line)
    print("Accurarcy when k=7:",accurarcy)

    model=KNeighborsClassifier(n_neighbors=5)

    model.fit(x_train,y_train)

    y_pred=model.predict(x_test)

    accurarcy=accuracy_score(y_test,y_pred)

    print(Line)
    print("Accurarcy when k=5:",accurarcy)
    print(Line)

def main():
    PlayPredict("PlayPredictor.csv")

if __name__=="__main__":
    main()
