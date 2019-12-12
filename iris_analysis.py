import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
     #データの読み込み
    df = pd.read_csv(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
    df.columns = ["sepal length", "sepal width", "petal length", "petal width", "class"]

    #ndarray型に変換
    #説明変数
    data = df.drop("class",axis=1).values
    data_y = df['class'].values

    data_setosa = data[data_y=="Iris-setosa"]
    data_versicolor = data[data_y=="Iris-versicolor"]
    data_virginica = data[data_y=="Iris-virginica"]

    i = 2
    j = 0
    
    plt.scatter(data_setosa[:,i],data_setosa[:,j],color="red",label="setosa")
    plt.scatter(data_versicolor[:,i],data_versicolor[:,j],color="blue",label="versicolor")
    plt.scatter(data_virginica[:,i],data_virginica[:,j],color="green",label="virginica")

    plt.xlabel(df.columns[i])
    plt.ylabel(df.columns[j])
    
    plt.legend()
    plt.show()
    
    
