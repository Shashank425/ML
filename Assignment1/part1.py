from matplotlib.pyplot import axis
import pandas as pd
import wget
import numpy as np


class MLR:

    def __init__(self):
        pass

    def readData(self):
        # url = "https://raw.githubusercontent.com/Shashank425/ML/main/mcs_ds_edited_iter_shuffled.csv"
        # dataFile = wget.download(url)
        df = pd.read_csv("mcs_ds_edited_iter_shuffled.csv")
        #print first 5 rows of dataset from csv file
        print(df.head())
        #normalize the dataset
        df = (df - df.mean())/df.std()
        #training data
        train_x = df.iloc[0:85,0:4]
        train_y = df.iloc[0:85,4:5]
        #test data
        test_x = df.iloc[85:107,0:4]
        test_y = df.iloc[85:107,4:5]
        self.train(train_x,train_y,0.1,1000,test_x,test_y)

    def train(self,x_inp,y_inp,learning_rate,epoch,test_x,test_y):
        alpha = learning_rate
        iterations = range(0,epoch)
        X = x_inp
        #add ones for the column x0
        ones = np.ones([X.shape[0],1])
        X = np.concatenate((ones,X),axis=1)
        y = y_inp.values
        theta = np.zeros([1,5])

        theta1, cost = self.grad_desc(X,y,theta,alpha,iterations)
        slope = theta1[0][0]
        slope1 = theta1[0][1]
        slope2 = theta1[0][2]
        slope3 = theta1[0][3]
        c = theta1[0][4]

        h = {"s1":slope,"s2":slope1,"s3":slope2,"s4":slope3,"const":c}
        rmse1 = self.checkRmse(x_inp,y_inp,h)
        print("rmseTrain = ",rmse1)
        rmse = self.checkRmse(test_x,test_y,h)
        print("rmseTest= ",rmse)



    def grad_desc(self,X,y,theta,alpha,iterations):
        cost = []
        for i in iterations:
            theta = theta - (alpha/len(X)) * np.sum((X@theta.T - y)*X , axis=0)
            cst = self.cost_calculation(X,y,theta)
            cost.append(cst)
        return (theta,cst) 

    def cost_calculation(self,X,y,theta):
        cst = np.sum(np.power(((X@theta.T)-y),2))/(2*len(X))
        return cst

    def checkRmse(self,test_x,test_y,h):
        test_y_pred = []
        for index,row in test_x.iterrows():
            test_y_pred.append(self.pred(row[0],row[1],row[2],row[3],h))
        for i,y_pred in enumerate(test_y_pred):
            ss = np.sum(np.power((y_pred-test_y["ale"].iloc[i]),2))
        rmse = np.sqrt(ss/float(2*len(test_x)))
        return rmse

        
    def pred(self,x1,x2,x3,x4,h):
        y = x1*h["s1"]+x2*h["s2"]+x3*h["s3"]+x4*h["s4"]*h["const"]
        return y
    

if __name__ == '__main__':
    mlr = MLR()
    mlr.readData()


