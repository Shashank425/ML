from matplotlib.pyplot import axis
import pandas as pd
from sklearn.metrics import r2_score
import wget
import numpy as np
import logging
import matplotlib.pyplot as plt


class MLR:

    def __init__(self):
        pass
        logging.basicConfig(filename="logfile.log",
                            filemode = "w",
                            format = "%(levelname)s %(asctime)s %(message)s",
                            level = logging.DEBUG)
        self.logger = logging.getLogger()

    def readData(self):
        url = "https://raw.githubusercontent.com/Shashank425/ML/main/mcs_ds_edited_iter_shuffled.csv"
        dataFile = wget.download(url)
        df = pd.read_csv(dataFile)
        #df = pd.read_csv("mcs_ds_edited_iter_shuffled.csv") #use after downloading once

        #if there are tuples with null replace it with the mean of the column data
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mean())
        
        #remove duplicate rows in dataframe
        df = df.drop_duplicates()
            
        #print first 5 rows of dataset from csv file
        print(df.head())
        #normalize the dataset
        df = (df - df.mean())/df.std()
        #training data
        train_x = df.iloc[0:int(0.8*df.shape[0]),0:4]
        train_y = df.iloc[0:int(0.8*df.shape[0]),4:5]
        self.logger.info('Training X data\n\t'+ train_x.to_string().replace('\n', '\n\t')) 
        self.logger.info('Training y data\n\t'+ train_y.to_string().replace('\n', '\n\t'))
        #test data
        test_x = df.iloc[int(df.shape[0]*0.8):df.shape[0],0:4]
        test_y = df.iloc[int(df.shape[0]*0.8):df.shape[0],4:5]
        self.logger.info('Testing X data\n\t'+ test_x.to_string().replace('\n', '\n\t')) 
        self.logger.info('Testing y data\n\t'+ test_y.to_string().replace('\n', '\n\t'))

        self.train(train_x,train_y,0.01,10000,test_x,test_y)

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

        print("MSE: ",cost)
        h = {"s1":slope,"s2":slope1,"s3":slope2,"s4":slope3,"const":c}
        rmse,r2 = self.checkRmseR2(test_x,test_y,h)
        self.logger.info(' RMSE Value: \n\t'+ str(rmse))
        print("RMSE= ",rmse)
        print("R squared= ",r2)



    def grad_desc(self,X,y,theta,alpha,iterations):
        cost = []
        for i in iterations:
            theta = theta - (alpha/len(X)) * np.sum((X@theta.T - y)*X , axis=0)
            cst = self.cost_calculation(X,y,theta)
            cost.append(cst)
        #write to logging file
        costStr = ""
        for s in cost:
            costStr = costStr + " " + str(s)

        #MSE vs Iterations plots
        plt.plot(range(1,len(cost)+1),cost)
        plt.title('MSE vs Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('MSE')
        plt.show()

        self.logger.info(' MSE changes as cost is reduced: \n\t'+ costStr) 

        return (theta,cst) 

    def cost_calculation(self,X,y,theta):
        cst = np.sum(np.power(((X@theta.T)-y),2))/(2*len(X))
        return cst

    def checkRmseR2(self,test_x,test_y,h):
        test_y_pred = []
        tss = 0
        for index,row in test_x.iterrows():
            test_y_pred.append(self.pred(row[0],row[1],row[2],row[3],h))
        for i,y_pred in enumerate(test_y_pred):
            rss = np.sum(np.power((y_pred-test_y["ale"].iloc[i]),2))
        for i in range(test_y.shape[0]):
            tss = tss + np.power((test_y["ale"].iloc[i] - np.mean(test_y['ale'])),2)
        
        r2 = 1 - rss/tss
        rmse = np.sqrt(rss/float(2*len(test_x))) 
        return rmse,r2
        
    def pred(self,x1,x2,x3,x4,h):
        y = x1*h["s1"]+x2*h["s2"]+x3*h["s3"]+x4*h["s4"]*h["const"]
        return y
    

if __name__ == '__main__':
    mlr = MLR()
    mlr.readData()


