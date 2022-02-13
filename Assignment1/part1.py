from matplotlib.pyplot import axis
import pandas as pd
from pyparsing import col
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import wget
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sb


class MLR:

    def __init__(self):
        self.trail_data = pd.DataFrame(columns=['Alpha','Iterations','Coefficients','Intercept','Mean Squared Error','Root Mean Squared Error','Mean Absolute Error','R Squared Value'])
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.basicConfig(filename="logfile.log",
                            filemode = "w",
                            format = "%(levelname)s %(asctime)s %(message)s",
                            level = logging.INFO)
        self.logger = logging.getLogger()

    def readAndRunData(self):
        url = "https://raw.githubusercontent.com/Shashank425/ML/main/mcs_ds_edited_iter_shuffled.csv"
        dataFile = wget.download(url)
        self.df = pd.read_csv(dataFile)
        #self.df = pd.read_csv("mcs_ds_edited_iter_shuffled.csv") #use after downloading once

        #if there are tuples with null replace it with the mean of the column data
        for col in self.df.columns:
            self.df[col] = self.df[col].fillna(self.df[col].mean())
        
        #remove duplicate rows in dataframe
        self.df = self.df.drop_duplicates()
            
        #print first 5 rows of dataset from csv file
        print(self.df.head())

        #split dataframe into X features and y output
        X_feature = self.df.loc[:, 'anchor_ratio': 'iterations']
        y_out = self.df.loc[:, ['ale']]

        #normalize the dataset
        scaler = StandardScaler()
        X = scaler.fit_transform(X_feature)
        y = scaler.fit_transform(y_out)

        #training and testing data
        self.train_x,self.test_x,self.train_y,self.test_y = train_test_split(X,y,test_size=0.2,)

        self.logger.info('Training X data\n'+ np.array2string(self.train_x)) 
        self.logger.info('Training y data\n'+ np.array2string(self.train_y))
        self.logger.info('Testing X data\n'+ np.array2string(self.test_x)) 
        self.logger.info('Testing y data\n'+ np.array2string(self.test_y))
        #Values for different trials

        trial_learning_rates = [0.001, 0.01, 0.1]
        trial_iterations = [5000,2000,1000]
        
        for l in trial_learning_rates:
            for epoch in trial_iterations:
                self.trial(l,epoch)
        print(self.trail_data)
        self.logger.info("\n"+self.trail_data.to_string())
        self.plotGraphs()

    #gradient descent algorithm for linear regression
    def grad_desc(self,X,y,learning_rate,epoch):
        alpha = learning_rate
        theta = np.random.randn(X.shape[1],1)
        b = np.random.randn(1)
        self.y_pred_ls = []
        mse=[]
        for i in range(0,epoch):
            y_pred = self.pred(X,theta,b)
            diff_y = y_pred - y
            diff_theta = np.dot(X.T,diff_y) / X.shape[0]
            theta = theta - alpha*diff_theta
            diff_b = np.sum(diff_y)/X.shape[0]
            b = b - alpha*diff_b
            self.y_test_predicted = self.pred(self.test_x,theta,b)
            mse.append(metrics.mean_squared_error(self.test_y,self.y_test_predicted))
        #MSE vs Iterations plots
        plt.plot(range(1,len(mse)+1),mse)
        plt.title('MSE vs Iterations LR = '+str(learning_rate)+" Iter= "+str(epoch))
        plt.xlabel('Iterations')
        plt.ylabel('MSE')
        f_name = 'MSE vs Iterations LR = '+str(learning_rate)+" Iter= "+str(epoch)+".png"
        plt.savefig(f_name)
        plt.show()
            

        return theta,b
    
    def trial(self,learning_rate,epoch):
        theta,b = self.grad_desc(self.train_x,self.train_y,learning_rate,epoch)
        
        
        #calculate the metrics
        mse = metrics.mean_squared_error(self.test_y,self.y_test_predicted)
        rmse = np.sqrt(metrics.mean_squared_error(self.test_y,self.y_test_predicted))
        r2 = metrics.r2_score(self.test_y,self.y_test_predicted)
        mae = metrics.mean_absolute_error(self.test_y,self.y_test_predicted)

        res = {'Alpha':learning_rate,
                'Iterations':epoch,
                'Coefficients':theta,
                'Intercept':b,
                'Mean Squared Error':mse,
                'Root Mean Squared Error':rmse,
                'Mean Absolute Error':mae,
                'R Squared Value':r2}
        self.trail_data = self.trail_data.append(res,ignore_index=True)
        

    def pred(self,X,theta,b):
        y = np.dot(X,theta)+b
        return y

    
    def plotGraphs(self):
        #Heatmap
        sb.heatmap(self.df.corr(),linewidth=0.5)
        plt.title("HeatMap")
        plt.savefig("Heatmap.png")
        plt.show()
        #Imp feature vs target
        plt.plot(self.df['ale'],self.df['node_density'])
        plt.title('important Feature vs Target')
        plt.xlabel('Important Feature')
        plt.ylabel('Target')
        plt.savefig('ImportantFeatureVSTarget.png')
        plt.show()
        #R2 value vs iterations
        grouped = self.trail_data.groupby('Alpha')
        cnt = 1
        for name, group in grouped:
            plt.plot(group['Iterations'], group['R Squared Value'])
            plt.xlabel("Iterations")
            plt.ylabel("R Squared Value")
            plt.title("Iterations VS R Squared Value for Alpha: " + str(name))
            f_name = "Iterations VS R Squared " + str(cnt) + ".png"
            plt.savefig(f_name)
            cnt = cnt + 1
            plt.show()
        #RMSE vs iterations
        cnt = 1
        for name, group in grouped:
            plt.plot(group['Iterations'], group['Root Mean Squared Error'])
            plt.xlabel("Iterations")
            plt.ylabel("RMSE")
            plt.title("Iterations VS RMSE for Alpha: " + str(name))
            f_name = "Iterations VS RMSE " + str(cnt) + ".png"
            plt.savefig(f_name)
            cnt = cnt + 1
            plt.show()
        

        
    

if __name__ == '__main__':
    mlr = MLR()
    mlr.readAndRunData()


