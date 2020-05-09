import numpy as np
import math
import pandas as pd

class GuassianClassifier:
    
    def fit(self, X, y):
        
        totalObservations=len(y['label'])
        
        cls, counts = np.unique(y, return_counts=True)
        self.class_freq = dict(zip(cls, counts))
        
        self.prob_0=self.class_freq[0]/totalObservations
        
        self.prob_1=self.class_freq[1]/totalObservations
        
        df = pd.concat([X, y], axis=1)

        self.meanCls={}
        self.varCls={}
        
        self.aggdf=df.groupby(df['label']).agg(["mean", "var"])
        aggdf = self.aggdf

        for cls in np.unique(y):
            self.meanCls[cls]={}
            self.varCls[cls]={}
            for i in range(1,23):
                feature='f{}'.format(i)
                self.meanCls[cls][feature]=aggdf.loc[cls,(feature,'mean')]
                self.varCls[cls][feature]=aggdf.loc[cls,(feature,'var')]               
                
        
    def getPriors(self):
        return self.prob_0,self.prob_1
    
    def getClasswiseMeanAndVariance(self):
        return self.meanCls, self.varCls
    
    def probability(self,x,f,cls):
        m=self.aggdf.loc[cls,(f,'mean')]
        v=self.aggdf.loc[cls,(f,'var')]
        
        exponent=math.exp(-(math.pow(x-m,2)/(2*v)))
        prob=(exponent/(math.sqrt(2*math.pi*v)))
        return prob
    
    def makePrediction(self,X_test):

        self.y_pred={}
        for row in range(0,len(X_test)):
            result={}
            p0=self.prob_0
            p1=self.prob_1
#             print("----- for row ",row," --- \n")

            for f in X_test.columns:
                x_row=X_test[f].loc[row]
#                 print(x_row)
                p0 *= self.probability(x_row,f,0)
                p1 *= self.probability(x_row,f,1)
                p=p0+p1
                result[0]=p0/p
                result[1]=p1/p

                self.y_pred[row]=int(max(result,key=result.get))
                
        return self.y_pred
    
    
    def AccuracyScore(self,ypred,ytest):
        count=0
        for i in ypred.keys():
            if(ytest[i]==ypred[i]):
                count += 1
        print("total values: ",len(ypred.keys()))
        print("wrong predictions: ",len(ypred.keys())-count)
        print("right predictions: ",count)
        return count/len(ytest)
