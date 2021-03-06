import numpy as np

def join_ones(X):
	'''It adds one more feature of ones with all the other features'''
	if X.shape[0]==X.size:
		X=[[i] for i in X.tolist()]
	else:
		X=X.tolist()
	X_=[]
	for i in X:
		a=[1,]
		for n in list(i):
			a.append(n)

		X_.append(a)
	return np.array(X_)
    
class one_vs_all:

    def __init__(self,epochs=100,alpha=0.01,lambda_=0.01):
        self.epochs=epochs
        self.alpha=alpha
        self.lambda_=lambda_

    def __train__(self,features,targets):
        X=features
        y=targets
        try:
            m,n=X.shape
        except:
            m=X.shape
            n=1
            
        theta=np.zeros(n)
        h0=lambda X:(1/(1+np.exp(-np.dot(X,theta))))
        
        for i in range(self.epochs):
            S=h0(X)
            gradient0=(1/m)*sum((S-y)*X[:,0].transpose())
            gradientj=((1/m)*((S-y)*X[:,1:].transpose()).sum(axis=1))+((self.lambda_/m)*theta[1:])
            gradient=np.concatenate((np.array([gradient0]),gradientj))
            theta=theta-(self.alpha*gradient)
        return theta

    def fit(self,X,y):
        k=range(len(set(y)))
        X=join_ones(X)
        self.models=[]
        for i in k:
            model=self.__train__(X,np.array(y==i,dtype=int))
            self.models.append(model)
        self.models=np.array(self.models)
        return self.models

    def predict(self,X):
        X=join_ones(X)
        h0=lambda X,model:(1/(1+np.exp(-np.dot(X,model))))
        confidence=[]
        for model in self.models:
            conf=h0(X,model)
            confidence.append(conf)
        confidence=np.array(confidence)
        confidence=confidence.transpose()
        prediction=[]
        for i in zip(confidence,confidence.max(axis=1)):
            a,b=i
            prediction.append(list(a).index(b))
        return np.array(prediction)
        
if __name__=='__main__':
    from sklearn.datasets import load_iris
    from sklearn.utils import shuffle
    from sklearn.model_selection import train_test_split
    data=load_iris()
    
    X,y=shuffle(data.data,data.target,random_state=1000)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=10)
    
    clf=one_vs_all(epochs=1000,lambda_=0.3,alpha=0.03)
    clf.fit(X_train,y_train)
    
    predTest=clf.predict(X_test)
    predTrain=clf.predict(X_train)
    
    Test_accuracy=round(np.array(predTest==y_test,dtype=int).mean()*100,1)
    Train_accuracy=round(np.array(predTrain==y_train,dtype=int).mean()*100,1)
    
    print('Train Accuracy: {} %'.format(Train_accuracy))
    print(' Test Accuracy: {} %'.format(Test_accuracy))
  
