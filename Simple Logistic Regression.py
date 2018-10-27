import numpy as np
class Simple_Logistic_Regression:
    def __init__(self,alpha=0.01,epochs=100):
        self.alpha=alpha
        self.epochs=epochs
        
    def predict(self,X,theta=None):
        try:
            if theta==None:
                theta=self.theta
        except ValueError:
            theta=theta    
        h=self.h0(X,theta)
        p=[]
        try:
            for i in h:
                    if i>=0.5:
                            p.append(1)
                    else:
                            p.append(0)
        except:
            for i in np.array([h]):
                    if i>=0.5:
                            p.append(1)
                    else:
                            p.append(0)
        return np.array(p)    
    def J(self,thetas):
        gradients=[]
        
        self.g=lambda z:1/(1+np.exp(-z))
        self.h0=lambda X,theta:self.g(np.dot(X,theta))
        
        pred_percent=self.h0(X,thetas)
        
        prediction=self.predict(self.X,thetas)
        error=prediction-self.y
        
        l=-y*np.log(pred_percent)-(1-y)*np.log(pred_percent)

        k=0
        for theta in thetas:
            gradient=(1/self.m)*np.sum(error*self.X[:,k])
            gradients.append(gradient)
            k+=1
        
        return np.array(gradients),np.sum(l)*(1/self.m)
    
    def fit(self,X,y):
        self.X=X
        self.y=y
        try:
            self.m,self.n=self.X.shape
        except ValueError:
            self.m=self.X.shape
            self.n=1
        self.theta=np.zeros(self.n)
        self.errors=[]
        for iterations in range(self.epochs):
            gradients,cost=self.J(self.theta)
            self.errors.append(cost)
            self.theta=self.theta-(self.alpha*gradients)
        return self.theta

    def iteration_vs_error_plot(self):
        import matplotlib.pyplot as plt
        plt.plot(range(self.epochs),self.errors)
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.title('Iteration vs Error plot')
        plt.show()
