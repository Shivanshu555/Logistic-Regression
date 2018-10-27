# Logistic-Regression
This respository contains some python programs related to Logistic Regression
class Binary_Logistic_Regression:
    '''Using this we are training a logistic model. Here we accept only two
features in our dataset and two (0,1) target variables.It runs using the algorithm
gradient descent. It is only for data that have two features.It fails on data
that have more than two features.

>>> clf=Logistic_Regression_Binary_GD(epochs=1000,alpha=0.1)
>>> clf.fit(X,y)
>>> clf.predict(X)

And we trained our model in just three lines of code.
For visualizing error vs iteration graph

>>> clf.plot_iteration_graph()

For visualizing decision boundary

>>> clf.plot_decision_boundary()

Here you change the value of xlabel,ylabel and background

>>> clf.plot_decision_boundary(
        xlabel='X0',ylabel='X1',cmap='GnBu',title='Decision Boundary')
    
'''
    
    def __init__(self,epochs=100,alpha=0.01,theta=np.array([0,0])):
        #Collection of all the things needed
        self.epochs=epochs
        self.alpha=alpha
        self.theta=theta

    def j(self,X,y,theta):
        '''Here we are calculating J(theta) and the partial derivative of J(theta)
          with respect to the theta (gradient)
          '''

        #list of gradients
        gradients=[]
        # number of the training examples
        m=len(y)
        # predicting 'y' with theta using our hypthesis h(theta)=1/(1+exp(X*theta))
        pred=self.h0(X,theta)

        # here we are calculating cost using our cost function
        l=-y*np.log(pred)-(1-y)*np.log(pred)

        # calulating and appending the values of gradients
        gradient1=(1/m)*np.sum( (self.predict(X,theta)-y) * X[:,0] )
        gradient2=(1/m)*np.sum( (self.predict(X,theta)-y) * X[:,1] )
        gradients.append(gradient1)
        gradients.append(gradient2)
        return np.sum(l)*(1/m),np.array(gradients)
    
    def fit(self,X,y):
        '''Here our actual training process is done.We have already defined other
needed thing in __init__ part so know we have to only use them'''
        
        self.X=X
        self.y=y
        self.g=lambda z:1/(1+np.exp(-z))
        self.h0=lambda X,theta:self.g(np.dot(X,theta))
        self.error=[]
        for i in range(self.epochs):
            jVal,gradients=self.j(self.X,self.y,self.theta)
            self.error.append(jVal)
            self.theta=self.theta-(self.alpha*gradients)
        return self.theta

    def predict(self,X,theta=None):
        '''It predicts the value of y on the basis of X with our parameters theta'''
        
        try:
            if theta==None:
                theta=self.theta
        except ValueError:
            theta=theta
        h=self.h0(X,theta)
        p=[]
        for i in h:
                if i>=0.5:
                        p.append(1)
                else:
                        p.append(0)
        return np.array(p)
    
    def plot_iteration_graph(self):
         '''Here we are plotting the error vs iteration graph that helps us to
visualize how learning rate is affecting our model.'''
        
         import matplotlib.pyplot as plt
         plt.plot(range(self.epochs),self.error)
         plt.xlabel('Iterations')
         plt.ylabel('Error')
         plt.title('Error vs Iteration graph')
         plt.show()

    def plot_decision_boundary(self,xlabel='X0',ylabel='X1',cmap='GnBu',title='Decision Boundary'):
        '''It is a copied part from Internet. Here to visualize fairly we are
plotting our decision boundary'''
        
        import matplotlib.pyplot as plt
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = self.X[:, 0].min() - .5, self.X[:, 0].max() + .5
        y_min, y_max = self.X[:, 1].min() - .5, self.X[:, 1].max() + .5
        h = .02  # step size in the mesh
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()],self.theta)

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z,cmap=cmap)

        # Plot also the training points
        color={1:'r',0:'g'}
        for n,i in enumerate(self.X):
            plt.scatter(i[0],i[1],color=color[self.y[n]])

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())

        plt.show()
