# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Startv the program.
2. import numpy as np.
3. Give the header to the data.
4. Find the profit of population.
5. Plot the required graph for both for Gradient Descent Graph and Prediction Graph.
6. End the program.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: T.LISIANA
RegisterNumber: 212222240053 
*/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data=pd.read_csv("/content/ex1.txt",header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of city (10,000s)")
plt.ylabel("Profit ($10,000")
plt.title("Profit Predication")

def computeCost(x,y,theta):
  m=len(y)
  h=x.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err)
  
data_n=data.values
m=data_n[:,0].size
x=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(x,y,theta)

def gradientDescent(x,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]
  for i in range(num_iters):
    predictions=x.dot(theta)
    error=np.dot(x.transpose(),(predictions-y))
    descent=alpha*1/m*error
    theta-=descent
    J_history.append(computeCost(x,y,theta))
  return theta,J_history
  
theta,J_history = gradientDescent(x,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")


plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color='r')
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000")
plt.title("Profit Prediction")

def predict(x,theta):
  predictions= np.dot(theta.transpose(),x)
  return predictions[0]
  
predict1=predict(np.array([1,3.5]),theta)*1000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*1000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))

```

## Output:

## Profit Prediction graph
![image](https://github.com/lisianathiruselvan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119389971/a0637537-8378-442b-b166-61fb935bb744)

## Compute Cost Value
![image](https://github.com/lisianathiruselvan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119389971/ef36aeef-7a8e-44a0-a49e-9de7d79ac58f)

## h(x) Value
![image](https://github.com/lisianathiruselvan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119389971/20cdf0cf-af29-4bdf-b401-8c9402405345)

## Cost function using Gradient Descent Graph
![image](https://github.com/lisianathiruselvan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119389971/26f782eb-eec0-4712-a54b-c8d3e02ec123)

## Profit Prediction Graph
![image](https://github.com/lisianathiruselvan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119389971/665a96b2-bd0a-42eb-b79f-271cbc47fa3f)

## Profit for the Population 35,000
![image](https://github.com/lisianathiruselvan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119389971/d9d68600-6f81-4a8e-bc38-a39c97d789e7)

## Profit for the Population 70,000
![image](https://github.com/lisianathiruselvan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119389971/40dca0ba-9e1b-4c33-8c3e-d5397320160c)

## Result:

Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
