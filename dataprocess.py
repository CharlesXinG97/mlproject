import numpy
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

data= np.genfromtxt('C:/Users/dell/Desktop/ml2018/res.txt', delimiter=' ',dtype=None)
print('loaded')
#print(data.shape,data[0])

#a=data[0]
#print(a[0][0],a[0][1],a[0][2],a[0][3],a[0][4],a[0][9],a[0][9]==67)
X=np.zeros((66347,4,300))
Y=np.zeros((66347,37))
for i in range(0,66347):
    a=data[i]
    b=a[0]
    #print(i)
    for j in range(1,38):
        if a[j]==1:
            Y[i][j-1]=1
    for k in range(0,300):
        if b[k]==65:
            X[i][0][k]=1
        if b[k]==84:
            X[i][1][k]=1
        if b[k]==71:
            X[i][2][k]=1
        if b[k]==67:
            X[i][3][k]=1
np.save('C:/Users/dell/Desktop/ml2018/X',X)
np.save('C:/Users/dell/Desktop/ml2018/Y',Y)
