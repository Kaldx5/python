import numpy as np  
import matplotlib.pyplot as plt 
from sklearn import svm 
from matplotlib import style 
style.use("ggplot")

x = np.array([[3,2],
             [6,6],
             [2.6,3] ,
             [7,8],
             [3.5,5],
             [6,11],])

i = [0,1,0,1,0,1]

my_clf = svm.SVC(kernel='linear', C= 1.0)
my_clf.fit(x,i)

print(" SVR proedict[0.5,0.8] : ", my_clf.predict([[0.5,0.8]]))

print(" SVR proedict[8.5,10] : ", my_clf.predict([[8.5,10]]))

plt.scatter(x[:,0], x[:,1], c=i)
plt.scatter(0.5,0.8, c='r')
plt.scatter(8.5,10, c='r')
plt.show()
