import json
import numpy as np
import matplotlib.pyplot as plt
import random
from perceptron import Perceptron

with open('train_02.json', encoding='utf-8') as f:
    file = json.load(f)

TrainX = []
TrainY = []

for inside in file['inside']:
    TrainX.append([1, inside[0], inside[1], inside[0]*inside[0], inside[0]*inside[1], inside[1]*inside[1]])
    TrainY.append(-1)

for outside in file['outside']: 
    TrainX.append([1, outside[0], outside[1], outside[0]*outside[0], outside[0]*outside[1], outside[1]*outside[1]])
    TrainY.append(1)

TrainX = np.array(TrainX)
TrainY = np.array(TrainY)

alpha = Perceptron(TrainX, TrainY)
print(alpha)

Test = []
for i in range(50):
    Test.append([random.random()*2 - 1, random.random()*2 - 1])

TestX = []
for i in range(len(Test)):
    TestX.append([1, Test[i][0], Test[i][1], Test[i][0]*Test[i][0], Test[i][0]*Test[i][1], Test[i][1]*Test[i][1]])

# Я действительно не знаю как нарисовать кривую по уравнению(я пытался)
x1 = np.linspace(-1, 1.1, 100)
x2 = np.linspace(-1, 1.1, 100)
z = []
for i in range(len(x1)):
    for j in range(len(x2)):
        z.append([x1[i], x2[j]])
zX = []
for i in range(len(z)):
    zX.append([1, z[i][0], z[i][1], z[i][0]*z[i][0], z[i][0]*z[i][1], z[i][1]*z[i][1]])
    
fig, ax = plt.subplots()  

for i in range(len(zX)):
    if(abs(np.dot(zX[i], alpha)) < 0.05 ):
        ax.scatter(zX[i][1], zX[i][2], color='black')


for i in range(len(TestX)):
    if(np.dot(TestX[i], alpha) > 0):
        ax.scatter(TestX[i][1], TestX[i][2], color='green') # Outside ellipse Test
    if(np.dot(TestX[i], alpha) < 0):
        ax.scatter(TestX[i][1], TestX[i][2], color='orange') # Inside ellipse Test

for i in range(len(TrainX)):    
    if(TrainY[i] == -1):
        ax.scatter(TrainX[i][1], TrainX[i][2], color='blue') # Inside ellipse Train
    else:
        ax.scatter(TrainX[i][1], TrainX[i][2], color='red') # Outside ellipse Train
ax.grid()

plt.show()
