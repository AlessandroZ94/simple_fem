# Problem: -u(x)'' = f(x) on O=[0,1]
# f(x) = 2
# BC: u[0]=u[1]=0
# weak formulation: integral(u'v') = integral(2*v)

import numpy as np
import matplotlib.pyplot as plt


start = 0
end = 1
space = end - start
num = 10
x = np.linspace(start, end, num)
f = 2 * np.ones(x.shape)
#f = 2
v = []

for i in range(num):
    temp = np.zeros(num)
    temp[i] = 1
    v.append(temp)
v = np.asarray(v)

for i in range(num):
    plt.plot(x,v[i])
plt.show()
b = np.empty([(num-2), 1])
a = np.zeros([(num-2), (num-2)])
for i in range(0, (num-2)):
    c = (1/2*(f[i]+f[i-1])+1/2*(f[i+1]+f[i]))/2 * space/((num-1))
    #c = f * space/((num-1))
    b[i, 0] = c    

for i in range(0, (num-2)):
    for j in range(0, (num-2)):
        if i==j:
            a[i, j] = (num-1)*2
        if abs(i-j) == 1:
            a[i, j] = -(num-1)

print(a)
r = np.linalg.solve(a,b)

r = np.insert(r, 0, 0)
r = np.insert(r, r.size, 0)
#print(r)
#print(v)



plt.plot(x,r)
plt.show()
