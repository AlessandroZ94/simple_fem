# Problem: -u(x)'' = f on O=[0,1]
# f = 2
# BC, Dirichlet: u[0]=u[1]=0
# weak formulation: integral(u'v') = integral(2*v)

import numpy as np
import matplotlib.pyplot as plt

def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)
    
start = 0
end = 1
omega = end - start
num = 10
x = np.linspace(start, end, num)
f = 2 
v = []

for i in range(num):
    temp = np.zeros(num)
    temp[i] = 1
    v.append(temp)
v = np.asarray(v)

for i in range(num):
    plt.plot(x,v[i])
plt.show()
b =  np.full((num-2,1),f * omega/((num-1)))
a = np.zeros([(num-2), (num-2)])


lower_upper_diag = np.full((num-3,1),-(num-1))
main_diag = np.full((num-2,1), (num-1)*2)
A = tridiag(lower_upper_diag, main_diag, lower_upper_diag)

print(a)
r = np.linalg.solve(a,b)

r = np.insert(r, 0, 0)
r = np.insert(r, r.size, 0)

plt.plot(x,r)
plt.show()
