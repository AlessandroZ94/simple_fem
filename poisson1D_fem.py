# Problem: -u(x)'' = f on omega=[0,1]
# f = 2
# BC, Dirichlet: u[0]=u[1]=0
# weak formulation: integral(u'v') = integral(2*v)
from scipy.sparse import diags
import numpy as np
import matplotlib.pyplot as plt
    
start = 0
end = 1
omega = end - start #domain
num = 15  #num nodes
num_elements = num-1
x = np.linspace(start, end, num)
f = 2 
v = []

for i in range(num):
    temp = np.zeros(num)
    temp[i] = 1
    v.append(temp)
v = np.asarray(v)

for i in range(num):
    plt.plot(x,v[i]) #plot of shape triangular functions 
plt.show()

# Ax=b
b =  np.full((num-2,1),f * omega/((num_elements))) # known vector


k = [
    np.full(num-3, -(num_elements)),         
    np.full(num-2, (num_elements)*2),       
    np.full(num-3, -(num_elements))          
]
offset = [-1, 0, 1]                  # Offsets for diagonals
A = diags(k, offset).toarray()       # Create the matrix and convert to array


print(A)
print(np.linalg.eig(A))
r = np.linalg.solve(A,b)

r = np.insert(r, 0, 0) #insert BC (u=0) in first position
r = np.insert(r, r.size, 0) #insert BC (u=0) in last postion

plt.plot(x,r)
plt.show()
