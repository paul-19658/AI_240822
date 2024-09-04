import numpy as np
import time
import random

'''
    向量化代码，对比for循环和numpy的效率差异
    np.dot(a,b) result=249825.02337924892
    Vectorized version duration: 13.940811157226562 ms
    For loop version result=249825.02337923684
    For loop version duration: 357.04588890075684 ms
    对于多元线性回归这样的公式，向量化的效率要高于for循环，因为for循环的效率是O(n^2)，而向量化的效率是O(n)
'''

np.random.seed(1)
a = np.random.rand(1000000) # very large arrays
b = np.random.rand(1000000)
tic = time.time()
c = np.dot(a,b)
toc = time.time()

print('np.dot(a,b) result={}'.format(c))
print('Vectorized version duration: {} ms'.format(1000*(toc-tic)))
tic = time.time()
c = 0
for i in range(1000000):
    c += a[i]*b[i]

toc = time.time()
print('For loop version result={}'.format(c))
print('For loop version duration: {} ms'.format(1000*(toc-tic)))