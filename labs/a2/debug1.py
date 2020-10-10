import numpy as np

quad = lambda x: (np.sum(x ** 2), x * 2)
print(quad(2))