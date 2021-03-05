import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

Cobs = [188,
264,
340,
338,
338,
458,
540,
668,
803,
1031,
1401]

Aobs = [0.6,
2.9,
4.8,
5.4,
5.5,
7.9,
9.6,
11.4,
12.4,
14.1,
15.9]

plt.scatter(Cobs, Aobs)
plt.show()