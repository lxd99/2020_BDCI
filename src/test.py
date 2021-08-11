import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import numpy as np
import matplotlib.pyplot as plt
a = [5,6,7,8]
b = [5,5,5,5]
c = [7,3,5,7]
plt.figure()
plt.plot(a,b,'r',label=f'T={0}')
plt.plot(a,c,'b',label=f'T={5}')
plt.xlabel('iters')
plt.ylabel('loss')
plt.title('cool')
plt.legend()
plt.show()
