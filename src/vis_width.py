import numpy as np
import matplotlib.pyplot as plt 

N = 350
def ci_width(f):
	upper = f + np.sqrt(f*(1-f)/N)*3.891
	lower = f - np.sqrt(f*(1-f)/N)*3.891
	return upper - lower 

plt.figure()
x = np.arange(0, 1.01, 0.01)
y = []
for i in x:
	y.append(ci_width(i))

plt.plot(x, y)
plt.xlabel('Support')
plt.ylabel('Width')
plt.title('Support vs. Width for 99.99% CI')
plt.show()