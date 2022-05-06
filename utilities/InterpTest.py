
import scipy
import scipy.interpolate
import numpy as np
import matplotlib.pyplot as plt

xs = np.array([0, 1])
ys = np.array([[1, 1], [2, 2]])

interp = scipy.interpolate.CubicSpline(
    x=xs, y=ys, bc_type=((1, np.array([1, 0])*np.sqrt(2)), (1, np.array([0, 1])*np.sqrt(2))))

xps = np.linspace(0, 1, 101)

curve = interp(xps)
plt.plot(curve[:, 0], curve[:, 1], '.')

plt.show()
