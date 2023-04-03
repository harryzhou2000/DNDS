# %%
import numpy as np
import matplotlib.pyplot as plt



fname = "out/test_Euler_.NACA0012_WIDE_H4_M_2023-04-03_276352.log"

for rep in range(10):

    fin = open(fname, "r")
    lines = fin.readlines()
    fin.close()

    headline = lines[0]
    nData = headline.split().__len__()
    nLines = lines.__len__()

    dataIn = np.zeros([nLines, nData])

    for iLine in range(nLines):
        dataIn[iLine, :] = np.array(
            list(map(lambda x: float(x), lines[iLine].split())))

    dataInInner = dataIn[dataIn[:, 1] > 0, :]

    isee = 2 # res_rho

    fig = plt.figure(1, figsize=np.array([4, 3]) * 2, facecolor = 'white')
    ax = plt.axes()
    ax.set_yscale('log')
    ax.grid('both')
    ax.set_xlabel('n_iterin')
    plt.plot(dataInInner[:,1], dataInInner[:,isee])
    plt.draw()
    fig.set_frameon(True)
    fig.savefig('cur.png')




# %%




