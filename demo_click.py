from __future__ import print_function

from six.moves import input

from matplotlib.widgets import LassoSelector
from matplotlib.path import Path






import time
import copy
import pickle
import numpy as np
import pandas as pd
import seaborn as sn
from tqdm import tqdm
import matplotlib.pyplot as plt

import sklearn
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from models import *
from utils import *
from datasets import *

# import numpy as np
# import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title('click on points')

# line, = ax.plot(np.random.rand(100), 'o', picker=5)  # 5 points tolerance

# def onpick(event):
#     thisline = event.artist
#     xdata = thisline.get_xdata()
#     ydata = thisline.get_ydata()
#     ind = event.ind
#     points = tuple(zip(xdata[ind], ydata[ind]))
#     print('onpick points:', points)


# fig.canvas.mpl_connect('pick_event', onpick)

# plt.show()






# from pylab import *
# from matplotlib.path import Path
# import matplotlib.patches as patches

# data = np.random.rand(100,4)

# verts = [(0.3, 0.7), (0.3, 0.3), (0.7, 0.3), (0.7, 0.7)]

# path1 = Path(verts)
# index = path1.contains_points(data[:,:2])

# print(data[index, :2])

# plot(data[:,0],data[:,1], 'b.')
# patch = patches.PathPatch(path1, facecolor='orange', lw=2)
# gca().add_patch(patch)
# plot(data[index,0], data[index,1], 'r.')
# show()





def create_counter():

    def increase():
        n = 0
        while True:
            n += 1
            yield n

    it = increase()

    def counter():
        return next(it)

    return counter

counter_ = create_counter() # 0

class SelectFromCollection(object):
    """Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool highlights
    selected points by fading them out (i.e., reducing their alpha values).
    If your collection has alpha < 1, this tool will permanently alter them.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.

    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to `alpha_other`.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, self.Npts).reshape(self.Npts, -1)

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []
        self.result = {}

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero([path.contains_point(xy) for xy in self.xys])[0]
        # print("self.ind: {}".format(type(self.ind), self.ind.shape))
        # np.save("./clusters_{}.npy".format(counter_()), self.xys[self.ind])
        self.result["clusters_{}".format(counter_())] = self.xys[self.ind]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.ion()
    # data = np.random.rand(100, 2)
    data = normalise(torch.load("./data/Z_train_pretrainDR_60k.pt"))

    subplot_kw = dict(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1), autoscale_on=False)
    fig, ax = plt.subplots(subplot_kw=subplot_kw)

    ax.set_xmargin(0.1)
    ax.set_ymargin(0.1)

    pts = ax.scatter(data[:, 0], data[:, 1], s=1)
    selector = SelectFromCollection(ax, pts)

    plt.draw()
    input('Press Enter to accept selected points')
    print("Selected points:")
    print(selector.xys[selector.ind])
    selector.disconnect()

    # save dictionary to person_data.pkl file
    with open('clusters.pkl', 'wb') as fp:
        pickle.dump(selector.result, fp)
        print('dictionary saved successfully to file')

    # Block end of script so you can check that the lasso is disconnected.
    input('Press Enter to quit')