import pandas as pd
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import mplcursors as mplc
import geopandas
import PyQt5 as pq
import json
import math
from sklearn.neighbors import KernelDensity
import seaborn as sns

fig, ax = plt.subplots(1,1)
eqData = pd.read_csv('geodata.csv')
world = geopandas.read_file('us_states.json')
eqData = eqData.dropna()
#print(eqData)
X = []
Y = []

for x in eqData['LON']:
    X.append(x)

#print(X)

for y in eqData['LAT']:
    Y.append(y)

magTypes = []
for mag in eqData['MAG']:
    mag = int(mag)
    if mag >= 1.5 and mag < 3.5:
        magTypes.append('L')
    elif mag >= 3.5 and mag < 5.5:
        magTypes.append('M')
    else:
        magTypes.append('H')

#print(len(magTypes))

eqData['MagClass'] = magTypes
#print(eqData) # 1.5 - 7.8

ax.set(xlim=(-166, -66))
ax.set(ylim=(23.9, 72))

world.plot(ax = ax, legend=True, color='white', edgecolor='black', linewidth=1, alpha=.3)

#initiliaztion of model
clustering = DBSCAN(eps=1.5, min_samples=100)

#column extraction
latcol = eqData['LAT']
loncol = eqData['LON']
magcol = eqData['MAG']
depthcol = eqData['DEPTH']

eqDataDepth = pd.DataFrame({'LAT': latcol,'LON':loncol,'DEPTH':depthcol})
eqDataMag = pd.DataFrame({'LAT': latcol,'LON':loncol,'MAG':magcol})

#train model on the 2 datasets
modelDepth = clustering.fit(eqDataDepth)
modelMag = clustering.fit(eqDataMag)

#grab the labels from each model
labelsDepth = modelDepth.labels_
labelsMag = modelDepth.labels_

#print(np.unique(labelsMag))

#extract labels and map it to a viridis color map
viridis = cm.get_cmap('viridis')
colors = []
for label in labelsMag:
    if label == -1:
        colors.append([0,0,0])
    else:
        colors.append(viridis(label/10))

#plot points and color them based on chosen labels
plt.scatter(X, Y, s=4, color=colors)
plt.show()
