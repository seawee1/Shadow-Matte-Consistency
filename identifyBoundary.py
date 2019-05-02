import numpy as np
import bisect

indices = []

def boundaryRecursion(boundaries, idx, dist):
    curVec = boundaries[idx]
    neighbors = (boundaries[:, 0] <= curVec[0] + dist) 
    neighbors = np.logical_and(neighbors, (boundaries[:, 0] >= curVec[0] - dist))
    neighbors = np.logical_and(neighbors, (boundaries[:, 1] <= curVec[1] + dist))  
    neighbors = np.logical_and(neighbors, (boundaries[:, 1] >= curVec[1] - dist))  
    
    recursionPnts = []
    for i in np.nonzero(neighbors)[0]:
        if i not in indices:
            recursionPnts.append(i)
            bisect.insort(indices, i)
    for i in recursionPnts:
        boundaryRecursion(boundaries, i, dist)


# Method to identify a connected boundary based on a starting point, specified by idx.
# For every point, all the points in its neighborhood (x +- dist, y +- dist) get identified and added to the boundary.
# This process propagates forward via recursion.
def identifyConnectedBoundary(boundaries, idx, dist = 1.0):
    global indices
    indices = []
    boundaryRecursion(boundaries, idx, dist)
    return indices