from sklearn.neighbors import NearestNeighbors
from sklearn import manifold
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats 
from scipy import interpolate

# Here the acual sampling is performed
def bilinearInterpolation(slope, intercept, im, x_start, x_end, samples_x, samples_y):
    # Bilinear RGB interpolators
    r = interpolate.interp2d(np.arange(im.shape[1]), np.arange(im.shape[0]), im[:,:,0], kind='linear')
    g = interpolate.interp2d(np.arange(im.shape[1]), np.arange(im.shape[0]), im[:,:,1], kind='linear')
    b = interpolate.interp2d(np.arange(im.shape[1]), np.arange(im.shape[0]), im[:,:,2], kind='linear')
    
    i, j = samples_y -1 , 0
    patch = np.zeros((samples_y, samples_x*2 + 1, 3), 'uint8')
    for x in np.linspace(x_start, x_end, samples_y):
        y = slope*x + intercept
        # (x,y) is the current line segment location
        # create line normal vector with unit length
        dir_vec = np.array([0,0])
        if(slope == 0):
            dir_vec = np.array([0, 1])
        else:
            dir_vec = np.array([1, -(1/slope)])
        
        dir_vec = dir_vec / np.linalg.norm(dir_vec)
        for s in range(-samples_x, samples_x + 1):
            # Sample interpolated points
            pos = np.array([x,y]) + s * dir_vec
            patch[i, j, 0] = np.round(r(pos[0], pos[1]))
            patch[i, j, 1] = np.round(g(pos[0], pos[1]))
            patch[i, j, 2] = np.round(b(pos[0], pos[1]))
            j += 1
        i -= 1
        j = 0
    
    # The shadow part should be on the left side, the non-shadow part on the right.
    # Compute mean RGB vector of left and right side and flip if necessary.
    mean_left = np.mean(np.mean(patch[:, 0:samples_x], axis=0), axis=0)
    mean_right = np.mean(np.mean(patch[:, samples_x:], axis=0), axis=0)
    if(np.linalg.norm(mean_left) > np.linalg.norm(mean_right)):
        patch = np.flip(patch, 1)
    return patch
            
        
# Samples 'sample_y' boundary points around point 'point'
# For this, the method divides the boundary segment into connected groups of size 10 and fits a line to each group.
# After that, from each line 10 equidistant samples get drawn.
# Then, 'sample_x' points into positive and negative normal direction get sampled per line point.             
def sampleBoundary(im, boundary, point, samples_x, samples_y):
    # Identify connected boundary segment with size 'samples_y' via K-NN
    # This can yield some errors if two different boundaries are very near. Fixing this later.
    nbrs = NearestNeighbors(n_neighbors=samples_y, algorithm='ball_tree').fit(boundary)
    dist, ind = nbrs.kneighbors([point])
    
    # Apply LLE in order to unravel the boundary manifold.
    # Kind of an overkill but no other idea on how to do this.
    X_r, err = manifold.locally_linear_embedding(boundary[ind[0]], n_neighbors=2, n_components=1)
    X_r = np.concatenate((X_r, ind.transpose()), axis=1)
    X_r_sorted = X_r[X_r[:,0].argsort()]
    
    # Some plotting
    plt.figure()    
    plt.imshow(im)
    
    samples = np.zeros((samples_y, samples_x*2+1, 3), 'uint8')
    # Iterate over boundary segments of size 10
    for i in range(0,samples_y - 9, 10):
        boundarySegment = boundary[X_r_sorted[:,1].astype(int)[i:i+10]]
        plt.scatter(boundarySegment[:, 0], boundarySegment[:, 1], marker='.', s=5)
 
        # Fit line to boundary point coordinates
        slope, intercept, _, _, _ = stats.linregress(boundarySegment[:, 0], boundarySegment[:, 1])
        line_x1 = min(boundarySegment[:,0])
        line_x2 = max(boundarySegment[:, 0])
        line_y1 = slope*line_x1 + intercept
        line_y2 = slope*line_x2 + intercept
        
        plt.plot([line_x1, line_x2], [line_y1, line_y2], marker = '.')
        
        # Sample patch
        patch = bilinearInterpolation(slope, intercept, im, line_x1, line_x2, samples_x, 10)
        samples[i:i+10, :] = patch
    return samples
            