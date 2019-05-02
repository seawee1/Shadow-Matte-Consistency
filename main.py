import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import shadowIntersection as si
import identifyBoundary as ib
import sampling
import sigmoid
import numpy as np
import skimage
from skimage.color import rgb2gray
import spline
import copy
import os
import sys

def test_shadow_removal():
    # Load test image
    imgName = 'stone.jpg'
    img = plt.imread(imgName)
    img = np.log(skimage.img_as_float(img))
    
    # Apply artificial shadow
    C = [1.0, 1.0, 1.0]
    shadow_end = int(img.shape[1]/2)
    img_shadowed = copy.deepcopy(img)
    for i in range(0,3):
        img_shadowed[:,:shadow_end,i] -= C[i]
        
    # Plot shadowed image
    plt.figure(1)
    plt.imshow(np.exp(img_shadowed))
    
    plt.figure(2)
    patch = img_shadowed[50:100, shadow_end -20:shadow_end+21]
    plt.imshow(np.exp(patch))
    
    pen_inter = sigmoid.penumbraRegion(np.log(skimage.img_as_float(rgb2gray(np.exp(patch)))))
    A,b = spline.build_Ab(pen_inter, patch)
    spline_params = spline.solve_Ab(A,b)
    C =spline.compute_C(spline_params, patch)
    print(C)
    
    img_nonshadowed = copy.deepcopy(img_shadowed)
    for i in range(0,3):
        img_nonshadowed[:,:shadow_end,i] += C[i]
    plt.figure(3)
    plt.imshow(np.exp(img_nonshadowed)) 
 
im1_name = ''
im2_name = ''
bnd1_name = ''
bnd2_name = ''
if(len(sys.argv) == 1):
    data_path = 'Data/shadowDb/'
    image_names = os.listdir(data_path + 'img/')
    
    counter = 0
    for images in image_names:
        print('[' + str(counter) + "] " + images)
        counter += 1
        
    im1_idx = int(input('First image: '))
    im2_idx = int(input('Second image: '))
    im1_name = data_path + 'img/' + image_names[im1_idx]
    im2_name = data_path + 'img/' + image_names[im2_idx]
    bnd1_name = data_path + 'csv/' + os.path.splitext(image_names[im1_idx])[0] + '.csv'
    bnd2_name = data_path + 'csv/' + os.path.splitext(image_names[im2_idx])[0] + '.csv'
elif(len(sys.argv) == 2):
    path = str(os.getcwd()) + '/'
    im1_name = path + sys.argv[1]
    bnd1_name = os.path.splitext(im1_name)[0] + '.csv'
    im2_name = im1_name
    bnd2_name = bnd1_name
elif(len(sys.argv) == 3):
    path = str(os.getcwd()) + '/'
    im1_name = path + sys.argv[1]
    bnd1_name = os.path.splitext(im1_name)[0] + '.csv'
    im2_name = path + sys.argv[2]
    bnd2_name = os.path.splitext(im2_name)[0] + '.csv'
else:
    sys.exit()
#test_shadow_removal()
#sys.exit()

# Read images.
im1 = plt.imread(im1_name)
im2 = plt.imread(im2_name)

# Read shadow boundary information from csv file.
boundary_im1 = pd.read_csv(bnd1_name)
boundary_im2 = pd.read_csv(bnd2_name)

# Make user draw in two lines, where each line goes from the inside to the outside of a shadow.
line1 = si.intersectionLine_userInput(im1_name, boundary_im1.values[:, 0:2])
line2 = si.intersectionLine_userInput(im2_name, boundary_im2.values[:, 0:2])

# Find nearest boundary intersection point indices
idx1 = si.nearestIntersectionPoint(boundary_im1.values[:, 0:2], line1)
idx2 = si.nearestIntersectionPoint(boundary_im2.values[:, 0:2], line2)

# Identify all points connected to this boundary
bnd1 = ib.identifyConnectedBoundary(boundary_im1.values[:, 0:2], idx1)
bnd2 = ib.identifyConnectedBoundary(boundary_im2.values[:, 0:2], idx2)

# Sample the boundary
patch1 = sampling.sampleBoundary(im1, boundary_im1.values[bnd1, 0:2], boundary_im1.values[idx1, 0:2], 20, 50)
patch2 = sampling.sampleBoundary(im2, boundary_im2.values[bnd2, 0:2], boundary_im2.values[idx2, 0:2], 20, 50)

# Convert patch to flaot image
patch1 = skimage.img_as_float(patch1)
patch2 = skimage.img_as_float(patch2)

# Because of taking np.log of img, it is a problem if some pixels have values near 0
# This gets solved by adding a constant to both patches if this occurs
thresh = 1.0/256 * 3.0
if(np.any(patch1 <= thresh)) or (np.any(patch2 <= thresh)):
    print("Very dark areas detected. Adding constant to all pixels.")
    patch1 += thresh
    np.clip(patch1, None, 1.0)
    patch2 += thresh
    np.clip(patch2, None, 1.0)

# Show both patches
plt.figure()
plt.imshow(patch1)
plt.figure()
plt.imshow(patch2)

# Transfer to log domain
patch1_log = np.log(patch1)
patch2_log = np.log(patch2)

# Fit sigmoids to shadow boundary
plt.figure()
pen_inter1 = sigmoid.penumbraRegion(np.log(rgb2gray(patch1)))
plt.figure()
pen_inter2 = sigmoid.penumbraRegion(np.log(rgb2gray(patch2)))


# Determine f_s and f_n for all three color channels
A_1,b_1 = spline.build_Ab(pen_inter1, patch1_log)
spline_params_1 = spline.solve_Ab(A_1,b_1)
C_1 = spline.compute_C(spline_params_1, patch1_log)

A_2,b_2 = spline.build_Ab(pen_inter2, patch2_log)
spline_params_2 = spline.solve_Ab(A_2,b_2)
C_2 = spline.compute_C(spline_params_2, patch2_log)

print("C_1: " + str(C_1))
print("C_2: " + str(C_2))

# Detect if some forgery occured
forgery = spline.detect_forgery(C_1, C_2)
if(forgery):
    print("Forgery detected!")
else:
    print("No forgery detected!")


# Plot f_s and f_n for all channels
x = np.arange(0, patch1_log.shape[1], step=1)
y = np.arange(0, patch1_log.shape[0], step=1)
xx, yy = np.meshgrid(x, y)
fig = plt.figure()
base = 230
fig_name = ['R', 'G', 'B']
for c in range(0,3):
    z_nonshadow_1 = spline.splineSurface(xx, yy, spline_params_1[c][0:patch1_log.shape[0]*3], False)
    z_shadow_1 = spline.splineSurface(xx, yy, spline_params_1[c][patch1_log.shape[0]*3:], True)
    z_nonshadow_2 = spline.splineSurface(xx, yy, spline_params_2[c][0:patch2_log.shape[0]*3], False)
    z_shadow_2 = spline.splineSurface(xx, yy, spline_params_2[c][patch2_log.shape[0]*3:], True)
    ax = fig.add_subplot(base+(c+1), projection='3d')    
    ax.plot_surface(xx, yy, z_nonshadow_1)
    ax.plot_surface(xx, yy, patch1_log[:,:,c])
    ax.plot_surface(xx, yy, z_shadow_1)
    ax.set_title(fig_name[c] + "-Channel")
    
    ax = fig.add_subplot(base+(c+4), projection='3d')    
    ax.plot_surface(xx, yy, z_nonshadow_2)
    ax.plot_surface(xx, yy, patch2_log[:,:,c])
    ax.plot_surface(xx, yy, z_shadow_2)
    ax.set_title(fig_name[c] + "-Channel")
plt.show()