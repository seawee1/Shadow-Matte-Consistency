import numpy as np
from sympy import Matrix, Array
from sympy.abc import x
from sympy.physics.quantum import TensorProduct
import pickle
import spline

np.set_printoptions(threshold=np.inf)

def E_d(x_shadow, x_nonshadow, pen_inter, y_dim, patch):
    ############################
    # Compute A
    ############################
    # Compute quadratic outer product via sympy
    quadratic = Matrix(3,1,[x*x, x, 1.0])
    quadratic_outer = TensorProduct(quadratic, quadratic.T)
    
    A_nonshadow = np.zeros((3, 3))
    # Non-shadow pattern
    for x_ in x_nonshadow:
        A_nonshadow += np.array(quadratic_outer.subs(x, x_)).astype(np.float64)
    
    # Shadow pattern
    A_shadow = np.zeros((3, 3))
    for x_  in x_shadow:
        A_shadow += np.array(quadratic_outer.subs(x, x_)).astype(np.float64)
    
    # Fill A along diagonal with the two patterns
    # First half non_shadow, second  half shadow pattern
    A_dim = y_dim * 3 * 2
    A = np.zeros((A_dim, A_dim))
    for i in np.arange(0, A_dim / 2, 3, dtype=np.int16):
        A[i:i+3, i:i+3] = A_nonshadow
    for i in np.arange(A_dim/2, A_dim, 3, dtype=np.int16):
        A[i:i+3, i:i+3] = A_shadow
    ###########################
    # Compute b
    ###########################
    b = np.zeros(y_dim*3*2)
    for i in range(0, y_dim):
        # Cut out nonshadow patch
        patch_nonshadow = patch[i, int(np.ceil(pen_inter[1])):]
        # Cut out shadow patch
        patch_shadow = patch[i, 0:int(np.floor(pen_inter[0]))+1]
        
        # sum_j I_{i,j} * x_i^2
        b[3*i] = np.sum(np.multiply(patch_nonshadow , np.square(x_nonshadow)))
        # sum_j I_{i,j} * x_i
        b[3*i + 1] = np.sum(np.multiply(patch_nonshadow , x_nonshadow))
        # sum_j I_{i,j}
        b[3*i + 2] = np.sum(patch_nonshadow)
        
        b[3*i + y_dim * 3] = np.sum(np.multiply(patch_shadow, np.square(x_shadow)))
        b[3*i + y_dim * 3 + 1] = np.sum(np.multiply(patch_shadow, x_shadow))
        b[3*i + y_dim * 3 + 2] = np.sum(patch_shadow)
    
    return A,b

def E_s(x_, y_dim):
    # (2 a_j)^2 
    xx_derivative = Matrix(9,1,[0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    xx_outer = TensorProduct(xx_derivative, xx_derivative.T)
    # (\frac{1}{2} (2 a_{j+1} x_i + b_{j+1} - 2 a_{j-1} x_i - b_{j-1}))^2
    xy_derivative = Matrix(9,1,[-2.0*x, -1.0, 0.0, 0.0, 0.0, 0.0, 2.0*x, 1.0, 0.0])
    xy_derivative = 0.5 * xy_derivative
    xy_outer = TensorProduct(xy_derivative, xy_derivative.T)
    # (a_{j+1} x_i^2 + b_{j+1} x_i + c_{j+1} - 2a_{j} x_i^2 - 2 b_{j} x_i - 2 c_{j} + a_{j-1} x_i^2 + b_{j-1} x_i + c_{j-1})^2
    yy_derivative = Matrix(9, 1,[x*x, x, 1.0, -2.0*x*x, -2.0 * x, -2.0, x*x, x, 1.0])
    yy_outer = TensorProduct(yy_derivative, yy_derivative.T)
    
    combined = xx_outer + 2.0 * xy_outer + yy_outer
    A_pattern = np.zeros((9,9))
    for i in x_:
        A_pattern += np.array(combined.subs(x, i)).astype(np.float64)
    
    # Fill along diagonal
    A_dim = y_dim * 3 * 2
    A = np.zeros((A_dim, A_dim))
    for i in range(0, y_dim- 2):
        idx_n = i*3
        A[idx_n:idx_n+9, idx_n:idx_n+9] += A_pattern
        idx_s = i*3 + y_dim * 3
        A[idx_s:idx_s+9, idx_s:idx_s+9] += A_pattern
    return A

def E_c(x_, y_dim):
    # Read hard coded sympy expressions
    file_Ec_sympy = open('Ec_sympy.obj', 'rb')
    combined_reduced = pickle.load(file_Ec_sympy)
    file_Ec_idx = open('Ec_idx.obj', 'rb')
    idx = pickle.load(file_Ec_idx)
    
    A_dim = y_dim * 3 * 2
    A = np.zeros((A_dim, A_dim))
    # Build A_pattern by summing up over x coordinates
    A_pattern = np.zeros(combined_reduced.shape[0])
    for i in x_:
        A_pattern += np.array(combined_reduced.subs(x, i)).astype(np.float64)
    
    # Build A by shifting pattern along diagonal
    for i in range(0, y_dim - 2):
        A[np.add(idx[0], i*3), np.add(idx[1], i*3)] += A_pattern
    return A

#################################################################################################################################
################################################ E_c hardcoder ##################################################################
#################################################################################################################################    
def E_c_hardcoder(y_dim):
    A_pattern_dim = y_dim * 3 + 9
    # (2 a_{j, n} x_i + b_{j, n} - 2 a_{j, s} x_i - b_{j, s})^2
    x_derivative = Matrix.zeros(A_pattern_dim, 1)
    x_derivative[3] = 2.0*x             # 2 a_{j, n}
    x_derivative[4] = 1.0               # b_{j, n}
    x_derivative[-6] = -2.0 * x         # - 2 a_{j, s} x_i
    x_derivative[-5] = -1.0             # - b_{j, s}
    x_outer = TensorProduct(x_derivative, x_derivative.T)
    
    # \frac{1}{4} (a_{j+1,n} x_i^2 + b_{j+1,n} x_i + c_{j+1,n} 
    # - a_{j+1,s} x_i^2 - b_{j+1,s} x_i - c_{j+1,s} - a_{j-1,n} x_i^2 
    # - b_{j-1,n} x_i - c_{j-1,n} + a_{j-1,s} x_i^2 + b_{j-1,s} x_i + c_{j-1,s})^2
    y_derivative = Matrix.zeros(A_pattern_dim, 1)
    y_derivative[6] = x*x               # a_{j+1,n} x_i^2
    y_derivative[7] = x                 # b_{j+1,n} x_i
    y_derivative[8] = 1.0               # c_{j+1,n}
    y_derivative[-3] = - x*x            # - a_{j+1,s} x_i^2
    y_derivative[-2] = - x              # - b_{j+1,s} x_i
    y_derivative[-1] = - 1.0            # - c_{j+1,s}
    y_derivative[0] = - x*x             #- a_{j-1,n} x_i^2
    y_derivative[1] = -x                #- b_{j-1,n} x_i
    y_derivative[2] = - 1.0             #- c_{j-1,n}
    y_derivative[-9] = x * x            # a_{j-1,s} x_i^2
    y_derivative[-8] = x                # b_{j-1,s} x_i
    y_derivative[-7] = 1.0              # c_{j-1,s}
    y_derivative = y_derivative * 0.5
    y_outer = TensorProduct(y_derivative, y_derivative.T)
    
    # Combine x and y derivatives
    combined = x_outer + y_outer

    # Convert to numpy array
    combined_np = np.array(combined.tolist())
    # Find indices of nonzero matrix entries
    idx = np.nonzero(combined_np)
    # Build sympy array of nonzero entries
    combined_reduced = Array(combined_np[idx[0], idx[1]])
    # Now we have an index list and an sympy expression Array, where
    # combined_reduced[i] corresponds to matrix position (idx[0,i], idx[1, i])
    
    # Save these two data structures as files
    file_Ec_sympy = open('Ec_sympy.obj', 'wb')
    pickle.dump(combined_reduced, file_Ec_sympy)
    file_Ec_idx = open('Ec_idx.obj', 'wb')
    pickle.dump(idx, file_Ec_idx)
################################################################################################################################# 
#################################################################################################################################
#################################################################################################################################

def build_Ab(pen_inter, patch):
    x = np.arange(0, patch.shape[1], step=1)
    x_shadow = np.arange(0, int(np.floor(pen_inter[0])) + 1, step=1)
    x_nonshadow = np.arange(np.ceil(pen_inter[1]), patch.shape[1], step=1)

    #E_c_hardcoder(patch.shape[0])
    #return
    
    # Compute A and b for all three color channels and return them as lists
    A_rgb = []
    b_rgb = []
    
    A_s = E_s(x, patch.shape[0])
    A_c = E_c(x, patch.shape[0])
    for i in range(0,3):
        A_d, b_d = E_d(x_shadow, x_nonshadow, pen_inter, patch.shape[0], patch[:,:,i])
        A = A_d + A_s + A_c
        b = b_d
        A_rgb.append(A)
        b_rgb.append(b)
        
    return A_rgb, b_rgb

def solve_Ab(A, b):
    params = []
    for i in range(0,3):
        params.append(np.linalg.solve(A[i], b[i]))
    return params

def compute_C(params, patch):
    x = np.arange(0, patch.shape[1], step=1)
    y = np.arange(0, patch.shape[0], step=1)
    xx, yy = np.meshgrid(x, y)
    
    C_rgb = []
    for i in range(0, 3):        
        z_shadow = spline.splineSurface(xx, yy, params[i][patch.shape[0]*3:], True)
        z_nonshadow = spline.splineSurface(xx, yy, params[i][0:patch.shape[0]*3], False)
        C_rgb.append(np.mean(z_nonshadow - z_shadow))
        
    return np.array(C_rgb)

sigma = 2.7
forgery_threshold = 1.96
def detect_forgery(C_1, C_2):
    D = np.exp(sigma) * (np.exp(-C_1) - np.exp(-C_2))
    print("D: " + str(D))
    return np.any(np.abs(D) > forgery_threshold, axis=0)

def splineSurface(x, y, params, shadow=True):
    a = params[y*3]
    b = params[y*3+1]
    c = params[y*3+2]
    return np.multiply(a, np.square(x)) + np.multiply(b, x) + c