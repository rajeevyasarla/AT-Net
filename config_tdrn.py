import numpy as np
from scipy import signal
import h5py
import random
from scipy import interpolate
from scipy.stats import norm
import copy
import pdb



def __init__():
    global kernels
    global Num_D
    global Num_k
    global Warp_mat
    k_filename ='./kernels.mat'
    kfp = h5py.File(k_filename)
    kernels = np.array(kfp['kernels'])
    kernels = kernels.transpose([0,2,1])
    Num_D = 4000
    Num_k = 39990
    Warp_mat = np.zeros((Num_D,2,256+42,256+32))



def distort_gen(W,H,Num_itr,patch_size,distortion_strength):
    # ch,W,H = I.shape
    # print(I.shape)
    U = np.zeros((W,H))
    V = np.zeros((W,H))
    x_itr = np.linspace(-patch_size-0.5,patch_size+1,2*patch_size+2)
    sigma = 16
    cdf_x = norm.cdf(x_itr,0,sigma)
    # cdf_y = copy.deepcopy(cdf_x)
    pdf_x = cdf_x[1:]-cdf_x[:-1]
    pdf_x = pdf_x/pdf_x.max()
    pdf_x = np.reshape(pdf_x,(2*patch_size+1,1))
    K = np.matmul(pdf_x,pdf_x.transpose())
    for i in range(Num_itr): 
        a = K*random.gauss(0,1)*distortion_strength
        b = K*random.gauss(0,1)*distortion_strength
        x = random.randint(patch_size+1,W-patch_size-2)
        y = random.randint(patch_size+1,H-patch_size-2)
        U[x-patch_size:x+patch_size+1,y-patch_size:y+patch_size+1] += a
        V[x-patch_size:x+patch_size+1,y-patch_size:y+patch_size+1] += b
    return U,V

def warp(I,X_new,Y_new):
    W,H = I.shape
    # print(W,H)
    X_new_1d = np.reshape(X_new,(W*H))
    Y_new_1d = np.reshape(Y_new,(W*H))
    x_f = np.floor(X_new_1d)
    x_f[x_f<0] =0
    x_f[x_f>H-1] = H-1
    y_f = np.floor(Y_new_1d)
    y_f[y_f<0] =0
    y_f[y_f>W-1] = W-1
    x_c = np.ceil(X_new_1d)
    x_c[x_c<0] =0
    x_c[x_c>H-1] = H-1
    y_c = np.ceil(Y_new_1d)
    y_c[y_c<0] =0
    y_c[y_c>W-1] = W-1
    P1 = (X_new_1d-x_f)*(Y_new_1d-y_f)
    P2 = (X_new_1d-x_f)*(Y_new_1d-y_c)
    P3 = (X_new_1d-x_c)*(Y_new_1d-y_f)
    P4 = (X_new_1d-x_c)*(Y_new_1d-y_c)
    # print(x_c.max(),y_c.max())
    
    A1 = P1* I[y_f.astype(int),x_f.astype(int)]
    A2 = P2* I[y_c.astype(int),x_f.astype(int)]
    A3 = P3* I[y_f.astype(int),x_c.astype(int)]
    A4 = P4* I[y_c.astype(int),x_c.astype(int)]
    I_new = np.reshape(A1-A2-A3+A4,(W,H))

    return I_new


def warp_gen():
  print("creating warp matrix")
  global Warp_mat
  global Num_D
  for i in range(Num_D):
    U,V = distort_gen(256+42,256+32,random.randint(0,4)*3000+1000,6,0.13)
    Warp_mat[i,0,:,:] = U
    Warp_mat[i,1,:,:] = V
    if (i+1)%int(Num_D/3) == 0:
      print("Still creating")
  print("Done creating warp matrix")
  return