#%% Utils for Burger's equation

import scipy.io
import numpy as np
import tensorflow as tf
import time
from datetime import datetime
from pyDOE import lhs
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
#import myplots


# "." for Colab/VSCode, and ".." for GitHub
#repoPath = os.path.join(".", "PINNs")
# repoPath = os.path.join("..", "PINNs")
# utilsPath = os.path.join(os.pardir, "utils")
# dataPath = os.path.join(os.getcwd(), "data")
# appDataPath = os.path.join(repoPath, "appendix", "Data")

# sys.path.append(utilsPath)
# from plotting import newfig, savefig, saveResultDir

def prep_data(datafname,hp, FROGtype = 'SHG', GlobalPath = os.getcwd(), RKsteps = 1, N_u=None, N_f=None, N_n=None, q=100, ub=None, lb=None, SNR=np.inf, AmpNoise = True, N = 2**9, IsBaseModel = False):
    dataPath = os.path.join(GlobalPath, "data")
    utilsPath = os.path.join(GlobalPath,'..', "utils")
    from Make_FROG import makeFROG
    D = scipy.io.loadmat(os.path.join(dataPath,datafname))
    D2 = D
    # build a list of keys and values for each entry in the structure
    vals = D['params'][0,0] #<-- set the array you want to access. 
    keys = D['params'][0,0].dtype.descr
    sim_hp = {}
    
    for i in range(len(keys)):
        key = keys[i][0]
        val = np.squeeze(vals[key])  # squeeze is used to covert matlat (1,n) arrays into numpy (1,) arrays. 
        sim_hp[key] = np.array(val)
        
        
    data_vals = D2['data'][0,0] #<-- set the array you want to access. 
    data_keys = D2['data'][0,0].dtype.descr
    sim_data = {}
     
    for i in range(len(data_keys)):
        data_key = data_keys[i][0]
        data_val = np.squeeze(data_vals[data_key])  # squeeze is used to covert matlat (1,n) arrays into numpy (1,) arrays. 
        sim_data[data_key] = np.array(data_val) 
   # Reading external data [t is 100x1, usol is 256x100 (solution), x is 256x1]
    

    # Flatten makes [[]] into [], [:,None] makes it a column vector
    z = sim_data['z'].flatten() # T x 1
    # Get all the Points where we have "collected" data i.e. divide z into the RK steps
    Lz = z[-1]
    dz = Lz/RKsteps
    RKz = np.expand_dims(dz*(np.arange(0,RKsteps+1)),axis = 1) # +1 because the last point needs to be included at Lz as the 1
    zINX = []
    for zp in RKz:
        zINX.append(np.argmin(np.abs(zp-z)))
    
    t = sim_data['t'].flatten() # N x 1
    # Keeping the 2D data for the solution data (real() is maybe to make it float by default, in case of zeroes)
    Exact_h = (sim_data['u_sim']).T+1j*sim_data['v_sim'].T # T x N

    # Meshing x and t in 2D (256,100)
    T, Z = np.meshgrid(t,z)

    # Preparing the inputs x and t (meshed as X, T) for predictions in one single array, as X_star
    TZ_star = np.hstack((T.flatten()[:,None], Z.flatten()[:,None]))

    # Preparing the testing u_star
    h_star = Exact_h.flatten()[:,None]
                
    dt = -2*np.min(t)/N
    # idx_x = np.random.choice(Exact_h.shape[0], N_0, replace=False)
    tNN = np.arange(-hp['nt']/2,hp['nt']/2)*hp['dt']

    #factor = 1
    #tNN = t/factor
    


    
    tNN = t
    #print("np.max(tNN) BM" + str(np.max(tNN)))
    #rint("np.min(tNN) " + str(np.min(tNN)))

    #print("np.max(t) BM" + str(np.max(t)))
    #print("np.min(t) BM" + str(np.min(t)))
    #else:
    #    print("np.max(tNN) " + str(np.max(tNN)))
    #    print("np.min(tNN) " + str(np.min(tNN)))

    #    print("np.max(t) " + str(np.max(t)))
    #    print("np.min(t) " + str(np.min(t)))
        

    

    
    
    wNN = np.arange(-hp['nw']/2,hp['nw']/2)*hp['dw']
    #t_1 = np.arange(-N/2,N/2)*dt
    # Get all the RK4 steps
    hh_0 = []; hh_1 = []
    uu_0 = []; uu_1 = []
    vv_0 = []; vv_1 = []
    FROG_0 = []; FROG_1 = []
    sim_data['Clean_h0'] = []; 
    sim_data['Clean_h1'] = []
    for i in range(0,len(zINX)-1):
        # Get the input of the RK
        h0 = np.expand_dims(np.interp(tNN,t,Exact_h[:,zINX[i]]),1)
        h0 = h0.squeeze()
        sim_data['Clean_h0'].append(h0)
        h0 = np.expand_dims(h0, axis=1)
        hh_0.append(h0)
        uu_0.append(np.real(h0))
        vv_0.append(np.imag(h0))
        FROG_0.append(makeFROG(h0,h0, pad = hp['pad'], wcrop = hp['nw']))
        # Get the output of RK
        h1 = np.expand_dims(np.interp(tNN,t,Exact_h[:,zINX[i+1]]),1)
        h1 = h1.squeeze()
        sim_data['Clean_h1'].append(h1)
        h1 = np.expand_dims(h1, axis=1)
        hh_1.append(h1)
        uu_1.append(np.real(h1))
        vv_1.append(np.imag(h1))
        FROG_1.append(makeFROG(h1,h1, pad = hp['pad'], wcrop = hp['nw']))
        
    
    dz = np.asscalar(z[-1] - z[0])/(RKsteps)      
    tmp = np.float32(np.loadtxt(os.path.join(utilsPath, "IRK_weights", "Butcher_IRK%d.txt" % (q)), ndmin = 2))
    weights =  np.reshape(tmp[0:q**2+q], (q+1,q))     
    IRK_alpha = weights[0:-1,:]
    IRK_beta = weights[-1:,:] 
    tmp = np.float32(np.loadtxt(os.path.join(utilsPath, "IRK_weights", "Butcher_IRKc%d.txt" % (q)), ndmin = 2))
    
    zNN = np.zeros(q*RKsteps)
    for i in range(0,RKsteps):
        zNN[i*q:q*(i+1)] = dz*tmp[:,0]+dz*i
    
  
    NN_hp = {}
    NN_hp['u_0'] = uu_0
    NN_hp['v_0'] = vv_0
    NN_hp['FROG_0'] = FROG_0
    NN_hp['u_1'] = uu_1
    NN_hp['v_1'] = vv_1
    NN_hp['FROG_1'] = FROG_1
    NN_hp['z']  =  zNN
    
    NN_hp['t'] = tNN
    NN_hp['w'] = wNN
    NN_hp['dz']  = dz
    NN_hp['dt'] = dt
    NN_hp['q']   = q
    NN_hp['IRK_alpha'] = IRK_alpha
    NN_hp['IRK_beta']  = IRK_beta
    NN_hp['lb'] = np.array([t[0],z[0]])
    NN_hp['ub'] = np.array([t[-1],z[-1]])
    NN_hp['RKsteps'] = RKsteps
    NN_hp['zINX']  = zINX
    NN_hp['N']  = N
    
    NN_hp['Ram_res'] = np.interp(tNN,t,sim_hp['Ram_res'])
    
    # from SSFM_Solver import SSFM_GNLSE
    # nsave_z = q-1 # Size of saved array
    # nz = nsave_z*50
    # save_freq = int(nz/nsave_z)
    # u_sim, v_sim, h_sim, z2, tau = SSFM_GNLSE(np.sqrt(sim_hp['N2']), sim_hp['D']/2,IsAmpNoise = True, SNR = SNR,nz = nz, save_freq = save_freq, T0 = sim_hp['T0'],tWin = -t.min(),Lz = z.max())
    # sim_data['u_sim'] = u_sim
    # sim_data['v_sim'] = v_sim
    # sim_data['z'] = z2.squeeze()
    # sim_data['t'] = tau
    
    return sim_hp,sim_data,NN_hp

def prep_dataBM(datafname,hp, FROGtype = 'SHG', GlobalPath = os.getcwd(), RKsteps = 1, N_u=None, N_f=None, N_n=None, q=100, ub=None, lb=None, SNR=np.inf, AmpNoise = True, N = 2**9, IsBaseModel = False):
    dataPath = os.path.join(GlobalPath, "data")
    utilsPath = os.path.join(GlobalPath,'..', "utils")
    from Make_FROG import makeFROG
    D = scipy.io.loadmat(os.path.join(dataPath,datafname))
    D2 = D
    # build a list of keys and values for each entry in the structure
    vals = D['params'][0,0] #<-- set the array you want to access. 
    keys = D['params'][0,0].dtype.descr
    sim_hp = {}
    
    for i in range(len(keys)):
        key = keys[i][0]
        val = np.squeeze(vals[key])  # squeeze is used to covert matlat (1,n) arrays into numpy (1,) arrays. 
        sim_hp[key] = np.array(val)
        
        
    data_vals = D2['data'][0,0] #<-- set the array you want to access. 
    data_keys = D2['data'][0,0].dtype.descr
    sim_data = {}
     
    for i in range(len(data_keys)):
        data_key = data_keys[i][0]
        data_val = np.squeeze(data_vals[data_key])  # squeeze is used to covert matlat (1,n) arrays into numpy (1,) arrays. 
        sim_data[data_key] = np.array(data_val) 
   # Reading external data [t is 100x1, usol is 256x100 (solution), x is 256x1]
    

    # Flatten makes [[]] into [], [:,None] makes it a column vector
    z = sim_data['z'].flatten() # T x 1
    # Get all the Points where we have "collected" data i.e. divide z into the RK steps
    Lz = z[-1]
    dz = Lz/RKsteps
    RKz = np.expand_dims(dz*(np.arange(0,RKsteps+1)),axis = 1) # +1 because the last point needs to be included at Lz as the 1
    zINX = []
    for zp in RKz:
        zINX.append(np.argmin(np.abs(zp-z)))
    
    t = sim_data['t'].flatten() # N x 1
    # Keeping the 2D data for the solution data (real() is maybe to make it float by default, in case of zeroes)
    Exact_h = (sim_data['u_sim']).T+1j*sim_data['v_sim'].T # T x N

    # Meshing x and t in 2D (256,100)
    T, Z = np.meshgrid(t,z)

    # Preparing the inputs x and t (meshed as X, T) for predictions in one single array, as X_star
    TZ_star = np.hstack((T.flatten()[:,None], Z.flatten()[:,None]))

    # Preparing the testing u_star
    h_star = Exact_h.flatten()[:,None]
                
    dt = -2*np.min(t)/N
    # idx_x = np.random.choice(Exact_h.shape[0], N_0, replace=False)
    tNN = np.arange(-hp['nt']/2,hp['nt']/2)*hp['dt']

    #factor = 1
    #tNN = t/factor
    


    
    tNN = tNN
    #print("np.max(tNN) BM" + str(np.max(tNN)))
    #rint("np.min(tNN) " + str(np.min(tNN)))

    #print("np.max(t) BM" + str(np.max(t)))
    #print("np.min(t) BM" + str(np.min(t)))
    #else:
    #    print("np.max(tNN) " + str(np.max(tNN)))
    #    print("np.min(tNN) " + str(np.min(tNN)))

    #    print("np.max(t) " + str(np.max(t)))
    #    print("np.min(t) " + str(np.min(t)))
        

    

    
    
    wNN = np.arange(-hp['nw']/2,hp['nw']/2)*hp['dw']
    #t_1 = np.arange(-N/2,N/2)*dt
    # Get all the RK4 steps
    hh_0 = []; hh_1 = []
    uu_0 = []; uu_1 = []
    vv_0 = []; vv_1 = []
    FROG_0 = []; FROG_1 = []
    sim_data['Clean_h0'] = []; 
    sim_data['Clean_h1'] = []
    for i in range(0,len(zINX)-1):
        # Get the input of the RK
        #h0 = np.expand_dims(np.interp(tNN,t,Exact_h[:,zINX[i]]),1)
        h0 = np.expand_dims(Exact_h[:, zINX[i]], 1)
        h0 = h0.squeeze()
        sim_data['Clean_h0'].append(h0)
        h0 = np.expand_dims(h0, axis=1)
        hh_0.append(h0)
        uu_0.append(np.real(h0))
        vv_0.append(np.imag(h0))
        FROG_0.append(makeFROG(h0,h0, pad = hp['pad'], wcrop = hp['nw']))
        # Get the output of RK
        #h1 = np.expand_dims(np.interp(tNN,t,Exact_h[:,zINX[i+1]]),1)
        h1 = np.expand_dims(Exact_h[:, zINX[i+1]], 1)

        h1 = h1.squeeze()
        sim_data['Clean_h1'].append(h1)
        h1 = np.expand_dims(h1, axis=1)
        hh_1.append(h1)
        uu_1.append(np.real(h1))
        vv_1.append(np.imag(h1))
        FROG_1.append(makeFROG(h1,h1, pad = hp['pad'], wcrop = hp['nw']))
        
    
    dz = np.asscalar(z[-1] - z[0])/(RKsteps)      
    tmp = np.float32(np.loadtxt(os.path.join(utilsPath, "IRK_weights", "Butcher_IRK%d.txt" % (q)), ndmin = 2))
    weights =  np.reshape(tmp[0:q**2+q], (q+1,q))     
    IRK_alpha = weights[0:-1,:]
    IRK_beta = weights[-1:,:] 
    tmp = np.float32(np.loadtxt(os.path.join(utilsPath, "IRK_weights", "Butcher_IRKc%d.txt" % (q)), ndmin = 2))
    
    zNN = np.zeros(q*RKsteps)
    for i in range(0,RKsteps):
        zNN[i*q:q*(i+1)] = dz*tmp[:,0]+dz*i
    
  
    NN_hp = {}
    NN_hp['u_0'] = uu_0
    NN_hp['v_0'] = vv_0
    NN_hp['FROG_0'] = FROG_0
    NN_hp['u_1'] = uu_1
    NN_hp['v_1'] = vv_1
    NN_hp['FROG_1'] = FROG_1
    NN_hp['z']  =  zNN
    
    NN_hp['t'] = tNN
    NN_hp['w'] = wNN
    NN_hp['dz']  = dz
    NN_hp['dt'] = dt
    NN_hp['q']   = q
    NN_hp['IRK_alpha'] = IRK_alpha
    NN_hp['IRK_beta']  = IRK_beta
    NN_hp['lb'] = np.array([t[0],z[0]])
    NN_hp['ub'] = np.array([t[-1],z[-1]])
    NN_hp['RKsteps'] = RKsteps
    NN_hp['zINX']  = zINX
    NN_hp['N']  = N
    
    NN_hp['Ram_res'] = np.interp(tNN,t,sim_hp['Ram_res'])
    
    # from SSFM_Solver import SSFM_GNLSE
    # nsave_z = q-1 # Size of saved array
    # nz = nsave_z*50
    # save_freq = int(nz/nsave_z)
    # u_sim, v_sim, h_sim, z2, tau = SSFM_GNLSE(np.sqrt(sim_hp['N2']), sim_hp['D']/2,IsAmpNoise = True, SNR = SNR,nz = nz, save_freq = save_freq, T0 = sim_hp['T0'],tWin = -t.min(),Lz = z.max())
    # sim_data['u_sim'] = u_sim
    # sim_data['v_sim'] = v_sim
    # sim_data['z'] = z2.squeeze()
    # sim_data['t'] = tau
    
    return sim_hp,sim_data,NN_hp





def plot_prediction(u_0_pred, v_0_pred, u_1_pred, v_1_pred, u_pred, v_pred,sim_data,z,t,zNN,tNN,zINX,logger,pinn,lambdas_star):
    
    plt.rcParams.update({
    'font.family': 'Arial',  # Font family
    'font.size': 11,          # Font size
    'axes.labelweight': 'normal',  # Label weight
    'axes.labelcolor': 'black',   # Label color
    'axes.labelsize': 11,         # Label size
    'axes.titlesize': 11,         # Title size
    'axes.titleweight': 'bold',   # Title weight
    })
    u_sim = sim_data['u_sim'].T
    v_sim = sim_data['v_sim'].T
    h2_sim = u_sim**2+v_sim**2
    
    h2_pred = u_pred**2+v_pred**2
    
    # Create a list of 2D arrays (replace this with your data)
    list_ar = [u_pred, u_sim,
               v_pred, v_sim, 
               np.sqrt(h2_pred), np.sqrt(h2_sim)]
    list_tit = ['u^*', 'u', 
                'v^*', 'v',
                'h^{2,*}','h^2']
    
    # Create a figure with 6 subplots arranged in a 3x2 grid
    fig, axs = plt.subplots(3, 2, figsize=(9, 6), sharex=True, sharey=True)

    # Display each 2D array using imshow in subplots
    cbar = {}
    xx = (zNN,z,zNN,z,zNN,z)
    yy = (tNN,t,tNN,t,tNN,t)
    yl  = 4
    xl  = (np.min(zNN),np.max(zNN))
    for axes, ar, tit,i,zz,tt in zip(axs.flatten(), list_ar, list_tit, np.arange(0,9),xx,yy):
        
        Z,T = np.meshgrid(zz,tt)
        # im = axes.pcolor(Z,T,ar, cmap='viridis')
        
        axes.set_title(tit)
        axes.set_xlabel('z (LD)')
        axes.set_ylabel('t (t/T_0)')
        axes.set_ylim([-yl,yl])
        axes.set_xlim(xl)
    
    plt.savefig('Im_show_plot.png', dpi = 300)
    
    for i in range(0,pinn.RKsteps):
        # Create a figure with 6 subplots arranged in a 3x2 grid
        fig, axs = plt.subplots(2, 2, figsize=(4.5, 2), sharex=True, sharey=False)
        fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
        
        uL_sim = sim_data['u_sim'][zINX[i+1],:].T
        vL_sim = sim_data['v_sim'][zINX[i+1],:].T
        u0_sim = sim_data['u_sim'][zINX[i],:].T
        v0_sim = sim_data['v_sim'][zINX[i],:].T

        axes = axs.flatten()
        xl = [-5,5]
        LW1 = 1.5
        LW2 = 1.25
        alph = 1
        if i==0:
            ylL = [-1,-.5,-0.5,-.75]
            ylU = [0.5,2,0.5,0.75]
        if i==1:
            ylL = [0,-2,-1,-.5]
            ylU = [1.5,1,1,1.25]
        # axes[0].plot(t,u_0_pred.numpy(),'-b',linewidth = LW1)
        axes[0].plot(tNN,u_0_pred[i],'-r',linewidth = LW1,alpha = alph)
        axes[0].plot(t,u0_sim,'--b',linewidth = LW2)
        axes[0].set_ylim((ylL[0],ylU[0]))
        axes[0].set_yticks((0,1))
        #axes[0].set_title('u0')
        axes[2].plot(tNN,v_0_pred[i],'-r',linewidth = LW1,alpha = alph)
        axes[2].plot(t,v0_sim,'--b',linewidth = LW2)
        axes[2].set_ylim((ylL[1],ylU[1]))
        axes[2].set_yticks((-1,1))
        #axes[1].set_title('v0')
        axes[1].plot(tNN,u_1_pred[i],'-r',linewidth = LW1,alpha = alph)
        axes[1].plot(t,uL_sim,'--b',linewidth = LW2)
        axes[1].set_ylim((ylL[2],ylU[2]))
        axes[1].set_yticks((-2,0))
        #axes[2].set_title('uL')
        axes[3].plot(tNN,v_1_pred[i],'-r',linewidth = LW1,alpha = alph)
        axes[3].plot(t,vL_sim,'--b',linewidth = LW2)
        # axes[3].set_title('vL')
        axes[3].set_ylim((ylL[3],ylU[3]))
        axes[3].set_yticks((0,0.5))
        axes[3].set_xlim((xl[0],xl[-1]))
        axes[3].set_xticks((-5,5))
        
        plt.savefig('InputOutput_RKstep_'+str(i), dpi = 600)
    
    
    # Create a list of 2D arrays (replace this with your data)
    D, N, fR, Ram_res = pinn.get_params(numpy=True)
    D_star, N_star, fR_star, Ram_res_star = lambdas_star
    
    Ram_res_t      = np.concatenate((np.zeros(Ram_res.shape),Ram_res))
    Ram_res_star_t = np.concatenate((np.zeros(Ram_res_star.shape),Ram_res_star))
    def Omega_res(Response_t):
        Response_w = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(Response_t.T))).T*(-2*t.min())
        return Response_w
    Ram_res_w = Omega_res(Ram_res_t)
    Ram_res_star_w = Omega_res(Ram_res_star_t)
    w = np.pi/(-np.min(t))*np.arange(-len(t)/2,len(t)/2)
    wNN = np.pi/(-np.min(tNN))*np.arange(-len(tNN)/2,len(tNN)/2)
    
    list_ar = [Ram_res_t, np.real(Ram_res_w), np.imag(Ram_res_w)]
    list_ar_star = [Ram_res_star_t, np.real(Ram_res_star_w), np.imag(Ram_res_star_w)]
    list_tit = ['$h(t)$','$Re{H(\omega)}$','$Im{H(\omega)}$']
    xLab = ['$t/T_0$','$\omega T_0$','$\omega T_0$']
    xaxes        = [tNN,wNN,wNN]
    
    XL   = [(-2.5,5),(-20,20),(-20,20)]
    YL   =[(-1,2.5),(-0.5,1.5),(-1.5,1.5)]
    # Create a figure with 6 subplots arranged in a 3x2 grid
    fig, axs = plt.subplots(3, 1, figsize=(4.5, 6), sharex=False, sharey=False)

    # Display each 2D array using imshow in subplots
    cbar = {}
    yl  = 4
    lw = 2
    def RMSE(x1,x2):
        nx = len(x1)
        nx = np.sum(np.abs(x2)**2)
        RMSE = (np.sum(np.abs(x1.squeeze()-x2)**2)/nx)
        return RMSE
        
    for axes, ar, ar2, x,tit,xl,XLi,YLi in zip(axs.flatten(), list_ar, list_ar_star,xaxes, list_tit, xLab,XL,YL):
        axes.plot(x,ar,'-r',label = 'Prediction',linewidth = lw)
        axes.plot(x,ar2,'--b',label = 'Truth',linewidth = lw)
        print('RMSE:' + str(RMSE(ar,ar2)) )
        axes.set_ylabel(tit)
        axes.set_xlabel(xl)
        axes.set_xlim(XLi)
        axes.set_ylim(YLi)
        
    #plt.legend()
    # axs.flatten()[-1].legend('Predicted','Ground Truth')
    # plt.subplots_adjust(top = 0.1,\
    #                         bottom = 0\
    #                            )
    plt.tight_layout()
    plt.savefig('Raman_Response.png', dpi = 600)
    
    # plot_prediction_waterfall(u_pred.numpy(),v_pred.numpy(), u_sim, v_sim, z, t)
        
    return
def plot_prediction_waterfall(u_pred,v_pred, u_sim, v_sim, z, t):
    if u_pred.shape != u_sim.shape:
        try:
            # u_pred = np.reshape(u_pred, u_sim.shape)
            # v_pred = np.reshape(v_pred, u_sim.shape)
            u_pred = np.reshape(u_pred, u_sim.shape)
            v_pred = np.reshape(v_pred, u_sim.shape)
        except:
            RuntimeError('Error in plot_prediction: u_pred could not be reshaped into simulation size')
    plt.rcParams.update({
    'font.family': 'calibri',  # Font family
    'font.size': 11,          # Font size
    'axes.labelweight': 'normal',  # Label weight
    'axes.labelcolor': 'blue',   # Label color
    'axes.labelsize': 11,         # Label size
    'axes.titlesize': 11,         # Title size
    'axes.titleweight': 'bold',   # Title weight
    })
    # print(u_sim.shape)
    h2_pred = np.abs(u_pred)**2+np.abs(v_pred)**2
    h2_sim  = np.abs(u_sim )**2+np.abs(v_sim )**2
    fig = plt.figure(figsize = (5,5))
    ax = fig.add_subplot(111, projection = '3d', facecolor = 'white')
    nz = len(z)
    n  = int(nz/6)
    yl = [-4,4]
    for i in range(0, nz, n):
        tinx = ((t>=yl[0])*(t<=yl[-1]))
        yy = t[tinx]
        xx = z[int(i)]*np.ones(len(yy))
        zz = h2_pred[i,tinx]
        zz2 = h2_sim[i,tinx]
        if i == 0:
            ax.plot(xx,yy,zz2,'-b',label = 'SSFM')
            ax.plot(xx,yy,zz,'--r',label = 'MD-PINN')
        else:
            ax.plot(xx,yy,zz2,'-b')
            ax.plot(xx,yy,zz,'--r')
    
    ax.plot(z,t[np.argmax(h2_pred, axis=1)],np.zeros(len(z)),'--k',label = 'Raman Shift')
    ax.set_ylim(yl[0],yl[1])
    ax.set_box_aspect([yy.max(),yy.max()/2,yy.max()/3])
    ax.axis('on')
    ax.set_facecolor('white')
    # ax.grid(False)
    # Manually adjust the view
    ax.view_init(elev=20, azim=245)
    x_ti = z[range(0,nz,n)]
    x_ti_lb = ["{:.2f}".format(i) for i in x_ti]  # Convert positions to labels
    y_ti = [-3,0,3]
    y_ti_lb = ["{:.0f}".format(i) for i in y_ti]  # Convert positions to labels
    z_ti = [0,1.5, 3]
    z_ti_lb = ["{:.0f}".format(i) for i in z_ti]  # Convert positions to labels
    ax.set_xticks(x_ti)  # Set tick positions
    ax.set_xticklabels(x_ti_lb)  # Set tick labels
    ax.set_yticks(y_ti)  # Set tick positions
    ax.set_yticklabels(y_ti_lb)  # Set tick labels
    ax.set_zticks(z_ti)  # Set tick positions
    ax.set_zticklabels(z_ti_lb)  # Set tick labels
    ax.set_xlim(z.min(),z.max())

    # Turn on y-axis grid lines
    ax.xaxis._axinfo['grid'].update(color = [0,0,0], linestyle = '-', linewidth = 0.5, alpha =1)
    ax.yaxis._axinfo['grid'].update(color = [0,0,0], linestyle = '-', linewidth = 0.5, alpha = 1)
    ax.zaxis._axinfo['grid'].update(color = [0,0,0], linestyle = '-', linewidth = 0.5, alpha =1 )
    ax.legend()
    
    plt.savefig('Waterfall_plot.png',dpi = 300)
    
    return

def plot_error_vsz(u_pred,v_pred, u_sim, v_sim, z, t, zNN,u0_pred,v0_pred,u1_pred,v1_pred,return_ErVals = False, SaveDir = os.getcwd(),IsAmpNoise=False, SNR = np.inf, compare = False, dz = np.pi/2/5000, N = 2, D = 1, Lz = np.pi/2, lam0 = 835,T0 = 50, tWin = 20, nt = 2**9):
    # if Compare true will plot SSFM vs PINN must give the system the dz used to calculate
    plt.rcParams.update({
    'font.family': 'calibri',  # Font family
    'font.size': 11,          # Font size
    'axes.labelweight': 'normal',  # Label weight
    'axes.labelcolor': 'black',   # Label color
    'axes.labelsize': 11,         # Label size
    'axes.titlesize': 11,         # Title size
    'axes.titleweight': 'bold',   # Title weight
    })
    if not type(u_pred) == type(np.array([])):
        u_pred = u_pred.numpy()
        v_pred = v_pred.numpy()
        u0_pred = u0_pred.numpy()
        v0_pred = v0_pred.numpy()
        u1_pred = u1_pred.numpy()
        v1_pred = v1_pred.numpy()
    if compare:        
        from SSFM_Solver import SSFM_GNLSE
        save_freq = 50
        nz = int(save_freq*(u_sim.shape[1]-1))
        print(N)
        D = 1
        uu0  = u_sim[:,0] +1j*v_sim[:,0]
        z = zNN
        u_3d, v_3d, h_3d, z2, tau = SSFM_GNLSE(np.sqrt(N), D,z = zNN, customShape = uu0, IsAmpNoise = IsAmpNoise, SNR = SNR, Lz = z.max(), nz = nz, save_freq = save_freq, T0 = T0,tWin = -t.min())
        
    
    if u_pred.shape != u_sim.shape:
        try:
            # u_pred = np.reshape(u_pred, u_sim.shape)
            # v_pred = np.reshape(v_pred, u_sim.shape)
            u_pred = np.reshape(u_pred, u_sim.shape)
            v_pred = np.reshape(v_pred, u_sim.shape)
        except:
            RuntimeError('Error in plot_prediction: u_pred could not be reshaped into simulation size')
    
    u_sim = u_sim.T
    v_sim = v_sim.T
    u_pred = u_pred.T
    v_pred = v_pred.T
    def error_fx2(u_pred,v_pred,u_sim,v_sim):
        h2_pred = u_pred**2+v_pred**2
        h2_sim = u_sim**2+v_sim**2
        er_abs = np.abs(np.sqrt(h2_pred)-np.sqrt(h2_sim))
        er_L2  = np.sum((np.sqrt(h2_pred)-np.sqrt(h2_sim))**2)/(np.sum(h2_sim))
        h_pred = u_pred+1j*v_pred
        h_sim = u_sim+1j*v_sim
        er_L2 = np.sum(np.abs(h_pred - h_sim)**2)/(np.sum(np.abs(h_sim)**2))
        return er_abs, er_L2
    def error_fx1(u_pred,v_pred,u_sim,v_sim):
        h_pred = u_pred+1j*v_pred
        h_sim = u_sim+1j*v_sim
        ax = 1
        # RMSE of each dz step
        noise_z = np.sum(np.abs(h_pred-h_sim)**2,ax)
        signal_z = np.sum(np.abs(h_sim)**2,ax)
        er_z = signal_z/noise_z
        # er_z = er_z/(h2_pred.shape[1]*np.max(h2_sim,axis = 1))
        return er_z
    
    def FFT(u,v):
        h = u+1j*v
        H = np.fft.fftshift(np.fft.fft(np.fft.fftshift(h,axes=1),axis = 1),axes = 1)
        H = H/(-2*t.min())
        return H
    if  compare:
        
        h2_pred = np.abs(u_pred)**2+np.abs(v_pred)**2
        h2_sim  = np.abs(u_sim )**2+np.abs(v_sim )**2
        h2_3d   = np.abs(u_3d )**2+np.abs(v_3d )**2
        
        H_pred = FFT(u_pred,v_pred)
        U_pred = np.real(H_pred)
        V_pred = np.imag(H_pred)
        H_sim = FFT(u_sim,v_sim)
        U_sim = np.real(H_sim)
        V_sim = np.imag(H_sim)
        H_3d = FFT(u_3d,v_3d)
        U_3d = np.real(H_3d)
        V_3d = np.imag(H_3d)
        
        H2_pred = np.abs(FFT(u_pred,v_pred))**2
        H2_sim   = np.abs(FFT(u_sim,v_sim))**2
        H2_3d    = np.abs(FFT(u_3d,v_3d))**2
        
        def PLOT_ER(name,u_predt,v_predt,u_simt,v_simt, u_3dt,v_3dt,y, YL = (-4,4),YLab = '$t/T_0$',cMap = 'viridis'):
            er_PINN_z = error_fx1(u_predt,v_predt,u_simt,v_simt)
            er_PINN_abs,er_PINN_L2 = error_fx2(u_predt,v_predt,u_simt,v_simt)
            er_SSFM_z = error_fx1(u_3dt,v_3dt,u_simt,v_simt)
            er_SSFM_abs,er_SSFM_L2 = error_fx2(u_3dt,v_3dt,u_simt,v_simt)
            
            GLOBALER = dz**3*np.ones(z.shape)
            fig, axes = plt.subplots(1,1, figsize=(4.0, 2), sharex=True, sharey=False)
            
            print(name+ ' L2 Norm Error: \n PINN: ' + str(er_PINN_L2) + '\n SSFM: ' +str(er_SSFM_L2))
            # Error Graph vs z
            axes.plot(z,er_PINN_z,'-r')
            axes.plot(z,er_SSFM_z,'-b')
            # plt.legend(('PINN','SSFM'))
            # axes.plot(z, GLOBALER)
            axes.set_yscale('linear')
            axes.set_xlim((z.min(),z.max()))
            axes.grid(visible=True,which = 'both')
            axes.set_ylabel('ER RMSE')
            axes.set_xlabel('$z/L_D$')
            plt.tight_layout()
            
            # plt.savefig('RMSE_Error.png',dpi = 600)
            h2_3d = u_3dt**2+v_3dt**2
            h2_pred = u_predt**2+v_predt**2
            fig, axes = plt.subplots(3,2, figsize=(9, 6), sharex=True, sharey=False)
            axes= axes.flatten()
            ars = (er_SSFM_z,er_PINN_z,er_SSFM_abs, er_PINN_abs, np.sqrt(u_3dt**2+v_3dt**2), np.sqrt(u_predt**2 + v_predt**2))
            titles = ('ER(z) SSFM','ER(z) PINN,','|Er| SSFM','|Er| PINN', '$h_{SSFM}$','$h_{PINN}$')
            xx = (z,z,z,z,z,z)
            yy = (y,y,y,y,y,y)
            VMAX = np.max((np.max(er_SSFM_abs),np.max(er_PINN_abs)))
            VMIN = np.min((np.min(er_SSFM_abs),np.min(er_PINN_abs)))*0
            VMAX2 = np.max((np.max(np.sqrt(h2_3d)),np.max(np.sqrt(h2_pred))))
            VMIN2 = np.min((np.min(np.sqrt(h2_3d)),np.min(np.sqrt(h2_pred))))
            print(VMAX2)
            # YL = (-y.max()/3,y.max()/3)
            YL2 = (np.min((ars[0],ars[1])),np.max((ars[0],ars[1])))
            # YL2 = (10**-6,5*10**-3)
            plt.suptitle(name)
            for ax,ar,tit,x,y,i in zip(axes, ars,titles,xx,yy,range(0,len(ars))):

                if i==0 or i==1:
                    ax.plot(x,ars[0],'-b')
                    ax.plot(x,ars[1],'-r')
                    if i == 0:
                        ax.set_yscale('linear')
                    if i == 1:
                        ax.set_yscale('log')
                    ax.set_xlim((z.min(),z.max()))
                    ax.grid(visible=True,which = 'both')
                    # axes.set_ylabel('ER RMSE')
                    ax.set_xlabel('$z/L_D$')
                    ax.set_title(tit)
                    ax.set_ylim(YL2)
                elif i ==2 or i==3:
                    INX1 = np.argmin(np.abs(y- YL[0]))
                    INX2 = np.argmin(np.abs(y- YL[1]))
                    # print(INX1)
                    ar = ar[0:-1,INX1:INX2]
                    im = ax.imshow(ar.T, extent=(z.min(), z.max(), YL[0], YL[1]), origin='lower', cmap='inferno',aspect = 'auto',vmin=VMIN, vmax=VMAX)
                    ax.set_ylim(YL[0],YL[1])
                    ax.set_title(tit)
                    ax.set_xlabel('$z/L_D$')
                    ax.set_ylabel(YLab)
                else:
                    INX1 = np.argmin(np.abs(y- YL[0]))
                    INX2 = np.argmin(np.abs(y- YL[1]))
                    
                    ar = ar[0:-1,INX1:INX2]
                    im = ax.imshow(ar.T, extent=(z.min(), z.max(), YL[0], YL[1]), origin='lower', cmap=cMap,aspect = 'auto',vmin=VMIN2, vmax=VMAX2)
                    ax.set_ylim(YL)
                    ax.set_title(tit)
                    ax.set_xlabel('$z/L_D$')
                    ax.set_ylabel(YLab)
                    
                    
                if i == 3:
                    cb_ax = fig.add_axes([.91,.391,.04,.445/2])
                    fig.colorbar(im,cax = cb_ax)    
                if i == 5:
                    cb_ax = fig.add_axes([.91,.124,.04,.445/2])
                    fig.colorbar(im,cax = cb_ax) 
                if i == len(ars)-1:
                    return er_PINN_z, er_PINN_L2, er_SSFM_z, er_SSFM_L2
                    #  axes.set_title('Total L2 Norm Error: \n '+ str(er_L2))
                    
        ert_PINN_z, ert_PINN_L2, ert_SSFM_z, ert_SSFM_L2 = PLOT_ER('Temporal Prediction',u_pred,v_pred,u_sim,v_sim, u_3d,v_3d,t)
        plt.savefig(os.path.join(SaveDir,'Temporal_Prediciton_Comparison'), dpi = 600)
        
        w = np.arange(-int(len(t)/2),int(len(t)/2))*np.pi/(2*t.max())
        erw_PINN_z, erw_PINN_L2, erw_SSFM_z, erw_SSFM_L2 = PLOT_ER('Frequency Prediction',U_pred,V_pred,U_sim,V_sim, U_3d,V_3d,w, YL = (-7,7),YLab = '$\omega T_0$', cMap = 'plasma')
        plt.savefig(os.path.join(SaveDir,'Frequency_Prediciton_Comparison'), dpi = 600)
        
        def plot_zINX(zinx = 0, zinxLabel = '0', name ='Input', AmpAndPhase =True):
            zinx = int(zinx)
            fig,ax = plt.subplots(2,2,sharex= False,sharey = False, figsize = (4.5,4))
            ax = ax.flatten()
            xx = (t,t,w,w)
            if AmpAndPhase:
                yy  = (u_pred+1j*v_pred, u_3d+1j*v_3d, FFT(u_pred,v_pred),FFT(u_3d,v_3d))
                yy2 = (u_sim+1j*v_sim, u_sim+1j*v_sim, FFT(u_sim,v_sim),FFT(u_sim,v_sim))
            else:
                yy  = (np.abs(u_pred+1j*v_pred)**2, np.abs(u_3d+1j*v_3d)**2, np.abs(FFT(u_pred,v_pred))**2,np.abs(FFT(u_3d,v_3d))**2)
                yy2 = (np.abs(u_sim+1j*v_sim)**2, np.abs(u_sim+1j*v_sim)**2, np.abs(FFT(u_sim,v_sim))**2,np.abs(FFT(u_sim,v_sim))**2)
            plt.suptitle(name)
            
            titles = ['$h_{PINN}$','$h_{SSFM}$','$H_{PINN}$','$H_{SSFM}$']
            xlabels = ['$t/T_0$','$t/T_0$','$\omega T_0$','$\omega T_0$']
            colors1 = ['r','b','r','b']
            colors2 = ['tab:orange','tab:cyan','tab:orange','tab:cyan']
            leg = False
            legName = ('$u_{'+zinxLabel+'}$','$v_{'+zinxLabel+'}$')
            
            for i,axes,x,y,y2, tit,xlab,colors_1,colors_2 in zip(range(0,len(ax)),ax,xx,yy,yy2, titles,xlabels,colors1,colors2):
                axes.plot(x,np.real(y2[zinx,:]),'--k',label = None)
                if AmpAndPhase: axes.plot(x,np.imag(y2[zinx,:]),'--k',label = None)
                axes.plot(x,np.real(y[zinx,:]),'-',color = colors_1, alpha = 0.75,label = legName[0])
                if AmpAndPhase: axes.plot(x,np.imag(y[zinx,:]),'-',color = colors_2, alpha = 0.75,label = legName[1])
                axes.set_xlabel(xlab)
                axes.set_title(tit)
                axes.set_xlim(x.min()/3,x.max()/3)
                if (i ==2 or i ==3) and leg == True:
                    axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                
            plt.tight_layout()
        plot_zINX(zinx = 0,AmpAndPhase =False,name = 'Input (Intensity)')
        plt.savefig(os.path.join(SaveDir,'Input_Intensity_Comparison.png'), dpi = 600)
        plot_zINX(zinx = 0,AmpAndPhase =True,name = 'Input (Magnitude and Phase)')
        plt.savefig(os.path.join(SaveDir,'Input_Mag_and_Phase_Comparison.png'), dpi = 600)
        plot_zINX(zinx = len(z)-1,AmpAndPhase =False,name = 'Output (Intensity)',zinxLabel = 'L_z')
        plt.savefig(os.path.join(SaveDir,'Output_Intensity_Comparison.png'), dpi = 600)
        plot_zINX(zinx = len(z)-1,AmpAndPhase =True,name = 'Output (Magnitude and Phase)',zinxLabel = 'L_z')
        plt.savefig(os.path.join(SaveDir,'Output_Mag_and_Phase_Comparison.png'), dpi = 600)
        
        plot_zINX(zinx = (len(z)-1)/2,AmpAndPhase =False,name = 'Midpoint (Intensity)',zinxLabel = 'L_z/2')
        plt.savefig(os.path.join(SaveDir,'Midpoint_Intensity_Comparison.png'), dpi = 600)
        plot_zINX(zinx = (len(z)-1)/2,AmpAndPhase =True,name = 'Midpoint (Magnitude and Phase)',zinxLabel = 'L_z/2')
        plt.savefig(os.path.join(SaveDir,'Midpoint_Mag_and_Phase_Comparison.png'), dpi = 600)
        
        def plot_input(ui_pred,vi_pred, zinx = 0, zinxLabel = '0', name ='Input', AmpAndPhase =True):
            zinx = int(zinx)
            fig,ax = plt.subplots(2,2,sharex= False,sharey = False, figsize = (4.5,4))
            ax = ax.flatten()
            xx = (t,w,t,w)
            def FFT2(u,v):
                h = u+1j*v
                H = np.fft.fftshift(np.fft.fft(np.fft.fftshift(h,axes=0),axis = 0),axes = 0)
                H = H/(-2*t.min())
                return H
            ui_pred = np.mean(ui_pred,axis = 1)
            vi_pred = np.mean(vi_pred,axis = 1)
            if AmpAndPhase:
                yy  = (ui_pred+1j*vi_pred, FFT2(ui_pred,vi_pred), np.sqrt(ui_pred**2+vi_pred**2), np.abs(FFT2(ui_pred,vi_pred)))
                yy2 = (u_sim+1j*v_sim,   FFT(u_sim,v_sim),   np.sqrt(u_sim**2+v_sim**2),   np.abs(FFT(u_sim,v_sim)))
                yy3 = (u_3d+1j*v_3d,     FFT(u_3d,v_3d),     np.sqrt(u_3d**2 +v_3d**2),    np.abs(FFT(u_3d,v_3d)))
            else:
                yy  = (np.abs(u_pred+1j*v_pred)**2, np.abs(u_3d+1j*v_3d)**2, np.abs(FFT(u_pred,v_pred))**2,np.abs(FFT(u_3d,v_3d))**2)
                yy2 = (np.abs(u_sim+1j*v_sim)**2, np.abs(u_sim+1j*v_sim)**2, np.abs(FFT(u_sim,v_sim))**2,np.abs(FFT(u_sim,v_sim))**2)
            plt.suptitle(name)
            
            titles = ['$uv_{PINN}$','$UV_{SSFM}$','$h_{PINN}$','$H_{SSFM}$']
            xlabels = ['$t/T_0$','$\omega T_0$','$t/T_0$','$\omega T_0$']
            colors1a = ['r','r','r','r']
            colors1b = ['tab:orange','tab:orange','tab:orange','tab:orange']
            colors3a = ['b','b','b','b']
            colors3b = ['tab:cyan','tab:cyan','tab:cyan','tab:cyan']
            leg = False
            legName = ('$u_{'+zinxLabel+'}$','$v_{'+zinxLabel+'}$','$u_{Label}$','$v_{Label}$')
            ms = 3
            for i,axes,x,y,y2, y3, tit,xlab,colors_1a,colors_1b,colors_3a,colors_3b in zip(range(0,len(ax)),ax,xx,yy,yy2,yy3, titles,xlabels,colors1a,colors1b,colors3a,colors3b):
                if i == 0 or i == 1:
                    axes.plot(x,np.real(y3[zinx,:]),'.', markersize =ms, alpha = 0.5, color = colors_3a,label = legName[2])
                    axes.plot(x,np.imag(y3[zinx,:]),'.', markersize =ms, alpha = 0.5, color = colors_3b,label = legName[3])
                    axes.plot(x,np.real(y),'-',color = colors_1a, alpha = 0.75,label = legName[0])
                    axes.plot(x,np.imag(y),'-',color = colors_1b, alpha = 0.75,label = legName[1])
                    axes.plot(x,np.real(y2[zinx,:]),'--k',label = None,alpha = 1)
                    axes.plot(x,np.imag(y2[zinx,:]),'--k',label = None,alpha = 1)
                axes.set_xlabel(xlab)
                axes.set_title(tit)
                axes.set_xlim(x.min()/3,x.max()/3)
                if i==2 or i==3:
                    axes.plot(x,(y3[zinx,:]),'.', markersize =ms,alpha = 0.5, color = colors_3a,label = None)
                    axes.plot(x,(y),'-',color = colors_1a, alpha = 0.75,label = legName[0])
                    axes.plot(x,(y2[zinx,:]),'--k',label = None)
                    
                axes.set_xlabel(xlab)
                axes.set_title(tit)
                if i == 0 or i == 2:
                    axes.set_xlim(x.min()/3,x.max()/3)
                if i == 1 or i == 3:
                    axes.set_xlim(x.min()/6,x.max()/6)
                if (i ==0 or i ==3) and leg == True:
                    axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                
            plt.tight_layout()
            
        
        plot_input(u0_pred,v0_pred,zinx =0 ,zinxLabel ='0',name ='input')
        plt.savefig(os.path.join(SaveDir,'Prediction_Labeled_data_L0'), dpi = 600)
        plot_input(u1_pred,v1_pred,zinx = -1,zinxLabel ='L_z',name ='Output')
        plt.savefig(os.path.join(SaveDir,'Prediction_Labeled_data_Lz'), dpi = 600)
        if return_ErVals:
            Error = {}
            Error['RMSE_z_t_PINN'] = ert_PINN_z
            Error['RMSE_z_w_PINN'] = erw_PINN_z
            Error['L2_t_PINN'] = ert_PINN_L2
            Error['L2_w_PINN'] = erw_PINN_L2
            Error['RMSE_z_t_SSFM'] = ert_SSFM_z
            Error['RMSE_z_w_SSFM'] = erw_SSFM_z
            Error['L2_t_SSFM'] = ert_SSFM_L2
            Error['L2_w_SSFM'] = erw_SSFM_L2
            return Error
    else: # Plot error compared to noiseless machine prediction
        h2_pred = np.abs(u_pred)**2+np.abs(v_pred)**2
        h2_sim  = np.abs(u_sim )**2+np.abs(v_sim )**2
        
        er_z = error_fx1(u_pred,v_pred,u_sim,v_sim)
        er_abs,er_L2 = error_fx2(u_pred,v_pred,u_sim,v_sim)
        
        H_pred = FFT(u_pred,v_pred)
        U_pred = np.real(H_pred)
        V_pred = np.imag(H_pred)
        H_sim = FFT(u_sim,v_sim)
        U_sim = np.real(H_sim)
        V_sim = np.imag(H_sim)
        
        H2_pred = np.abs(FFT(u_pred,v_pred))**2
        H2_sim   = np.abs(FFT(u_sim,v_sim))**2
        
        er_zw = error_fx1(U_pred,V_pred,U_sim,V_sim)
        er_absw, er_L2w = error_fx2(U_pred,V_pred,U_sim,V_sim)
        w = np.arange(-int(len(t)/2),int(len(t)/2))*np.pi/(t.max())
        
        xx = (z,z,z,z,z,z)
        yy = (t,w,t,t,t,w)
        ars = (np.sqrt(h2_pred),np.sqrt(H2_pred), np.sqrt(h2_pred),np.sqrt(H2_pred),   er_abs,er_absw,)
        ars2 = ((),(),np.sqrt(h2_sim),np.sqrt(H2_sim))
        fig, axes = plt.subplots(3,2, figsize=(9, 6), sharex=False, sharey=False)
        axes = axes.flatten()
        YLw = (-7,7)
        YLt = (-4,4)
        titles = ('','','','','','')
        cMapt = 'viridis'
        cMapw = 'plasma'
        # plt.suptitle('PINN Prediction')
        print('L2 Norm Error: \n t: ' + str(er_L2),' \n w: ' + str(er_L2w) )
        for ax,ar,ar2, tit,x,y,i in zip(axes, ars,ars2, titles,xx,yy,range(0,len(ars))):
            if i ==0 or i == 1:
                if i ==0: 
                    YL = YLt
                    cMap = cMapt
                    YLab = '$t/T_0$'
                    # fig, axes = plt.subplots(3,2, figsize=(9, 6), sharex=False, sharey=False)
                    #plt.subplots_adjust(bottom = 0.1)
                    er_z_ax = fig.add_axes([0.125,.9,0.3525,.08])
                    er_z_ax.plot(z,er_z,'-r')
                    er_z_ax.set_xticklabels('')
                if i == 1: 
                    YL = YLw
                    cMap = cMapw
                    YLab = '$\omega T_0$'
                    er_z_ax = fig.add_axes([0.125 +0.3525 +0.07 ,.9,0.3525,.08])
                    er_z_ax.plot(z,er_zw,'-r')
                    er_z_ax.set_xticklabels('')
                er_z_ax.set_yscale('log')
                er_z_ax.set_xlim((z.min(),z.max()))
                er_z_ax.grid(visible=True,which = 'both')
                VMIN = np.min(ar)
                VMAX = np.max(ar)
                INX1 = np.argmin(np.abs(y- YL[0]))
                INX2 = np.argmin(np.abs(y- YL[1]))
                ar1 = ar
                ar1 = ar1[0:-1,INX1:INX2]
                
                im = ax.imshow(ar1.T, extent=(z.min(), z.max(), YL[0], YL[1]), origin='lower', cmap=cMap,aspect = 'auto',vmin=VMIN, vmax=VMAX)
                ax.set_ylim(YL)
                ax.set_title(tit)
                ax.set_xlabel('$z/L_D$')
                ax.set_ylabel(YLab)
                
                
            if i == 2 or i == 3:
                if i == 2: 
                    yl = YLt
                    ax.plot(x,1/er_z,'-r')
                    ax.set_yscale('log')
                    ax.set_xlim((z.min(),z.max()))
                    ax.grid(visible=True,which = 'both')
                    ax.set_ylim((np.min(1/er_z)/2,np.max(1/er_z)*2))
                if i ==3:
                    yl = YLw
                nz = len(z)
                n  = int(nz/6)
                fig2, ax3 = plt.subplots(1,1, figsize=(6, 4), sharex=False, sharey=False)
                ax3.remove()
                ax2 = fig2.add_subplot(111,projection = '3d')
                for q in range(0, nz, n):
                    tinx = ((y>=yl[0]*2)*(y<=yl[-1]*2))
                    yyy = y[tinx]
                    xxx = z[int(q)]*np.ones(len(yyy))
                    zzz = ar[q,tinx]
                    zzz2 = ar2[q,tinx]
                    if q == 0:
                        ax2.plot(xxx,yyy,zzz2,'-b',label = 'SSFM')
                        ax2.plot(xxx,yyy,zzz,'--r',label = 'FD-PINN')
                    else:
                        ax2.plot(xxx,yyy,zzz2,'-b')
                        ax2.plot(xxx,yyy,zzz,'--r')
                if i ==2:
                    ax2.plot(z,y[np.argmax(ar, axis=1)],np.zeros(len(z)),'--k',label = 'Raman Shift')
                
                
                ax2.set_ylim(yl[0]*2,yl[1]*2)
                ax2.set_box_aspect([3,2,1])
                ax2.view_init(elev=20, azim=245)
                # ax2.autoscale_view(tight = True)
                ax2.set_yticks((yl[0],yl[1]))
                ax2.set_zticks((round(np.max(ar),1)/2, round(np.max(ar),1)))
                # ax2.set_box_aspect([yyy.max()*4,yyy.max()/2,yyy.max()/3])
                # ax2.set_position(pos= [0.125,.9,0.3525,.08])
                ax2.axis('on')
                ax2.set_facecolor('white')
                # ax.grid(False)
                # Manually adjust the view
                if i == 2:
                    fig2.savefig(os.path.join(SaveDir,'Temporal_waterfall'), dpi = 600)
                if i == 3:
                    plt.savefig(os.path.join(SaveDir,'Frequency_waterfall'), dpi = 600)
                
            if i == 3 or i == 5:
                if i == 3: 
                    ar = er_abs
                    YL = YLt
                    cMap = 'inferno'
                    YLab = '$t/T_0$'
                if i == 5: 
                    YL = YLw
                    cMap = 'inferno'
                    YLab = '$\omega T_0$'
                VMIN = np.min(ar)
                VMAX = np.max(ar)
                INX1 = np.argmin(np.abs(y- YL[0]))
                INX2 = np.argmin(np.abs(y- YL[1]))
                ar1 = ar
                ar1 = ar1[0:-1,INX1:INX2]
                
                im = ax.imshow(ar1.T, extent=(z.min(), z.max(), YL[0], YL[1]), origin='lower', cmap=cMap,aspect = 'auto',vmin=VMIN, vmax=VMAX)
                ax.set_ylim(YL)
                ax.set_title(tit)
                ax.set_xlabel('$z/L_D$')
                ax.set_ylabel(YLab)  
            
            if i == 1:
                cb_ax = fig.add_axes([.91,.658,.04,.445/2])
                fig.colorbar(im,cax = cb_ax)
            if i == 3:
                cb_ax = fig.add_axes([.91,.391,.04,.445/2])
                fig.colorbar(im,cax = cb_ax)    
          
            if i == 5:
                cb_ax = fig.add_axes([.91,.124,.04,.445/2])
                fig.colorbar(im,cax = cb_ax)
                fig.savefig(os.path.join(SaveDir,'Prediction_Only'), dpi = 600)
                
            if i ==len(ars)-1:
                if return_ErVals:
                    Error = {}
                    Error['L2_w_PINN'] = er_L2w
                    Error['L2_t_PINN'] = er_L2
                    Error['RMSE_z_w_PINN'] = er_zw
                    Error['RMSE_z_t_PINN'] = er_z
                    return Error
            fig.savefig(os.path.join(SaveDir,'Prediction'),dpi = 600)
            
def plot_params_fit(pinn,lambdas_star, SaveDir = os.getcwd()):
    fig = plt.figure(figsize = (3,2))
    epochs = pinn.logger.epoch_ax
    D_star, N_star, fR_star, fb_star, tau_1_star, tau_2_star, tau_b_star = lambdas_star
    if pinn.D.trainable: plt.plot(epochs,pinn.logger.Vars_log['D']-D_star,label = '$D$',color = 'r')
    if pinn.N.trainable: plt.plot(epochs,pinn.logger.Vars_log['N2']-N_star,label = '$N^2$',color = 'b')
    if pinn.fR.trainable: plt.plot(epochs,pinn.logger.Vars_log['fR']-fR_star,label = '$f_R$',color = 'y')
    if pinn.fb.trainable: plt.plot(epochs,pinn.logger.Vars_log['fb']-fb_star,label = '$f_b$',color = 'tab:cyan')
    if pinn.tau_1.trainable: plt.plot(epochs,pinn.logger.Vars_log['tau_1']-tau_1_star,label = ' $\\tau_1$',color = 'tab:purple')
    if pinn.tau_2.trainable: plt.plot(epochs,pinn.logger.Vars_log['tau_2']-tau_2_star,label = ' $\\tau_2$',color = 'tab:orange')
    if pinn.tau_b.trainable: plt.plot(epochs,pinn.logger.Vars_log['tau_b']-tau_b_star,label = ' $\\tau_b$',color = 'g')
    plt.legend()
    fig.savefig(os.path.join(SaveDir,'Training_Params'),dpi = 600)
    
    
def plot_Ram_response(pinn,Figure_name = 'RamanResponsePlot',scale = 1):
    error, preds, truth = pinn.logger.error_fn()
    wmin = -30/scale; wmax =  30/scale
    tmin = -1.5*scale; tmax =  1.5*scale
    # Plotting the Raman Response
    dt = pinn.t[2]-pinn.t[1]
    w = np.arange(int(-len(preds[1])/2),int(len(preds[1])/2))*np.pi/(dt*len(preds[1]))
    fig,axes = myplots.myfig(options = '3Vert',ratio = 0.6)
    myplots.myplot((pinn.t,pinn.t), (truth[0],preds[0]), \
                   ax = axes[0], xlabel = '$t/T_0$', ylabel = '$r(t)$',\
                       xticks = (-1*scale,0,1*scale),yticks = (0,6/scale),title = '',xl = (tmin,tmax),yl = (-1*3/scale,2.5*3/scale))
    myplots.myplot((w,w), (truth[1],preds[1]), ax = axes[1], xlabel = '$\omega T_0$', ylabel = '$\~R(\omega)$',\
                   xticks = (int(-20/scale),0,int(20/scale)),yticks = (-1,1),title = '',xl = (wmin,wmax),yl = (-1.5,1.5))
    myplots.myplot((w,w), (truth[2],preds[2]), ax = axes[2], xlabel = '$\omega T_0$', ylabel = '$\~R(\omega)$',\
                   xticks = (int(-20/scale),0,int(20/scale)),yticks = (0,1),title = '',xl = (wmin,wmax), yl = (-0.5,1.5))
    myplots.savemyfig(Figure_name)
    return


def plot_3d(pinn,\
               XL  = [(-5,5),(-7,7),(-5,5),(-7,7)],\
                YL  = [(-2,0.5),(-2,1),(-1.6,1.5),(-1.9,1.9)],\
                Yticks  = [(-2,0.5),(-1,0,1),(-1,0,1),(-1,0,1)],\
                Xticks  = [(-5,0,5),(-5,0,5),(-5,0,5),(-5,0,5)],\
                    Figure_name = '3d_plot'):
        u0p, v0p, u1p, v1p, Up, Vp,\
             z,t = pinn.get_predict(numpy = True) 
        dt = t[2]-t[1]
        u0s,v0s,u1s,v1s,ts = pinn.training_data
        u0i,v0i,u1i,v1i = pinn.PerfectData
        fig,axes = myplots.myfig(options = '4Square',ratio = 0.7)
        U0s,V0s = pinn.fft(u0s[0],v0s[0])
        U0p,V0p = pinn.fft(Up,Vp)
        U0i,V0i = pinn.fft(u0i[0],v0i[0])
        U1s,V1s = pinn.fft(u1s[0],v1s[0])
        U1p,V1p = pinn.fft(np.average(u1p[0],axis = 1),np.average(v1p[0],axis = 1))
        U1i,V1i = pinn.fft(u1i[0],v1i[0])
        RES = [u0s[0],U0s,u1s[0],U1s]
        IMS = [v0s[0],V0s,v1s[0],V1s]
        REP = [u0p[0],U0p,u1p[0],U1p]
        IMP = [v0p[0],V0p,v1p[0],V1p]
        REI = [u0i[0],U0i,u1i[0],U1i]
        IMI = [v0i[0],V0i,v1i[0],V1i]
        w = np.arange(int(-len(U0s)/2),int(len(V0s)/2))*np.pi/(dt*len(V0s))
        fig,axes = myplots.myfig(options = '2Vert',ratio = 0.7)
        myplots.myimshow(z,t,np.abs(Up)**2+np.abs(Vp)**2)
        # Time domain
        axes[0].imshow(z,t,np.abs(Up)**2+np.abs(Vp)**2)
        
        # frequency domain
        axes[1].imshow(np.abs(U0p)**2+np.abs(V0p)**2)
        
def plot_input(pinn,\
               XL  = [(-5,5),(-5,5),(-7,7),(-7,7)],\
                YL  = [(-2,0.5),(-1.6,1.5),(-2,1),(-1.9,1.9)],\
                Yticks  = [(-2,0.5),(-1,0,1),(-1,0,1),(-1,0,1)],\
                Xticks  = [(-5,0,5),(-5,0,5),(-7,0,7),(-7,0,7)],\
                    Figure_name = 'InputPlot'):
    # Get Error From Prediction
   
    u0p, v0p, u1p, v1p, Up, Vp,\
         z,t = pinn.get_predict(numpy = True) 
    dt = t[2]-t[1]
    u0s,v0s,u1s,v1s,ts = pinn.training_data
    u0i,v0i,u1i,v1i = pinn.PerfectData
    fig,axes = myplots.myfig(options = '4Square',ratio = 0.7)
    U0s,V0s = pinn.fft(u0s[0],v0s[0])
    U0p,V0p = pinn.fft(np.average(u0p[0],axis = 1),np.average(v0p[0],axis = 1))
    U0i,V0i = pinn.fft(u0i[0],v0i[0])
    U1s,V1s = pinn.fft(u1s[0],v1s[0])
    U1p,V1p = pinn.fft(np.average(u1p[0],axis = 1),np.average(v1p[0],axis = 1))
    U1i,V1i = pinn.fft(u1i[0],v1i[0])
    RES = [u0s[0],u1s[0],U0s,U1s]
    IMS = [v0s[0],v1s[0],V0s,V1s]
    REP = [u0p[0],u1p[0],U0p,U1p]
    IMP = [v0p[0],v1p[0],V0p,V1p]
    REI = [u0i[0],u1i[0],U0i,U1i]
    IMI = [v0i[0],v1i[0],V0i,V1i]
    w = np.arange(int(-len(U0s)/2),int(len(V0s)/2))*np.pi/(dt*len(V0s))
    xx  = (t,t,w,w)
    XLAB = ['$t/T_0$','$t/T_0$','$\omega T_0$','$\omega T_0$']
    for x,res,ims,rep,imp,rei,imi,axs,xl,yl,xt,yt,xlab in zip(xx,RES,IMS,REP,IMP,REI,IMI,axes,XL,YL,Xticks,Yticks,XLAB):
        myplots.myplot((x,x),(res,ims),ax = axs,xlabel = '$t/T_0$', ylabel ='', \
                        color = ('b','cyan'),linestyle = ('-','-'), \
                            marker= '', alpha = 0.2,linewidth = 1.5)
        myplots.myplot((x,x),(res,ims),ax = axs,xlabel = 't0', ylabel ='',\
                        color = ('b','cyan'),linestyle = ('',''), \
                            marker= '.',markersize = 2, alpha = 0.4,linewidth = 1.5)
        # fig,axes = myplots.myfig(options = '4Square',ratio = 0.7)
        if len(rep.shape)>1:
            myplots.myplot((x,x),(np.average(rep,axis = 1),\
                                  np.average(imp,axis = 1)),ax = axs,xlabel = 't0',ylabel ='', \
                            color = ('r','orange'),linestyle = ('-','-'), \
                                marker= None,markersize = 0.01, alpha = 1,linewidth = 1.5)
        else:
            myplots.myplot((x,x),(rep,imp),ax = axs,xlabel = 't0', \
                            color = ('r','orange'),linestyle = ('-','-'), \
                                marker= None,markersize = 0.01, alpha = 1,linewidth = 1.5)
        
        myplots.myplot((x,x),(rei,imi),ax = axs,xlabel = xlab, \
                       color = ('k','k'),linestyle = ('--','--'), ylabel ='',\
                           marker= None, alpha = 1,linewidth = 1.5,\
                               xl = xl,yl = yl,xticks = xt,yticks = yt,grid = False,title = '')
    myplots.savemyfig(Figure_name)