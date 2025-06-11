#%% IMPORTING/SETTINGÂ UP PATHS

import sys
import os
import json
import tensorflow as tf
import numpy as np
# import tensorflow_probability as tfp

# Manually making sure the numpy random seeds are "the same" on all devices
np.random.seed(1234)
tf.random.set_seed(1234)

#%% LOCAL IMPORTS
GlobalPath = os.getcwd() # Change this if in CURC
# GlobalPath = '/projects/jomu3154/CURC_Nstep_RKmethod'
sys.path.append(os.path.join(GlobalPath,'..', "utils"))

#sys.path.append("G:\My Drive\Ultrafast Photonics\PINNs-TF2.0-master")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from scipy.signal.windows import tukey as tuk
from custom_lbfgs import lbfgs, Struct
from PI_FROG_util import prep_data, plot_prediction
from neuralnetwork import NeuralNetwork
from logger import Logger
from Make_FROG import makeFROG
from tqdm import trange
from time import sleep
import matplotlib.pyplot as plt
#%% HYPER PARAMETERSs

# Default parameters
hp = {}
# Data size on the solution u
hp["N"] = 2**8
# DeepNN topology (1-sized input [x], 3 hidden layer of 50-width, q-sized output defined later [u_1^n(x), ..., u_{q+1}^n(x)]
hp["layers"] = [1, 100, 100, 100, 100, 0]
# Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
hp["tf_epochs"] = 0000
hp["tf_lr"] = 0.003
hp["tf_b1"] = 0.9
hp["tf_b2"] = 0.999
hp["tf_eps"] = 1e-9 


# Setting up the quasi-newton LBGFS optimizer (set nt_epochs=0 to cancel it)
hp["nt_epochs"] = 5000
hp["nt_lr"] = 0.6
hp["nt_ncorr"] = 50

# logger
hp["log_frequency"] = 5

hp['log_checkpoints'] = True
hp["log_checkpoint_freq"] = 5
hp['NN_name'] = 'PIFROG_2'

hp['RKsteps'] = 1
hp['SNR'] = np.inf
hp['q']     = 200

# Fourier Axes
hp['dt']    = 12/64 # in units of T0
hp['WinT']  = 6 # in hp[T0 single sided
hp['nt']    = int(2*hp['WinT']/(hp['dt']))
hp['nyq_w'] = np.pi/(hp['dt'])
hp['dw0']   = np.pi/(hp['WinT'])

# Define the wanted sampling in frequency space. Easiest to define as a factor of the nyquist
hp['WinW']  = hp['nyq_w']/10 # in units of 1/T0 and must be <= pi/dt (single sided)
hp['dw']    = hp['dw0']/4 # in units of 1/T0
# Dependent terms
hp['nwtot'] = int(2*np.pi/(hp['dw']*hp['dt'])) # Padded nw total, Window will be cropped according to hp['WinW]
hp['pad']   = int((hp['nwtot']-hp['nt'])/2) 
hp['nw']    = int(2*hp['WinW']/hp['dw'])


#%% DEFININGÂ THEÂ MODEL

class PI_FROG(NeuralNetwork):
    def __init__(self, hp, logger, NN_hp,ub,lb,Trainable_vars, Init_guess):
        super().__init__(hp, logger, ub, lb)
        self.dz =  NN_hp['dz']
        self.q = max(NN_hp['q'], 1)
        self.RKsteps = NN_hp['RKsteps']
        self.t = NN_hp['t']
        self.z = NN_hp['z']
        n = len(self.t)
        self.w = np.arange(-n/2,n/2)*np.pi/(NN_hp['dt']*n)
        # Add a reshape layer so the output is (,q,2) U is dim 1 V is dim 2
        self.model.add(tf.keras.layers.Reshape((NN_hp['q']*NN_hp['RKsteps'],2))) 
        self.sizes_w.append(0) # Has no weights
        self.sizes_b.append(0)
        
        self.IRK_alpha =  NN_hp['IRK_alpha']
        self.IRK_beta = NN_hp['IRK_beta']
    
        self.D  = tf.Variable([Init_guess[0]], dtype = self.dtype,trainable = Trainable_vars[0])
        self.N  = tf.Variable([Init_guess[1]], dtype = self.dtype,trainable = Trainable_vars[1])
        
        # No Raman to start 7/25/2024
        # self.fR = tf.Variable([Init_guess[2]], dtype = self.dtype,trainable = Trainable_vars[2])
        # self.Ram_res = tf.Variable(np.expand_dims(Init_guess[3],axis=1), dtype = self.dtype, trainable = Trainable_vars[3])
        # self.Ram_res = tf.Variable(np.zeros([len(Init_guess[3]),1]), dtype = self.dtype, trainable = Trainable_vars[3])
        self.logger.log_custom_variables( Names_of_vars = ['D','N2'], \
                                          Vars2save = [self.D.numpy(),self.N.numpy()], \
                                          Initialize = True)
        
        self.FFT_NormConstant = 1
        self.WinT = NN_hp['dt']*NN_hp['N']
        self.nt = NN_hp['N']
        self.pad   = hp['pad']
        self.nw = hp['nw']
        self.hp = hp
    
    def autograd(self, U,V, t, dummy,tape):
        # Using the new GradientTape paradigm of TF2.0,
        # which keeps track of operations to get the gradient at runtime
        # Watching the two inputs weâll need later, x and t
        
        # Deriving INSIDE the tape (2-step-dummy grad technique because U is a mat)
        g_U = tape.gradient(U, t, output_gradients=dummy)
        U_t = tape.gradient(g_U, dummy)
        g_U_t = tape.gradient(U_t, t, output_gradients=dummy)
        # Deriving INSIDE the tape (2-step-dummy grad technique because U is a mat)
        g_V = tape.gradient(V, t, output_gradients=dummy)
        V_t = tape.gradient(g_V, dummy)
        g_V_t = tape.gradient(V_t, t, output_gradients=dummy)
          
        # Doing the last one outside the with, to optimize performance
        # Impossible to do for the earlier grad, because theyre needed after
        U_tt = tape.gradient(g_U_t, dummy)
        V_tt = tape.gradient(g_V_t, dummy)
        # Letting the tape go
        return U_t, U_tt, V_t, V_tt
    
    def UV_0_model(self, t, customDummy=None):
        if customDummy != None:
            dummy = customDummy
        else:
            dummy = self.dummy_t_0
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            tape.watch(dummy)
            
            UV = self.model(t) # shape = (N0, 2*q, 2)
            UV0 = []
            for i in range(0,self.RKsteps):
                UV0.append(UV[:,i*self.q:(1+i)*self.q,:])
            UU0 = []; VV0 = []
            UU  = [];  VV = []
            UU_t = []; VV_t = []
            UU_tt = []; VV_tt = []
            for UV in UV0:
                U = UV[:,:,0]
                V = UV[:,:,1]
                U_t, U_tt, V_t, V_tt = self.autograd(U,V, t, dummy,tape)
                # Convl_Re, Convl_Im   = self.Compute_Convolution(U,V,self.Ram_res)
                # Buidling the PINNs
                D = self.D
                N = self.N
                # fR = self.fR
                
                H2 = U**2+V**2
                NU =    D*V_tt + (N)*((H2)*V) # shape=(len(x), q)
                NV = -  D*U_tt - (N)*((H2)*U)
                U0 = U + self.dz*tf.matmul(NU, self.IRK_alpha.T)
                V0 = V + self.dz*tf.matmul(NV, self.IRK_alpha.T)
                
                UU0.append(U0);     VV0.append(V0)
                UU.append(U);       VV.append(V)      
                UU_t.append(U_t)  ; VV_t.append(V_t)
                UU_tt.append(U_tt); VV_tt.append(V_tt)
        del tape # Letting tape go
        return UU0, VV0, UU, VV, UU_t, VV_t
    
    def UV_1_model(self, t, customDummy=None):
        if customDummy != None:
            dummy = customDummy
        else:
            dummy = self.dummy_t_0
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            tape.watch(dummy)
            
            UV = self.model(t) # shape = (N0, 2*q, 2)
            UV0 = []
            for i in range(0,self.RKsteps):
                UV0.append(UV[:,i*self.q:(1+i)*self.q,:])
            UU1 = []; VV1 = []
            UU  = [];  VV = []
            UU_t = []; VV_t = []
            UU_tt = []; VV_tt = []
            for UV in UV0:
                U = UV[:,:,0]
                V = UV[:,:,1]
                U_t, U_tt, V_t, V_tt = self.autograd(U,V, t, dummy,tape)
                # Convl_Re, Convl_Im   = self.Compute_Convolution(U,V,self.Ram_res)
                # Buidling the PINNs
                D = self.D
                N = self.N
                # fR = self.fR
                # First step
                H2 = U**2+V**2
                NU =  -  D*V_tt - (N)*((H2*V)) # shape=(len(x), q)
                NV =     D*U_tt + (N)*((H2*U))
                U1 = U + self.dz*tf.matmul(NU, (self.IRK_beta - self.IRK_alpha).T)
                V1 = V + self.dz*tf.matmul(NV, (self.IRK_beta - self.IRK_alpha).T)
                
                # Second step
                U1 = U + self.dz*tf.matmul(NU, (self.IRK_beta - self.IRK_alpha).T)
                V1 = V + self.dz*tf.matmul(NV, (self.IRK_beta - self.IRK_alpha).T)
                UU1.append(U1);     VV1.append(V1)
                UU.append(U);       VV.append(V)      
                UU_t.append(U_t)  ; VV_t.append(V_t)
                UU_tt.append(U_tt); VV_tt.append(V_tt)
        return UU1, VV1, UU, VV, UU_t, VV_t
    
    # def Compute_Convolution(self,U,V,Response_t):
    #     Ohm = tf.constant(np.zeros([int(self.nt/2),1]),dtype = Response_t.dtype)
    #     Response_t = tf.concat([Ohm,Response_t],axis = 0)
    #     Response_t = tf.cast(tf.tile(Response_t,[1,self.q]),'complex128')
    #     H = tf.complex(U,V)
    #     H2 = tf.cast(U**2+V**2 ,'complex128')
    #     h2 = self.IFFT_tensor(H2)
    #     Response_w = self.IFFT_tensor(Response_t)
    #     Convl = self.WinT*self.FFT_tensor(h2*Response_w)*self.BC
    #     Convl_H = Convl*H
    #     Convl_Re = tf.math.real(Convl_H)
    #     Convl_Im = tf.math.imag(Convl_H)
    #     return Convl_Re, Convl_Im
    
    def FFT_tensor(self,tensor, NormCons = tf.constant([1],'complex128')):
        shape = tf.shape(tensor).numpy()
        if tensor.dtype != NormCons.dtype:
            NormCons = tf.cast(NormCons,tensor.dtype)
        if shape.size>2:
            raise RuntimeError('ERROR in FFT_tensor: tensor input has shape' + str(shape) + ' expecting <= 2d')
        tensor = tf.transpose(tensor)
        FFTtensor = tf.signal.fftshift(tf.signal.fft(tf.signal.fftshift(tensor)))/(NormCons)
        FFTtensor = tf.transpose(FFTtensor)
        return FFTtensor
          
    def IFFT_tensor(self,tensor,NormCons = tf.constant([1],'complex128')): # Function which returns the FFT of a tensor if tensor is 2d takes fourier transform across the colomn t should be the colomns
        shape = tf.shape(tensor).numpy()
        if tensor.dtype != NormCons.dtype:
            NormCons = tf.cast(NormCons,tensor.dtype)
        if shape.size>2:
            raise RuntimeError('ERROR in IFFT_tensor: tensor input has shape' + str(shape) + ' expecting <= 2d')
        tensor = tf.transpose(tensor)
        IFFTtensor = tf.signal.ifftshift(tf.signal.ifft(tf.signal.ifftshift(tensor)))*(NormCons)
        IFFTtensor = tf.transpose(IFFTtensor)
        return IFFTtensor     
      
      # Defining custom loss
    def loss(self, t, uu_0, vv_0, uu_1, vv_1):
        UU_0_pred, VV_0_pred, UU0, VV0, UU0_t, VV0_t = self.UV_0_model(t)
        UU_1_pred, VV_1_pred, UU1, VV1, UU1_t, VV1_t = self.UV_1_model(t)
        
        mse0 = 0
        mse1 = 0
        for i in range(0,self.RKsteps):
            # Get the prediction at each input and output pair
            u_0_pred = UU_0_pred[i]
            v_0_pred = VV_0_pred[i]
            u_1_pred = UU_1_pred[i]
            v_1_pred = VV_1_pred[i]
                
            # Get the labeled data at every input and output
            u_0  = uu_0[i]
            v_0  = vv_0[i]
            u_1  = uu_1[i]
            v_1  = vv_1[i]
            
            # MSE loss function for each step at input 0 and output 1 of the RK of q order stages.
            mse0 = tf.reduce_sum(tf.square(u_0_pred- u_0)) +\
                tf.reduce_sum(tf.square(v_0_pred- v_0)) + mse0
                
            mse1 = tf.reduce_sum(tf.square(u_1_pred- u_1)) +\
                tf.reduce_sum(tf.square(v_1_pred- v_1)) + mse1
                    
        return mse0 + mse1
    
    
    def grad(self, t, u_0, v_0, u_1, v_1):
        with tf.GradientTape() as tape:
          loss_value = self.loss(t, u_0, v_0, u_1, v_1)
        return loss_value, tape.gradient(loss_value, self.wrap_training_variables())
    
    def wrap_training_variables(self):
        var = self.model.trainable_variables
        if self.D.trainable:       var.extend([self.D])
        if self.N.trainable:       var.extend([self.N])
        
        # if self.fR.numpy() > 1:
        #     self.fR.assign(np.array([1],dtype = self.dtype))
        # elif self.fR.numpy() < 0:
        #     self.fR.assign(np.array([0],dtype = self.dtype))
            
        # if self.Ram_res.trainable: var.extend([self.Ram_res])
        # if self.fR.trainable:      var.extend([self.fR])
        return var
    
    def get_weights(self):
            w = super().get_weights(convert_to_tensor=False, LayerInx = np.arange(1,len(self.model.layers[:])-1))
            self.NN_nw = len(w) # number of weights in neural network 
            # if self.Ram_res.trainable: 
            #     self.Ram_res_w_inx = (len(w),len(w)+len(self.Ram_res.numpy()[:,0]))
            #     w.extend(self.Ram_res.numpy()[:,0])
            if self.D.trainable:    
                self.D_w_inx = len(w)
                w.extend(self.D.numpy())
            if self.N.trainable:
                self.N_w_inx = len(w)
                w.extend(self.N.numpy())
            # if self.fR.trainable:
            #     self.fR_w_inx = len(w)
            #     w.extend(self.fR.numpy())
            return tf.convert_to_tensor(w, dtype=self.dtype)
    
    def set_weights(self, w):
        super().set_weights(w)
        # if self.Ram_res.trainable: self.Ram_res.assign(np.array([w[self.Ram_res_w_inx[0]:self.Ram_res_w_inx[1]]]).T*self.BC[int(len(self.BC)/2)-1:-1])
        if self.D.trainable:       self.D.assign([w[self.D_w_inx]])
        if self.N.trainable:       self.N.assign([w[self.N_w_inx]])
        # if self.fR.trainable:      self.fR.assign([w[self.fR_w_inx]])
    
    def get_params(self, numpy=False):
        D = self.D
        N = self.N
        # fR = self.fR
        # Ram_res = self.Ram_res
        if numpy:
          return D.numpy()[0], N.numpy()[0]# , fR.numpy()[0], Ram_res.numpy()[:,0:1]
        return D, N# , fR, Ram_res
    
    def createDummy(self, t):
        return tf.ones([t.shape[0], self.q], dtype=self.dtype)
    
      # The training function
    def fit(self, t, u0, v0, u1, v1):
        self.logger.log_train_start(self)
        u_0 = []; v_0 = []
        u_1 = []; v_1 = []
        # Creating the tensors
        t   = tf.convert_to_tensor(t, dtype=self.dtype)
        for i in range(0, self.RKsteps):
            u_0.append(tf.convert_to_tensor(u0[i], dtype=self.dtype)) 
            v_0.append(tf.convert_to_tensor(v0[i], dtype=self.dtype)) 
            u_1.append(tf.convert_to_tensor(u1[i], dtype=self.dtype)) 
            v_1.append(tf.convert_to_tensor(v1[i], dtype=self.dtype)) 
            
    
    
        # Creating dummy tensors for the gradients important because we only want to take the derivate w.r.t to the rows not the coloumns of the output
        self.dummy_t_0 = self.createDummy(t)
        self.dummy_t_1 = self.createDummy(t)
    
        def log_train_epoch(epoch, loss, NeurNet, is_iter):
            D, N, fR, Ram_res = self.get_params(numpy=True)
            custom = f"D = {D:5f}  N = {N:5f} fR = {fR:6f}"
            self.logger.log_train_epoch(epoch, loss, NeurNet, custom, is_iter, saveVars = True, Vars2save = [D,N,fR,Ram_res])
    
        self.logger.log_train_opt("Adam")
        for epoch in range(hp["tf_epochs"]):
            # Optimization step
            loss_value, grads = self.grad(t, u_0, v_0, u_1, v_1)
            self.tf_optimizer.apply_gradients(
              zip(grads, self.wrap_training_variables()))
            
            log_train_epoch(epoch, loss_value,self, False)
        
        self.logger.log_train_opt("LBFGS")
        def loss_and_flat_grad(w):
            with tf.GradientTape() as tape:
                self.set_weights(w)
                tape.watch(self.D)
                tape.watch(self.N)
                tape.watch(self.fR)
                tape.watch(self.Ram_res)
                loss_value = self.loss(t, u_0, v_0, u_1, v_1)
            grad = tape.gradient(loss_value, self.wrap_training_variables())
            grad_flat = []
            for g in grad:
                grad_flat.append(tf.reshape(g, [-1]))
            grad_flat =  tf.concat(grad_flat, 0)
            return loss_value, grad_flat
        
        lbfgs(loss_and_flat_grad,
          self.get_weights(),
          self.nt_config, Struct(), True, log_train_epoch, NeurNet = self)
        
    
    def predict(self, t_star):
        t_star = tf.convert_to_tensor(t_star, dtype=self.dtype)
        dummy = self.createDummy(t_star)
        U_0_star, V_0_star, U_0, V_0, U_t_0, V_t_0 = self.UV_0_model(t_star, dummy)
        U_1_star, V_1_star,U_1, V_1, U_t_1, V_t_1 = self.UV_1_model(t_star, dummy)
        UV = self.model(t_star)
        U_pred = UV[:,:,0]
        V_pred = UV[:,:,1]
        h0mean = tf.complex(tf.math.reduce_mean(U_0_star[0],axis = 1), tf.math.reduce_mean(V_0_star[0],axis = 1))
        h1mean = tf.complex(tf.math.reduce_mean(U_1_star[0],axis = 1), tf.math.reduce_mean(V_1_star[0],axis = 1))
        FROG0_pred = makeFROG(h0mean,h0mean,pad = self.pad,wcrop = self.nw)
        FROG1_pred = makeFROG(h1mean,h1mean,pad = self.pad,wcrop = self.nw)
        return U_0_star, V_0_star, U_1_star, V_1_star, U_pred, V_pred, FROG0_pred,FROG1_pred
    
    def load_latest_checkpoint(self, indexnum = 'last'):
        super().load_latest_checkpoint(chk_point_num = str(indexnum))
        self.logger.log_load_logged_data()    
        if indexnum == 'last':
            self.D.assign([self.logger.logger_data['D'][-1]])
            self.N.assign([self.logger.logger_data['N2'][-1]])
        else:
            inx = int(indexnum/self.logger.log_checkpoint_freq)
            self.D.assign([self.logger.logger_data['D'][inx]])
            self.N.assign([self.logger.logger_data['N2'][inx]])
        
    def get_predict(self, numpy = False):
        
        u_0_pred, v_0_pred, u_1_pred, v_1_pred, U_pred, V_pred, FROG_0_pred, FROG_1_pred = self.predict(self.t)
        if numpy:
            u0p_np = []; v0p_np = []
            u1p_np = []; v1p_np = []
            FROG_0_np = []
            FROG_1_np = []
            Up     = U_pred.numpy(); Vp     = V_pred.numpy()
            for i in range(0,self.RKsteps):
                u0p_np.append(u_0_pred[i].numpy())
                v0p_np.append(v_0_pred[i].numpy())
                u1p_np.append(u_1_pred[i].numpy())
                v1p_np.append(v_1_pred[i].numpy())
                FROG_0_np.append(FROG_0_pred.numpy())
                FROG_1_np.append(FROG_1_pred.numpy())
            return u0p_np, v0p_np, u1p_np, v1p_np, Up, Vp, FROG_0_np, FROG_1_np,\
                 self.z, self.t

        else:
            return u_0_pred, v_0_pred, u_1_pred, v_1_pred, U_pred, V_pred, FROG_0_pred, FROG_1_pred, \
                self.z, self.t

      
#%% TRAININGÂ THEÂ MODEL
def get_PINN(hp = hp,datafname = 'PI_FROG_SIM_v1.mat',indexnum = 'last',ModelDirectory = os.getcwd(),D_trainable = False, N2_trainable = False, Load_Trained_model = False):
    # Getting the data
    # datafname = 'GNLSE_Disc_Raman_On.mat'
    if 'datafname' in hp:
        datafname = hp['datafname']
    hp['datafname'] = datafname
    
    sim_hp,sim_data,NN_hp = prep_data(datafname,hp, GlobalPath = GlobalPath, RKsteps = hp['RKsteps'], N=hp["N"], SNR=hp['SNR'],q = hp['q'])
    # The True parameter values
    # Setting the output layer dynamically
    hp["layers"][-1] = int(2*NN_hp['q']*hp['RKsteps'])
    lambdas_star = (sim_hp['D']/2,sim_hp['N2'])
    Trainable_Variables = (D_trainable, N2_trainable)
    
    # Creating the model
    logger = Logger(hp,Directory = ModelDirectory)
    logger.sim_data = sim_data
    pinn = PI_FROG(hp, logger, NN_hp, NN_hp['ub'][0], NN_hp['lb'][0], Trainable_vars = Trainable_Variables, Init_guess = (lambdas_star))
    # Creating the modelDataParams
    dt = hp['dt']
    def fft(u,v,pad = hp['pad']):
        h = u+1j*v
        if len(h.shape) == 1:
            h = np.expand_dims(h, axis= 1)
        h = np.concatenate([np.zeros([pad,h.shape[1]]),h,np.zeros([pad,h.shape[1]])],axis = 0)
        H = np.fft.fftshift(np.fft.fft(np.fft.fftshift(h,axes = 0),axis = 0),axes = 0)/len(h)*dt*len(h)
        H = H[int(len(H)/2-hp['wcrop']/2):int(len(H)/2+hp['wcrop']/2)]
        return np.real(H),np.imag(H)
    pinn.fft = fft
    # Defining the error function if error function was not set
    def error():
      D, N = pinn.get_params(numpy=True)
      D_star, N_star = lambdas_star
      def L2(a,astar):
          return np.sqrt(a**2-astar**2)/np.sqrt(astar**2)
        
      err = (L2(D,D_star),L2(N,N_star))
      preds = (D,N)
      truth = (N,N_star)
      return err, preds, truth
    logger.set_error_fn(error)
    
    # Set the fitting and predicition functions for the specific data set 
    pinn.training_data = (NN_hp['FROG_0'],NN_hp['FROG_1'],NN_hp['t'],NN_hp['w'])
    pinn.PerfectData = (NN_hp['FROG_0'],NN_hp['FROG_1'],np.real(sim_data['Clean_h0']),np.imag(sim_data['Clean_h0']),np.real(sim_data['Clean_h1']),np.imag(sim_data['Clean_h1']))
    
    def Start_PINN_fit():
        pinn.fit(NN_hp['t'],\
                 NN_hp['FROG_0'],\
                  NN_hp['FROG_1'])
            
    def Continue_PINN_fit(numIters = hp['nt_epochs']):
        pinn.load_latest_checkpoint()
        # Extend the epochs too current training to redefined 
        pinn.nt_config.maxIter = int(numIters+pinn.logger.epoch_ax[-1])
        pinn.nt_config.startIter = int(pinn.logger.epoch_ax[-1])
        pinn.hp['nt_epochs'] = int(numIters+pinn.logger.epoch_ax[-1])
        pinn.fit(NN_hp['t_0'], NN_hp['u_0'], NN_hp['v_0'],\
                 NN_hp['t_1'], NN_hp['u_1'], NN_hp['v_1'])
        
 
    pinn.Start_fit = Start_PINN_fit
    pinn.Continue_fit = Continue_PINN_fit
    
    if Load_Trained_model:
        pinn.load_latest_checkpoint(indexnum = indexnum)
    
    return pinn
    # TRAINING!!!!