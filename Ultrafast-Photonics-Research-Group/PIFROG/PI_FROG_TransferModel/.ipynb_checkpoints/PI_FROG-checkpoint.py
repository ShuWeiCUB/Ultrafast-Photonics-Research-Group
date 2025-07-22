#%% IMPORTING/SETTINGÂ UP PATHS

import sys
import os
import json
import tensorflow as tf
import numpy as np
import time

import joblib
from joblib import Parallel, delayed
from tqdm import trange
import multiprocessing as mp  
import scipy.io as sio

from scipy.ndimage import zoom
from skimage.transform import resize
from scipy.interpolate import interp1d


# import tensorflow_probability as tfp

#import concurrent.futures


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
from PI_FROG_util import prep_data, prep_dataBM, plot_prediction
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
hp["N"] = 64
# DeepNN topology (1-sized input [x], 3 hidden layer of 50-width, q-sized output defined later [u_1^n(x), ..., u_{q+1}^n(x)]
hp["layers"] = [1, 100, 100, 100, 100, 0]
# Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
hp["tf_epochs"] = 0000
hp["tf_lr"] = 0.003
hp["tf_b1"] = 0.9
hp["tf_b2"] = 0.999
hp["tf_eps"] = 1e-9 


# Setting up the quasi-newton LBGFS optimizer (set nt_epochs=0 to cancel it)
hp["nt_epochs"] = 100
hp["nt_lr"] = 0.6
hp["nt_ncorr"] = 50
hp["log_frequency"] = 5
hp['log_checkpoints'] = True #True
hp["log_checkpoint_freq"] = 100 #5

# logger
hp['BM_nt_epochs'] = 200
hp["BM_log_frequency"] = 100
hp['BM_log_checkpoints'] = True
hp["BM_log_checkpoint_freq"] = 100 #100

hp['NN_name'] = 'PIFROG_BaseModel'

hp['RKsteps'] = 1
hp['SNR'] = np.inf
hp['q']     = 100

print("hp['N'] " + str(hp['N']))
# Fourier Axes
hp['dt']    = 2/hp['N'] #12 in units of T0 #12 

print("dt " + str(hp['dt']))
hp['WinT']  = 4 #6 # in hp[T0 single sided #6 in this case the 2 is going to be 2 ps
print("WinT " + str(hp['WinT']))

hp['nt']    = int(2*hp['WinT']/(hp['dt'])) # 2
hp['nyq_w'] = np.pi/(hp['dt'])
hp['dw0']   = np.pi/(hp['WinT'])

# Define the wanted sampling in frequency space. Easiest to define as a factor of the nyquist
hp['WinW']  = hp['nyq_w']/10 #10 # in units of 1/T0 and must be <= pi/dt (single sided)
hp['dw']    = hp['dw0']/4 #4 # in units of 1/T0
# Dependent terms
hp['nwtot'] = int(2*np.pi/(hp['dw']*hp['dt'])) # Padded nw total, Window will be cropped according to hp['WinW]
hp['pad']   = int((hp['nwtot']-hp['nt'])/2) 
hp['nw']    = int(2*hp['WinW']/hp['dw'])

'''
def compute_loss_and_grad(q, w, t, FROG_0, FROG_1, model):
    """Compute loss and gradients for a given q (must be outside the class for pickling)."""
    with tf.GradientTape() as tape:
        model.set_weights(w)  # Set model weights for each process
        tape.watch(model.D)
        tape.watch(model.N)
        loss_value_q = model.loss(t, q, FROG_0, FROG_1)

    grad_q = tape.gradient(loss_value_q, model.wrap_training_variables())
    return loss_value_q, grad_q
'''

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
        #self.w = np.arange(-n/2,n/2)*np.pi/(NN_hp['dt']*n)
        self.t = NN_hp['t']
        self.w = np.linspace(-6.4,6.4, n)

        # Add a reshape layer so the output is (,q,2) U is dim 1 V is dim 2
        self.model.add(tf.keras.layers.Reshape((NN_hp['q']*NN_hp['RKsteps'],2))) 
        self.sizes_w.append(0) # Has no weights
        self.sizes_b.append(0)
        
        self.IRK_alpha =  NN_hp['IRK_alpha']
        self.IRK_beta = NN_hp['IRK_beta']
    
        self.D  = tf.Variable([Init_guess[0]], dtype = self.dtype,trainable = Trainable_vars[0])
        self.N  = tf.Variable([Init_guess[1]], dtype = self.dtype,trainable = Trainable_vars[1])

        #self.D  = tf.Variable([0.5], dtype = self.dtype,trainable = Trainable_vars[0])
        #self.N  = tf.Variable([1.69], dtype = self.dtype,trainable = Trainable_vars[1])
        
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

    def fft_center_pulse_spatial(self, U, V):
        """
        Centers the pulse along the Q axis (spatial centering).
        U, V: real and imaginary parts of pulse, shape (T, Q)
        Returns centered real and imaginary parts with complex128 precision.
        """
        # Ensure float64
        U = tf.cast(U, tf.float64)
        V = tf.cast(V, tf.float64)
    
        # Combine into complex
        UC = tf.complex(U, V)  # shape (T, Q)
    
        # Compute intensity
        intensity = tf.abs(UC) ** 2  # (T, Q)
    
        # Q axis: assume symmetric about 0
        Q = tf.shape(U)[1]
        q = tf.linspace(-1.0, 1.0, Q)
        q = tf.cast(q[tf.newaxis, :], tf.float64)  # shape (1, Q)
    
        # Center of mass per T
        weighted_sum = tf.reduce_sum(q * intensity, axis=1)  # (T,)
        total = tf.reduce_sum(intensity, axis=1) + 1e-20
        center_of_mass = weighted_sum / total  # (T,)
    
        # Frequency axis for Q direction
        freqs = tf.signal.fftshift(tf.linspace(-np.pi, np.pi, Q))  # (Q,)
        freqs = tf.cast(freqs, tf.float64)
    
        # Apply phase ramp to shift center of mass to 0
        omega = tf.reshape(freqs, [1, -1])  # (1, Q)
        shift = -center_of_mass  # (T,)
        omega_shift = tf.transpose(shift[:, tf.newaxis] * omega)  # (Q, T)
        phase_ramp = tf.exp(tf.complex(tf.zeros_like(omega_shift), omega_shift))  # (Q, T)
    
        # FFT along Q (spatial) axis
        UC_fft = tf.signal.fft(tf.transpose(UC))  # (Q, T)
        UC_fft_shifted = UC_fft * tf.cast(phase_ramp, tf.complex128)  # (Q, T)
        UC_shifted = tf.signal.ifft(UC_fft_shifted)  # (Q, T)
        UC_shifted = tf.transpose(UC_shifted)  # (T, Q)
    
        return tf.math.real(UC_shifted), tf.math.imag(UC_shifted)


    def fft_center_pulse(self, U, V, t_axis=0):
        """
        U, V: real and imaginary parts of pulse, shape (T, Q)
        Returns centered real and imaginary parts with complex128 precision.
        """
        # Ensure inputs are float64
        U = tf.cast(U, tf.float64)
        V = tf.cast(V, tf.float64)
        
        # Convert to complex128
        UC = tf.complex(U, V)  # shape (T, Q), dtype complex128
    
        # Compute intensity
        intensity = tf.abs(UC) ** 2  # shape (T, Q)
    
        # Time axis (assumed to be symmetric about 0)
        T = tf.shape(U)[0]
        t = tf.linspace(-1.0, 1.0, T)
        t = tf.cast(t[:, tf.newaxis], tf.float64)  # shape (T, 1)
    
        # Center of mass per Q
        weighted_sum = tf.reduce_sum(t * intensity, axis=0)  # (Q,)
        total = tf.reduce_sum(intensity, axis=0) + 1e-20
        center_of_mass = weighted_sum / total  # shape (Q,)
    
        # Frequency axis (FFT-compatible)
        freqs = tf.signal.fftshift(tf.linspace(-np.pi, np.pi, T))  # shape (T,)
        freqs = tf.cast(freqs, tf.float64)
    
        # Phase ramp: exp(+i * omega * shift)
        omega = tf.reshape(freqs, [-1, 1])  # shape (T, 1)
        shift = -center_of_mass  # shape (Q,)
        omega_shift = omega * shift  # shape (T, Q)
        phase_ramp = tf.exp(tf.complex(tf.zeros_like(omega_shift), omega_shift))  # complex128

        print(phase_ramp)
    
        # FFT shift
        #UC_fft = tf.signal.fft(tf.cast(UC, tf.complex128), axis=0)

        UC_T = tf.transpose(UC)  # (Q, T)
        UC_fft_T = tf.signal.fft(UC_T)  # (Q, T)
        UC_fft = tf.transpose(UC_fft_T)  # (T, Q)
        
        UC_fft_shifted = UC_fft * tf.cast(phase_ramp, tf.complex128)
        
        #UC_shifted = tf.signal.ifft(UC_fft_shifted, axis=0)

        UC_fft_shifted_T = tf.transpose(UC_fft_shifted)  # (Q, T)
        UC_shifted_T = tf.signal.ifft(UC_fft_shifted_T)  # (Q, T)
        UC_shifted = tf.transpose(UC_shifted_T)  # (T, Q)
            
        # Return real and imaginary parts
        return tf.math.real(UC_shifted), tf.math.imag(UC_shifted)
        
    def fft_center_pulse_fix(self, U, V, t_axis=0):
        # Ensure float64
        #U = tf.cast(U, tf.float64)
        #V = tf.cast(V, tf.float64)
        
        # Combine into complex128
        UC = tf.complex(U, V)  # shape (T, Q), dtype complex128
        
        # Intensity (|E|²)
        intensity = tf.abs(UC) ** 2  # shape (T, Q), dtype float64
        
        # Time axis (must be float64)
        T = tf.shape(U)[0]
        t = tf.cast(self.t, tf.float64)  # ensure float64
        t = tf.reshape(t, [-1, 1])  # shape (T, 1)
        
        # Midpoint correction
        t_midpoint = (t[-1] + t[0]) / 2.0  # float64

        #print(t_midpoint)
        #t_centered = t - t_midpoint  # float64, shape (T, 1)
        t_centered = t # float64, shape (T, 1)
        
        # Center-of-mass in time per Q slice
        weighted_sum = tf.reduce_sum(t_centered * intensity, axis=0)  # shape (Q,), float64
        total = tf.reduce_sum(intensity, axis=0) + 1e-20  # shape (Q,), float64
        center_of_mass = weighted_sum / total  # shape (Q,), float64
        
        # Frequency axis
        delta_t = (t[-1] - t[0]) / tf.cast(T - 1, tf.float64)  # float64
        freqs = 2.0 * np.pi * tf.signal.fftshift(tf.linspace(-0.5 / delta_t, 0.5 / delta_t, T))  # shape (T,), float64
        freqs = tf.cast(freqs, tf.float64)  # explicit cast
        omega = tf.reshape(freqs, [-1, 1])  # shape (T, 1), float64
        
        # Phase ramp (complex128)
        shift = -center_of_mass  # shape (Q,), float64
        omega_shift = tf.cast(omega * shift, tf.complex128)  # shape (T, Q), complex128
        phase_ramp = tf.exp(omega_shift * 1j)  # shape (T, Q), complex128
        
        # FFT → apply phase ramp → IFFT
        UC_fft = tf.signal.fft(tf.cast(tf.transpose(UC), tf.complex128))  # shape (Q, T), complex128
        UC_fft_shifted = UC_fft * tf.transpose(phase_ramp)  # shape (Q, T), complex128
        UC_shifted = tf.signal.ifft(UC_fft_shifted)  # shape (Q, T), complex128
        
        # Return real and imaginary parts, shape (T, Q)
        UC_shifted = tf.transpose(UC_shifted)  # shape (T, Q)
        
        return tf.math.real(UC_shifted), tf.math.imag(UC_shifted)

    def center_complex_pulse_to_t0(self,E, t):
        """
        Shift a complex-valued pulse E(t) so its intensity center of mass is at t = 0.
    
        Args:
            E: tf.Tensor of shape (T,), dtype complex128
            t: tf.Tensor of shape (T,), dtype float64
    
        Returns:
            tf.Tensor of shape (T,), dtype complex128
        """
        # Ensure correct types
        E = tf.cast(E, tf.complex128)
        t = tf.cast(t, tf.float64)
    
        # Compute intensity and center of mass
        intensity = tf.math.abs(E) ** 2  # float64
        total_intensity = tf.reduce_sum(intensity) + tf.constant(1e-20, dtype=tf.float64)
        center_of_mass = tf.reduce_sum(t * intensity) / total_intensity  # float64 scalar
    
        # Frequency axis (omega)
        T = tf.cast(tf.shape(t)[0], tf.float64)
        delta_t = (t[-1] - t[0]) / (T - tf.constant(1.0, dtype=tf.float64))
        freqs = tf.signal.fftshift(tf.linspace(-0.5 / delta_t, 0.5 / delta_t, tf.shape(t)[0]))  # float64
        omega = tf.constant(2.0 * np.pi, dtype=tf.float64) * freqs  # float64
    
        # Phase ramp to center pulse
        zero64 = tf.constant(0.0, dtype=tf.float64)
        imag_part = -omega * center_of_mass  # float64
        phase_ramp = tf.exp(tf.complex(zero64, imag_part))  # complex128
    
        # FFT shift
        E_fft = tf.signal.fft(E)  # complex128
        E_fft_shifted = E_fft * tf.signal.fftshift(phase_ramp)  # complex128
        E_centered = tf.signal.ifft(E_fft_shifted)  # complex128
    
        return E_centered

    
    
    
    #####ORGINAL CODE
    def UV_0_model(self, t, customDummy=None):

        t = t*10
        if customDummy != None:
            dummy = customDummy
        else:
            dummy = self.dummy_t_0
            
        time0 = time.time()
        
        with tf.GradientTape(persistent=True) as tape:
            
            t1 = time.time()
            
            tape.watch(t)
            tape.watch(dummy)
           
            UV = self.model(t) # shape = (N0, 2*q, 2)


            #t_fourier = self.fourier_embed(y)
            #UV = self.model(t_fourier)
            UV0 = []
            
            t2 = time.time()
            
            #print("time for self.model(t) " + str(t2-t1))
            
            for i in range(0,self.RKsteps):
                UV0.append(UV[:,i*self.q:(1+i)*self.q,:])
                
            
            UU0 = []; VV0 = []
            UU  = [];  VV = []
            UU_t = []; VV_t = []
            UU_tt = []; VV_tt = []
            
            t3 = time.time()
            
            #print("time for appending" + str(t3-t2))

            phase_penalty = 0

            
            for UV in UV0:
                
                t4 = time.time()
                
                #U = UV[:,:,0]
                #V = UV[:,:,1]

                U = UV[:, :, 0]  # shape (64, 100)
                V = UV[:, :, 1]  # shape (64, 100)

                #U,V = self.fft_center_pulse_fix(U, V)
                '''
                # Convert to complex
                UC = tf.complex(U, V)  # shape (64, 100)
                
                # Compute mean complex amplitude per q slice (over time axis)
                mean_complex = tf.reduce_mean(UC, axis=0)  # shape (100,)
                
                # Extract the average phase for each q slice
                mean_phase = tf.math.angle(mean_complex)  # shape (100,)
                mean_phase = tf.cast(mean_phase, tf.complex128)

                phase_penalty = phase_penalty + tf.reduce_mean(tf.square(mean_phase))
                
                # Construct phase correction factor
                phase_shift = tf.exp(tf.complex(tf.cast(0.0, tf.float64), tf.cast(-1.0, tf.float64)) * mean_phase)  # shape (100,)
                
                
                # Apply correction across time (broadcast over time dimension)
                UC_aligned = UC * phase_shift  # (64, 100), broadcasting over axis 1
                
                # Convert back to real and imaginary parts
                U = tf.math.real(UC_aligned)
                V = tf.math.imag(UC_aligned)
                '''
                
                U_t, U_tt, V_t, V_tt = self.autograd(U,V, t, dummy,tape)
                # Convl_Re, Convl_Im   = self.Compute_Convolution(U,V,self.Ram_res)
                # Buidling the PINNs
                D = self.D
                N = self.N
                # fR = self.fR
                
                t5 = time.time()
                
                #print("time for self.autograd " + str(t5-t4))
                
                H2 = U**2+V**2
                NU =    D*V_tt + (N)*((H2)*V) # shape=(len(x), q)
                NV = -  D*U_tt - (N)*((H2)*U)
                
                t6 = time.time()
                
                #print("time for NU, NV computation " + str(t6-t5))
                
                
                
                U0 = U + self.dz*tf.matmul(NU, self.IRK_alpha.T)
                V0 = V + self.dz*tf.matmul(NV, self.IRK_alpha.T)
                
                t7 = time.time()
                
                #print("time for matmul " + str(t7-t6))
                
                UU0.append(U0);     VV0.append(V0)
                UU.append(U);       VV.append(V)      
                UU_t.append(U_t)  ; VV_t.append(V_t)
                UU_tt.append(U_tt); VV_tt.append(V_tt)
                
                t8 = time.time()
                
                #print("time for appending " + str(t8-t7))
              
                   
        del tape # Letting tape go
        return UU0, VV0, UU, VV, UU_t, VV_t
   

    
    def UV_1_model(self, t, customDummy=None):

        t = t*10
        if customDummy != None:
            dummy = customDummy
        else:
            dummy = self.dummy_t_0
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            tape.watch(dummy)
            
            UV = self.model(t) # shape = (N0, 2*q, 2)

            #t_fourier = self.fourier_embed(y)
            #UV = self.model(t_fourier)
            
            UV0 = []
            for i in range(0,self.RKsteps):
                UV0.append(UV[:,i*self.q:(1+i)*self.q,:])
            UU1 = []; VV1 = []
            UU  = [];  VV = []
            UU_t = []; VV_t = []
            UU_tt = []; VV_tt = []

            phase_penalty = 0
            for UV in UV0:
                #U = UV[:,:,0]
                #V = UV[:,:,1]

                U = UV[:, :, 0]  # shape (64, 100)
                V = UV[:, :, 1]  # shape (64, 100)
                #U,V = self.fft_center_pulse_fix(U, V)

                '''
                # Convert to complex
                UC = tf.complex(U, V)  # shape (64, 100)
                
                # Compute mean complex amplitude per q slice (over time axis)
                mean_complex = tf.reduce_mean(UC, axis=0)  # shape (100,)
                
                # Extract the average phase for each q slice
                mean_phase = tf.math.angle(mean_complex)  # shape (100,)
                mean_phase = tf.cast(mean_phase, tf.complex128)

                phase_penalty = phase_penalty + tf.reduce_mean(tf.square(mean_phase))

                
                # Construct phase correction factor
                phase_shift = tf.exp(tf.complex(tf.cast(0.0, tf.float64), tf.cast(-1.0, tf.float64)) * mean_phase)  # shape (100,)

                
                
                # Apply correction across time (broadcast over time dimension)
                UC_aligned = UC * phase_shift  # (64, 100), broadcasting over axis 1
                
                # Convert back to real and imaginary parts
                U = tf.math.real(UC_aligned)
                V = tf.math.imag(UC_aligned)
                '''

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
        
    '''  
    
    #########SPEEDUP ATTEMPT
    
    def UV_0_model(self, t, customDummy=None):
        dummy = customDummy if customDummy is not None else self.dummy_t_0

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            tape.watch(dummy)

            UV = self.model(t)  # shape = (N0, 2*q, 2)
            UV0 = tf.split(UV, num_or_size_splits=self.RKsteps, axis=1)  # More efficient than a loop

            UU0, VV0, UU, VV, UU_t, VV_t, UU_tt, VV_tt = [], [], [], [], [], [], [], []

            for UV in UV0:
                U, V = UV[..., 0], UV[..., 1]
                U_t, U_tt, V_t, V_tt = self.autograd(U, V, t, dummy, tape)

                H2 = U**2 + V**2  # Store once, use twice
                NU = self.D * V_tt + self.N * (H2 * V)
                NV = -self.D * U_tt - self.N * (H2 * U)

                IRK_alpha_T = tf.transpose(self.IRK_alpha)  # Reduce redundant transposes
                
                U0 = U + self.dz * tf.matmul(tf.cast(NU, tf.float64), tf.cast(IRK_alpha_T, tf.float64))
                V0 = V + self.dz * tf.matmul(tf.cast(NV, tf.float64), tf.cast(IRK_alpha_T, tf.float64))

                #U0 = U + self.dz * tf.matmul(NU, IRK_alpha_T)
                #V0 = V + self.dz * tf.matmul(NV, IRK_alpha_T)
                
                

                UU0.append(U0); VV0.append(V0)
                UU.append(U); VV.append(V)
                UU_t.append(U_t); VV_t.append(V_t)
                UU_tt.append(U_tt); VV_tt.append(V_tt)

        del tape  # Free memory
        return UU0, VV0, UU, VV, UU_t, VV_t
    
    def UV_1_model(self, t, customDummy=None):
        dummy = customDummy if customDummy is not None else self.dummy_t_0

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            tape.watch(dummy)

            UV = self.model(t)  # shape = (N0, 2*q, 2)
            UV0 = tf.split(UV, num_or_size_splits=self.RKsteps, axis=1)

            UU1, VV1, UU, VV, UU_t, VV_t, UU_tt, VV_tt = [], [], [], [], [], [], [], []

            for UV in UV0:
                U, V = UV[..., 0], UV[..., 1]
                U_t, U_tt, V_t, V_tt = self.autograd(U, V, t, dummy, tape)

                H2 = U**2 + V**2
                NU = -self.D * V_tt - self.N * (H2 * V)
                NV = self.D * U_tt + self.N * (H2 * U)

                IRK_beta_minus_alpha_T = tf.transpose(self.IRK_beta - self.IRK_alpha)
                
                U1 = U + self.dz * tf.matmul(tf.cast(NU, tf.float64), tf.cast(IRK_beta_minus_alpha_T, tf.float64))
                V1 = V + self.dz * tf.matmul(tf.cast(NV, tf.float64), tf.cast(IRK_beta_minus_alpha_T, tf.float64))

                


                #U1 = U + self.dz * tf.matmul(NU, IRK_beta_minus_alpha_T)
                #V1 = V + self.dz * tf.matmul(NV, IRK_beta_minus_alpha_T)

                UU1.append(U1); VV1.append(V1)
                UU.append(U); VV.append(V)
                UU_t.append(U_t); VV_t.append(V_t)
                UU_tt.append(U_tt); VV_tt.append(V_tt)

        del tape
        return UU1, VV1, UU, VV, UU_t, VV_t

    '''

    
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

    
    def loss(self, t, q, FROG_0, FROG_1, all_h0r, all_h0i, all_h1r, all_h1i, epoch, phase_loss_shared, frog_loss_shared):
        
        #print("t variable is " + str(t))
        
        time0 = time.time()
        
        #########################This is a lossy line
        #batch_size = 16  # Choose based on your GPU memory
        #t_batches = tf.split(t, num_or_size_splits=batch_size)
        #UU_0_pred, VV_0_pred, UU0, VV0, UU0_t, VV0_t = [self.UV_0_model(batch) for batch in t_batches]
        #UU_1_pred, VV_1_pred, UU1, VV1, UU1_t, VV1_t = [self.UV_1_model(batch) for batch in t_batches]
        
        #@tf.function
        #def run_models(t):
        #    results_0 = self.UV_0_model(t)
        #    results_1 = self.UV_1_model(t)
            
        #    return results_0, results_1

        #results_0, results_1 = run_models(t)
        #UU_0_pred, VV_0_pred, UU0, VV0, UU0_t, VV0_t = results_0
        #UU_1_pred, VV_1_pred, UU1, VV1, UU1_t, VV1_t = results_1


        UU_0_pred, VV_0_pred, UU0, VV0, UU0_t, VV0_t = self.UV_0_model(t)
        UU_1_pred, VV_1_pred, UU1, VV1, UU1_t, VV1_t = self.UV_1_model(t)
        
        time1 = time.time()

        # mse0 = 0 
        # mse1 = 0
        for i in range(0,self.RKsteps):
            # Get the prediction at each input and output pair
            # [nt,q]
            
            
            
            
            u_0_pred = UU_0_pred[i]
            v_0_pred = VV_0_pred[i]
            u_1_pred = UU_1_pred[i]
            v_1_pred = VV_1_pred[i]
            
            
                
            # Get the labeled data at every input and output
            # [nw,nt]
            FROG0  = FROG_0[i]
            FROG1  = FROG_1[i]

            
            ##################################!!!orginal
            #time1 = time.time()

            #FROG1 = FROG1 * 0.4738034399768933
            #0.4735616340092876
            #0.4738034399768933
            #correction factor FROG1
            
            h0 = tf.complex(u_0_pred[:,q], v_0_pred[:,q])
            #h0 = tf.complex(tf.math.real(h0), tf.zeros_like(tf.math.real(h0)))
            
            h1 = tf.complex(u_1_pred[:,q], v_1_pred[:,q])

            #h0 = h0/10
            #h1 = h1/10


            # Assuming h0 and h1 are defined as:
            # h0 = tf.complex(u_0_pred[:,q], v_0_pred[:,q])
            # h1 = tf.complex(u_1_pred[:,q], v_1_pred[:,q])
            
            # Convert to NumPy arrays
            h0_np = h0.numpy()
            h1_np = h1.numpy()

            if epoch %10 ==0 and q == 99:      
                fig, axs = plt.subplots(2, 2, figsize=(12, 6))
                
                axs[0, 0].plot(np.real(h0_np), label='Re(h0)')
                axs[0, 0].plot(np.real(h1_np), label='Re(h1)', linestyle='--')
                axs[0, 0].set_title('Real Part')
                axs[0, 0].legend()
                
                axs[0, 1].plot(np.imag(h0_np), label='Im(h0)')
                axs[0, 1].plot(np.imag(h1_np), label='Im(h1)', linestyle='--')
                axs[0, 1].set_title('Imaginary Part')
                axs[0, 1].legend()
                
                axs[1, 0].plot(np.abs(h0_np), label='|h0|')
                axs[1, 0].plot(np.abs(h1_np), label='|h1|', linestyle='--')
                axs[1, 0].set_title('Amplitude')
                axs[1, 0].legend()
                
                axs[1, 1].plot(np.angle(h0_np), label='arg(h0)')
                axs[1, 1].plot(np.angle(h1_np), label='arg(h1)', linestyle='--')
                axs[1, 1].set_title('Phase')
                axs[1, 1].legend()

                save_dir = 'trainingimg_h'
                os.makedirs(save_dir, exist_ok=True)
                filename = 'h0andh1inloss' + str(int(epoch/10)) + '.png'
                save_path = os.path.join(save_dir, filename)
                plt.tight_layout()
                plt.savefig(save_path)
                plt.close(fig)



            #t = tf.cast(self.t, tf.float64)  # shape (64,)

            # Center both
            #h0 = self.center_complex_pulse_to_t0(h0, t)
            #h1 = self.center_complex_pulse_to_t0(h1, t)

            #h0 = tf.complex(tf.math.real(h0), tf.zeros_like(tf.math.real(h0)))

            

            

            #amplitude0 = tf.abs(h0)       # shape: [batch, time]
            #amplitude1 = tf.abs(h1)
            '''
            #phase_penalty = 0
            phase_penalty = tf.constant(0.0, dtype=tf.float64)


            if q != 99 and q!= 0:
                h0q = tf.complex(u_0_pred[:,q+1], v_0_pred[:,q+1])
                h1q = tf.complex(u_1_pred[:,q+1], v_1_pred[:,q+1])

                #h0qb = tf.complex(u_0_pred[:,q-1], v_0_pred[:,q-1])
                #h1qb = tf.complex(u_1_pred[:,q-1], v_1_pred[:,q-1])

                for a in range(64):
                    phase_penalty += tf.abs(tf.math.angle(h0[a]) - tf.math.angle(h0q[a]))
                    phase_penalty += tf.abs(tf.math.angle(h1[a]) - tf.math.angle(h1q[a]))
                
                #phase_penalty += tf.reduce_mean(tf.abs(tf.math.angle(h0) - tf.math.angle(h0qb)))
                #phase_penalty += tf.reduce_mean(tf.abs(tf.math.angle(h1) - tf.math.angle(h1qb)))

                
            if q == 99:
                h0qb = tf.complex(u_0_pred[:,q-1], v_0_pred[:,q-1])
                h1qb = tf.complex(u_1_pred[:,q-1], v_1_pred[:,q-1])

                #phase_penalty += tf.reduce_mean(tf.abs(tf.math.angle(h0) - tf.math.angle(h0qb)))
                #phase_penalty += tf.reduce_mean(tf.abs(tf.math.angle(h1) - tf.math.angle(h1qb)))

                for a in range(64):
                    phase_penalty += tf.abs(tf.math.angle(h0[a]) - tf.math.angle(h0qb[a]))
                    phase_penalty += tf.abs(tf.math.angle(h1[a]) - tf.math.angle(h1qb[a]))

            if q == 0:
                h0q = tf.complex(u_0_pred[:,q+1], v_0_pred[:,q+1])
                h1q = tf.complex(u_1_pred[:,q+1], v_1_pred[:,q+1])

                #phase_penalty += tf.reduce_mean(tf.abs(tf.math.angle(h0) - tf.math.angle(h0q)))
                #phase_penalty += tf.reduce_mean(tf.abs(tf.math.angle(h1) - tf.math.angle(h1q)))

                for a in range(64):
                    phase_penalty += tf.abs(tf.math.angle(h0[a]) - tf.math.angle(h0q[a]))
                    phase_penalty += tf.abs(tf.math.angle(h1[a]) - tf.math.angle(h1q[a]))
                

            '''
            time2 = time.time()

            #pad = 0  # number of zeros to add on each side of the time axis
            #crop = int(0.0 * (128 + 2 * pad))  # or whatever padded length is

            #h0_pad = np.pad(h0, pad_width=((0, 0), (pad, pad)), mode='constant', constant_values=0)
            #h1_pad = np.pad(h1, pad_width=((0, 0), (pad, pad)), mode='constant', constant_values=0)

            #h0 = np.pad(h0, pad_width=(pad, pad), mode='constant', constant_values=0)
            #1 = np.pad(h1, pad_width=(pad, pad), mode='constant', constant_values=0)

            #h0 = tf.pad(h0, paddings=[[pad, pad]], mode='CONSTANT')
            #h1 = tf.pad(h1, paddings=[[pad,pad]], mode ='CONSTANT')


            FROG0_q_pred = makeFROG(h0,h0,pad = 0,wcrop = 0)
            FROG1_q_pred = makeFROG(h1,h1,pad = 0,wcrop = 0)

            #correction_factor = 500.8537481523134

            
            #1265.027602492969 correction factor correct for other
            #1265.0276024929688

            #FROG0_q_pred = FROG0_q_pred*correction_factor
            #FROG1_q_pred = FROG1_q_pred*correction_factor

            '''
            autoBM = np.trapz(FROG0_q_pred, axis = 0)
            autoFROG0 = np.trapz(FROG0, axis = 0)

            plt.figure(figsize=(10, 4))

            plt.subplot(1, 2, 1)
            plt.plot(autoBM, label='autoBM')
            plt.title("Autocorrelation: BM")
            plt.xlabel("Delay index")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(autoFROG0, label='autoFROG0', color='orange')
            plt.title("Autocorrelation: FROG0")
            plt.xlabel("Delay index")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig("autocorrelations_comparison.png")
            

            

            correction_factor = autoFROG0/autoBM

            FROG0_q_pred = FROG0_q_pred * correction_factor
            FROG1_q_pred = FROG1_q_pred * correction_factor

            print("the correction factor is " + str(correction_factor))
            '''
            

            
            
            
            # Convert tensors to NumPy arrays if needed
            FROG0_q_pred_np = FROG0_q_pred.numpy()
            FROG1_q_pred_np = FROG1_q_pred.numpy()
            FROG0_np = FROG0.numpy()  # Transpose to match predicted shape
            FROG1_np = FROG1.numpy()

            # Optional: Apply log scaling to improve visibility
            # Comment this block out if you want linear intensity
            def log_scale(x):
                return np.log1p(np.abs(x))  # log(1 + |x|)
            
            #FROG0_np = log_scale(FROG0_np)
            #FROG0_q_pred_np = log_scale(FROG0_q_pred_np)
            #FROG1_np = log_scale(FROG1_np)
            #FROG1_q_pred_np = log_scale(FROG1_q_pred_np)
            
            # Plot

            if epoch % 10==0 and q==99:
                
                fig, axs = plt.subplots(2, 2, figsize=(13, 10))
                
                im0 = axs[0, 0].imshow(FROG0_np.T, aspect='auto', cmap='inferno', origin='lower')
                axs[0, 0].set_title('True FROG 0')
                cbar0 = plt.colorbar(im0, ax=axs[0, 0])
                cbar0.set_label('Log Intensity' if 'log' in log_scale.__name__ else 'Intensity')
                
                im1 = axs[0, 1].imshow(FROG0_q_pred_np, aspect='auto', cmap='inferno', origin='lower')
                axs[0, 1].set_title('Predicted FROG 0')
                cbar1 = plt.colorbar(im1, ax=axs[0, 1])
                cbar1.set_label('Log Intensity' if 'log' in log_scale.__name__ else 'Intensity')
                
                im2 = axs[1, 0].imshow(FROG1_np.T, aspect='auto', cmap='inferno', origin='lower')
                axs[1, 0].set_title('True FROG 1')
                cbar2 = plt.colorbar(im2, ax=axs[1, 0])
                cbar2.set_label('Log Intensity' if 'log' in log_scale.__name__ else 'Intensity')
                
                im3 = axs[1, 1].imshow(FROG1_q_pred_np, aspect='auto', cmap='inferno', origin='lower')
                axs[1, 1].set_title('Predicted FROG 1')
                cbar3 = plt.colorbar(im3, ax=axs[1, 1])
                cbar3.set_label('Log Intensity' if 'log' in log_scale.__name__ else 'Intensity')
                
                # Label axes
                for ax in axs.flatten():
                    ax.set_xlabel('Delay')
                    ax.set_ylabel('Frequency')
                

                save_dir = 'trainingimg'
                os.makedirs(save_dir, exist_ok=True)
                filename = 'frog_inloss' + str(int(epoch/10)) + '.png'
                save_path = os.path.join(save_dir, filename)
                plt.tight_layout()
                plt.savefig(save_path)
                plt.close(fig)

            
            time3 = time.time()
            
            # Option 1: Transpose to match shapes
            mse0 = tf.reduce_sum(tf.square(FROG0_q_pred - tf.transpose(FROG0)))

            # Option 2: Reshape if necessary
            #mse0 = tf.reduce_sum(tf.square(FROG0_q_pred - tf.reshape(FROG0, FROG0_q_pred.shape)))

            #originalcode
            #mse0 = tf.reduce_sum(tf.square(FROG0_q_pred-FROG0))
            #mse1 = tf.reduce_sum(tf.square(FROG1_q_pred-FROG1))

            #Penalize phase variations (smoothness constraint)
            #phase0 = tf.math.angle(h0)
            #phase_loss0 = tf.reduce_mean(tf.abs(phase0[1:] - phase0[:-1]))  # Finite difference
        
            #phase1 = tf.math.angle(h1)
            #phase_loss1 = tf.reduce_mean(tf.abs(phase1[1:] - phase1[:-1]))
            
            # Option 1: Transpose to match shapes
            mse1 = tf.reduce_sum(tf.square(FROG1_q_pred - tf.transpose(FROG1)))
            
            time4 = time.time()
            
            #time analysis
            #print("decomposing print function")
            #print("time of predictions " + str(time1-time0))
            #print("time of complex " + str(time2-time1))
            #print("time of makeFROG " + str(time3-time2))
            #print("time of mse" + str(time4-ti
            # h0mean = tf.complex(tf.math.reduce_mean(u_0_pred,axis = 1), tf.math.reduce_mean(v_0_pred,axis = 1))
            # h1mean = tf.complex(tf.math.reduce_mean(u_1_pred,axis = 1), tf.math.reduce_mean(v_1_pred,axis = 1))
            # FROG0_pred = makeFROG(h0mean,h0mean,pad = self.pad,wcrop = self.nw)
            # FROG1_pred = makeFROG(h1mean,h1mean,pad = self.pad,wcrop = self.nw)
            # mse0 = tf.reduce_sum(tf.square(FROG0_pred-FROG0)) + mse0
            # mse1 = tf.reduce_sum(tf.square(FROG1_pred-FROG1)) + mse1
            
            # for q in range(0, self.q):
            #     h0 = tf.complex(u_0_pred[:,q], v_0_pred[:,q])
            #     FROG0_q_pred = makeFROG(h0,h0,pad = self.pad,wcrop = self.nw)
            #     h1 = tf.complex(u_1_pred[:,q], v_1_pred[:,q])
            #     FROG1_q_pred = makeFROG(h1,h1,pad = self.pad,wcrop = self.nw)
            #     print(q)
            #     mse0 = tf.reduce_sum(tf.square(FROG0_q_pred-FROG0)) + mse0
            #     mse1 = tf.reduce_sum(tf.square(FROG1_q_pred-FROG1)) + mse1
        #if q == 99:
        #    print(f"FROG loss: {mse0+mse1}, Phase loss: {phase_penalty}")   


  
            

        #alpha = tf.stop_gradient((mse0 + mse1) / (phase_penalty + 1e-8))
        #alpha = tf.stop_gradient((mse0+mse1)*phase_penalty)
        #adaptive_weight = tf.minimum(alpha, 1.0) * 0.2  # Scale cap if needed

        #epsilon = 1e-4
        #beta = 0.2

        #epsilon = tf.constant(1e-4, dtype=tf.float64)
        #beta = tf.constant(0.2, dtype=tf.float64)
        
        # Ratio term to preserve balance
        #alpha_ratio = (mse0 + mse1 + epsilon) / (phase_penalty + epsilon)
        
        # Smooth boost term that stays in (0,1)
        #boost = tf.sigmoid(phase_penalty - 0.1)  # kicks in when penalty > 0.1
        
        # Combine and cap
        #alpha = tf.stop_gradient(tf.minimum(alpha_ratio, tf.constant(1, dtype=tf.float64)) + beta * boost)
        #adaptive_weight = tf.clip_by_value(alpha, 0.05, 1.0) * 0.2
        
        #if q == 99:
        #    print("adaptive weight " + str(adaptive_weight))

        #print(adaptive_weight)

        phase_loss_shared[q] = 0
        #phase_penalty0 + phase_penalty1
        frog_loss_shared[q] = mse0 + mse1

        #return mse0 + mse1 +  0.1 * (phase_loss0 + phase_loss1)
        #return mse0 + mse1 + 0.1*(phase_penalty)

        #if epoch %2 == 0:
        #    return mse0 + mse1
        #else:
        #    return mse0 + mse1 + 0.01*(phase_penalty)

        #if phase_penalty > 1:
        #    return mse0 + mse1 +  0.1 * (phase_penalty)
        #else:
        #    return mse0 + mse1

        
        #n = 5

        # Use tf.math.real if your h0/h1 might be complex
        #boundary_penalty = tf.reduce_mean(tf.square(tf.abs(h0[0]))) + tf.reduce_mean(tf.square(tf.abs(h0[-1])))
        #boundary_penalty += tf.reduce_mean(tf.square(tf.abs(h1[0]))) + tf.reduce_mean(tf.square(tf.abs(h1[-1])))

        
        #adaptive_weight = tf.stop_gradient(0.66 * (mse0+mse1) / (phase_penalty + 1e-8))
        
        #adaptive_weight_bound = tf.stop_gradient(0.33 * (mse0+mse1) / (boundary_penalty + 1e-8))


        #print(phase_penalty)
        #print(boundary_penalty)

        #if phase_penalty < 2:
        #    weight = 0
        #else:
        #    weight = 0.001
        '''
        def smooth_weight(phase_penalty):
            scale = 10.0  # Controls sharpness of transition
            offset = 2.5  # Center of the transition
            max_weight = 0.001
        
            sigmoid_val = tf.sigmoid(scale * (phase_penalty - offset))
            return max_weight * sigmoid_val

        weight = smooth_weight(phase_penalty)
         '''  
        #loss = mse + adaptive_weight * phase_penalty
        #0.1*(phase_loss0 + phase_loss1)
        #return mse0 + mse1 + 0.1*(phase_penalty/64.0)
        #return mse0 + mse1 + adaptive_weight*(phase_penalty) + adaptive_weight_bound*boundary_penalty

        #adaptive_weight = tf.stop_gradient(1.0 * (mse0+mse1) / (weight*phase_penalty + 50*boundary_penalty+ 1e-8))

        #if epoch %2 == 0:
        #    return mse0 + mse1
        #else:  
        #   return adaptive_weight*(weight*phase_penalty+ 50*boundary_penalty)

        #return mse0 + mse1 + weight*phase_penalty+ 50*boundary_penalty

        #phase_penalty = phase_penalty0+phase_penalty1

        #if epoch <= 10000000:
        #    return mse0 + mse1
        #else:
        #    return mse0 + mse1 + 0.1*phase_penalty
        return mse0 + mse1
            

    def phase_loss(self, all_h0r, all_h0i, all_h1r, all_h1i):
        """ 
        all_h0: List of complex h0 fields for all q values (from stage 1).
        all_h1: List of complex h1 fields for all q values.
        FROG_losses: Precomputed MSE losses for each q.
        beta: Phase penalty weight (tune as needed).
        """
        # Compute mean phase across q
        #mean_phase0 = tf.reduce_mean([tf.math.angle(h0) for h0 in all_h0], axis=0)  # [nt]
        #mean_phase1 = tf.reduce_mean([tf.math.angle(h1) for h1 in all_h1], axis=0)
        
        # Phase deviation penalty

        #all_h0r_np = np.frombuffer(all_h0r.get_obj(), dtype=np.float64).reshape((100, 64))
        #all_h0i_np = np.frombuffer(all_h0i.get_obj(), dtype=np.float64).reshape((100, 64))
        #all_h1r_np = np.frombuffer(all_h1r.get_obj(), dtype=np.float64).reshape((100, 64))
        #all_h1i_np = np.frombuffer(all_h1i.get_obj(), dtype=np.float64).reshape((100, 64))


        #all_h0 = tf.complex(tf.identity(all_h0r), all_h0i)
        #all_h1 = tf.complex(all_h1r, all_h1i)

        all_h0 = tf.complex(tf.identity(all_h0r), tf.identity(all_h0i))
        all_h1 = tf.complex(tf.identity(all_h1r), tf.identity(all_h1i))

        #print(all_h0)
        #print(all_h1)

        #all_h0 = tf.complex(tf.convert_to_tensor(all_h0r_np, dtype=tf.float64),tf.convert_to_tensor(all_h0i_np, dtype=tf.float64))
        #all_h1 = tf.complex(tf.convert_to_tensor(all_h1r_np, dtype=tf.float64),tf.convert_to_tensor(all_h1i_np, dtype=tf.float64))
        
        phase_penalty = 0
        for i in range(len(all_h0)-1):
            phase_penalty += tf.reduce_mean(tf.abs(tf.math.angle(all_h0[i]) - tf.math.angle(all_h0[i+1])))  # L1 penalty
            phase_penalty += tf.reduce_mean(tf.abs(tf.math.angle(all_h1[i]) - tf.math.angle(all_h1[i+1])))  # L1 penalty
            print(phase_penalty)
            #print(phase_penalty)
            
        #replace tf.abs with tf.square for l2 penalty
        
        # Total loss
        return phase_penalty
    
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
    def fit(self, t, FROG0, FROG1, qvalue, grad, grad_flat, counter, weights, loss_value, all_h0r, all_h0i, all_h1r, all_h1i, phase_loss_shared, frog_loss_shared, barrier, lock):
        
        #print(grad_flat)
        
        time0 = time.time()
        
        #self.logger.log_train_start(self)

        for coordinate in qvalue:
            if(coordinate == 99):
                self.logger.log_train_start(self)
                
        
        FROG_0 = []; 
        FROG_1 = [];
        # Creating the tensors
        t   = tf.convert_to_tensor(t, dtype=self.dtype)
        for i in range(0, self.RKsteps):
            FROG_0.append(tf.convert_to_tensor(FROG0[i], dtype=self.dtype)) 
            FROG_1.append(tf.convert_to_tensor(FROG1[i], dtype=self.dtype)) 
            
        # Creating dummy tensors for the gradients important because we only want to take the derivate w.r.t to the rows not the coloumns of the output
        self.dummy_t_0 = self.createDummy(t)
        self.dummy_t_1 = self.createDummy(t)
        
        
        
        #loss_value = mp.Array('d', [0.0]*100)
        #counter = mp.Value('i', 0)

        
        def log_train_epoch(epoch, loss, NeurNet, is_iter):
            
         
            
            
            # D, N, fR, Ram_res = self.get_params(numpy=True)
            # custom = f"D = {D:5f}  N = {N:5f} fR = {fR:6f}"
            # self.logger.log_train_epoch(epoch, loss, NeurNet, custom, is_iter, saveVars = True, Vars2save = [D,N,fR,Ram_res])
            D, N = self.get_params(numpy=True)
            custom = f"D = {D:5f}  N = {N:5f} "
            self.logger.log_train_epoch(epoch, loss, NeurNet, custom, is_iter, saveVars = True, Vars2save = [D,N])
    
    
        self.logger.log_train_opt("Adam")
        
        time1 = time.time()
        
        for epoch in range(hp["tf_epochs"]):
            # Optimization step
            time02 = time.time()
            loss_value, grads = self.grad(t, FROG_0, FROG_1)
            time03 = time.time()
            
            self.tf_optimizer.apply_gradients(
              zip(grads, self.wrap_training_variables()))
            
            time04 = time.time()
            
            timeoptimizer = time04-time03
            timegrad = time03-time02
            
            if epoch % 100 == 0:
                print("the optimizertime is " + str(timeoptimizer))
                print("the grad time is " + str(timegrad))
                
                
            log_train_epoch(epoch, loss_value,self, False)
        
        
        self.logger.log_train_opt("LBFGS")
        


        def loss_and_flat_grad(w, epoch, grad, grad_flat, counter,weights, loss_value, barrier, lock):
            
            
            timeMAKEtotal = 0
            timeloss0 = time.time()
          

            Vars = self.wrap_training_variables()


            #grad_flat = [] ???
            
            grad_np = []
            grad_flat_np = []
           
        
            
            for i in range(len(Vars)):

                grad_shape = Vars[i].numpy().shape
            #print("gradshape "+ i + " " + Vars[i].numpy().shape)
                grad_np.append(np.zeros(grad_shape, dtype=self.dtype))  # Use NumPy array instead of TensorFlow tensor
                grad_flat_np.append(np.reshape(grad_np[i], -1))  # Flatten the gradients for shared memory
            #print(np.reshape(grad[i], -1))

            #for cord in qvalue:
            for cord in np.atleast_1d(qvalue):

                #barrier.wait()
                
                                
                #print(f"Running iteration q={cord} for Epoch {epoch}")

                #timeloss2 = time.time()

                with tf.GradientTape() as tape:
                    self.set_weights(w)
                    tape.watch(self.D)
                    tape.watch(self.N)

                    # tape.watch(self.fR)
                    # tape.watch(self.Ram_res)
                    #this loss function is time intensive

                    timeloss2 = time.time()

                    loss_value_q = self.loss(t,cord, FROG_0, FROG_1, all_h0r, all_h0i, all_h1r, all_h1i, epoch, phase_loss_shared, frog_loss_shared)

                    #if cord != 99:
                    #    loss_value_q1 = self.loss(t,(cord+1), FROG_0, FROG_1, all_h0r, all_h0i, all_h1r, all_h1i, epoch, phase_loss_shared, frog_loss_shared)
                        


                    barrier.wait()
                    

                timeloss2middle = time.time()
                
                #print("shape of weights after setting " + str(w.shape))
 

                #loss_value = loss_value + loss_value_q
                with lock: 
                    loss_value[cord] = loss_value_q
                #######################This is a lossy line related to gradient computation

                #@tf.function
                #def compute_gradients(loss_value_q):  
                #    return tape.gradient(loss_value_q, self.wrap_training_variables())

                #grad_q = compute_gradients(loss_value_q)

                grad_q = tape.gradient(loss_value_q, self.wrap_training_variables())



                timeloss3 = time.time()

                #print("grad_flat_shape " + str(grad_flat.shape))

                #grad_flat_np = np.array(grad_flat)
                #grad_flat_np = np.array(grad_flat, dtype=object)

                #print("grad_flat_shape " + str(grad_flat_np.shape))

                #print("grad q shape " + str(len(grad_q[0])))

                #for i, g_q in zip(range(len(Vars)), grad_q):
                #    print(f"grad_flat[{i}] shape: {grad_flat_np[i].shape}")
                #    print(f"grad_q[{i}] shape: {g_q.shape}")

                with lock:  # Ensure thread-safety
                    for i, g_q in zip(range(0,len(Vars)),grad_q): # summnation of gradients

                        #update = tf.reshape(g_q, -1).numpy() 
                        #grad_flat[i] += update  # Might be safer
                        
                        ##this one is original working
                        grad_flat[i] = grad_flat[i] + tf.reshape(g_q, -1).numpy()
                        
                        
                        #np.add(grad_flat[i], tf.reshape(g_q, -1).numpy(), out=grad_flat[i])  # ATOMIC
                        #np.add(grad_flat[i], (tf.reshape(g_q, -1).numpy() / 1.0), out=grad_flat[i])

                        #reshaped_g_q = np.reshape(g_q.numpy(), grad_flat[i].shape)  # Match shapes
                        #grad_flat[i] += reshaped_g_q  # Safe summation

                        #grad_flat_np[i] += np.reshape(g_q.numpy(), -1)  # Convert to NumPy and sum

                        #update = tf.reshape(g_q, -1).numpy()
                        # Use a temporary buffer for atomic update
                        #temp = np.frombuffer(grad_flat[i].get_obj())
                        #temp += update

                #print(grad_flat)
                #del tape

                timeloss4 = time.time()

                timeloss23= timeloss3-timeloss2
                timeloss34= timeloss4-timeloss3

                timeloss2mid2 = timeloss3-timeloss2middle
                timeloss2mid1 = timeloss2middle-timeloss2

                #8print("timelossfunctionwithmake " + str(timeloss2mid1))

                timeMAKEtotal +=timeloss2mid1


           
            with lock:
                counter.value += 10
                #print("counter value " + str(counter.value))

     


            barrier.wait()
            
            if(cord == 99):
                
                print(f"Running iteration q={cord} for Epoch {epoch}")


                grad_flat_tensor = tf.concat(list(grad_flat), axis=0)

                #grad_flat = tf.concat(grad_flat, 0)

                #counter.value = 0
                loss_value_tot = 0 

                #grad_flat.clear()
                #grad.clear()

                #grad_flat[:] = []
                grad[:] = []
                
                totalphaseloss = 0
                totalfrogloss = 0
                for i in range(100):
                    totalphaseloss += phase_loss_shared[i]
                    totalfrogloss += frog_loss_shared[i] 

                
                print("phase loss " + str(totalphaseloss))

                cf = 500.8537481523134
                print("frog loss " + str(totalfrogloss*cf*cf))

                

                with lock:
                    for i in range(len(grad_flat)):
                        grad_flat[i] = np.zeros_like(grad_flat[i])  
    
                    for i in range(100):
                        loss_value_tot += loss_value[i]

                    for i in range(100):
                        loss_value[i] = 0


                #phase_loss = self.phase_loss(all_h0r, all_h0i, all_h1r, all_h1i
                
                return loss_value_tot, grad_flat_tensor


            else:
                return None
        
        time4 = time.time()
    
        if(100==100):
            
           # lbfgs(loss_and_flat_grad,
           #   self.get_weights(),
           #   self.nt_config, Struct(), True, log_train_epoch, NeurNet = self)
           # time5 = time.time()
            
            lbfgs(lambda w, epoch: loss_and_flat_grad(w, epoch, grad, grad_flat, counter, weights, loss_value, barrier, lock),
              self.get_weights(), weights, barrier, 
              self.nt_config, Struct(), True, log_train_epoch, NeurNet=self)

            #self.logger.log_train_end(self.hp['nt_epochs'],self)

            for coordinate in qvalue:
                if(coordinate == 99):
                    self.logger.log_train_end(self.hp['nt_epochs'],self)
            
            
        #grad.clear()
        #grad_flat.clear() 
        
        
        time0001 = time1-time0
        time0104 = time4-time1
        #time0405 = time5-time4
        print("time before main training loop " + str(time0001))
        print("timeforADAMtraining " + str(time0104))
        #print("timeforLBFGStraining " + str(time0405))
              
        
########################## Base Model Loss ###################################
    def loss_basemodel(self, t, uu_0, vv_0, uu_1, vv_1):
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
    
    def fit_basemodel(self, t, u0, v0, u1, v1):
        
        time0 = time.time()
        
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
            D, N,  = self.get_params(numpy=True)
            custom = f"D = {D:5f}  N = {N:5f}"
            self.logger.log_train_epoch(epoch, loss, NeurNet, custom, is_iter, saveVars = True, Vars2save = [D,N])
        
        time1 = time.time()
        self.logger.log_train_opt("Adam")
        for epoch in range(hp["tf_epochs"]):
            # Optimization step
            time02 = time.time()
            loss_value, grads = self.grad(t, u_0, v_0, u_1, v_1)
            
            time03 = time.time()
            self.tf_optimizer.apply_gradients(
              zip(grads, self.wrap_training_variables()))
            
            time04 = time.time()
            
            timeoptimizer = time04-time03
            timegrad = time03-time02
            
            
            print("the optimizertime is " + str(timeoptimizer))
            print("the grad time is " + str(timegrad))
            
            log_train_epoch(epoch, loss_value,self, False)
        
        self.logger.log_train_opt("LBFGS")
        def loss_and_flat_grad(w,epoch):
            with tf.GradientTape() as tape:
                self.set_weights(w)
                tape.watch(self.D)
                tape.watch(self.N)
                loss_value = self.loss_basemodel(t, u_0, v_0, u_1, v_1)
            grad = tape.gradient(loss_value, self.wrap_training_variables())
            grad_flat = []
            for g in grad:
                grad_flat.append(tf.reshape(g, [-1]))
            grad_flat =  tf.concat(grad_flat, 0)
            return loss_value, grad_flat
        
        time4 = time.time()
        lbfgs(loss_and_flat_grad,
          self.get_weights(),
          self.nt_config, Struct(), True, log_train_epoch, NeurNet = self)
        time5 = time.time()
        
        time0001 = time1-time0
        time0104 = time4-time1
        time0405 = time5-time4
        print("time before main training loop " + str(time0001))
        print("timeforADAMtraining " + str(time0104))
        print("timeforLBFGStraining " + str(time0405))
        
        
        
        
    def predict(self, t_star):
        t_star = tf.convert_to_tensor(t_star, dtype=self.dtype)
        t_star = t_star / 10
        
        dummy = self.createDummy(t_star)
        U_0_star, V_0_star, U_0, V_0, U_t_0, V_t_0 = self.UV_0_model(t_star, dummy)
        U_1_star, V_1_star,U_1, V_1, U_t_1, V_t_1 = self.UV_1_model(t_star, dummy)
        UV = self.model(t_star)
        U_pred = UV[:,:,0]
        V_pred = UV[:,:,1]
        h0mean = tf.complex(tf.math.reduce_mean(U_0_star[0],axis = 1), tf.math.reduce_mean(V_0_star[0],axis = 1))
        h1mean = tf.complex(tf.math.reduce_mean(U_1_star[0],axis = 1), tf.math.reduce_mean(V_1_star[0],axis = 1))
        FROG0_pred = makeFROG(h0mean,h0mean,pad = 0,wcrop = 0)
        FROG1_pred = makeFROG(h1mean,h1mean,pad = 0,wcrop = 0)
        return U_0_star, V_0_star, U_1_star, V_1_star, U_pred, V_pred, FROG0_pred,FROG1_pred
    
    def load_latest_checkpoint(self, indexnum = 'last',basemodel = False):
        super().load_latest_checkpoint(chk_point_num = str(indexnum),basemodel = basemodel)
        #super().load_latest_checkpoint(chk_point_num = str(2900),basemodel = basemodel)
          
        if basemodel:
            print('Base Model Loaded')
        else:
            self.logger.log_load_logged_data()  
            if indexnum == 'last':
                #self.D.assign([self.logger.logger_data['D'][-1]])
                #self.N.assign([self.logger.logger_data['N2'][-1]])

                self.D.assign([-0.5]) 
                self.N.assign([2.0]) 
            else:
                inx = int(indexnum/self.logger.log_checkpoint_freq)
                #self.D.assign([self.logger.logger_data['D'][-1]])
                self.D.assign([-0.5]) 
                self.N.assign([2.0]) 
                #self.N.assign([self.logger.logger_data['N2'][-1]])
            
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
def get_PINN(hp = hp, datafname = 'PINN_FROG_model_chirpedgau64_N=2.0_dist=1.mat',indexnum = 'last',ModelDirectory = os.getcwd(),D_trainable = False, N2_trainable = False, Load_Trained_model = False):
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

    print("D_Truth sim" + str(sim_hp['D']/2))
    print("N2_Truth sim" + str(sim_hp['N2']))

    
    
    # Creating the model
    logger = Logger(hp,Directory = ModelDirectory)
    logger.sim_data = sim_data
    hpBM = hp
    hpBM['NN_name'] = 'BaseModel'
    hpBM['nt_epochs'] =    hp['BM_nt_epochs']
    hpBM["log_frequency"] = hp["BM_log_frequency"]
    hpBM["log_checkpoint_freq"] = hp["BM_log_checkpoint_freq"]
    
    #hpBM['datafname'] = datafname[0:-4] +'_BaseModel.mat' 
    hpBM['datafname'] = 'PINN_FROG_model_chirpedgau64_N=2.0_dist=1'
    
    loggerBM = Logger(hpBM, Directory = logger.SaveDir)
    logger.sim_data = sim_data
    #pinn = PI_FROG(hp, logger, NN_hp, NN_hp['ub'][0], NN_hp['lb'][0], Trainable_vars = Trainable_Variables, Init_guess = (lambdas_star))
    
    sim_hpBM,sim_dataBM,NN_hpBM = prep_data(hpBM['datafname'] ,hpBM, GlobalPath = GlobalPath, RKsteps = hp['RKsteps'], N=hp["N"], SNR=hp['SNR'],q = hp['q'], IsBaseModel = True)
    lambdasBM_star = (sim_hpBM['D']/2,sim_hpBM['N2'])
    # The True parameter values
    pinnBM = PI_FROG(hpBM, loggerBM, NN_hpBM, NN_hpBM['ub'][0], NN_hpBM['lb'][0], Trainable_vars = (False, False), Init_guess = (lambdasBM_star))
    pinn = PI_FROG(hp, logger, NN_hp, NN_hp['ub'][0], NN_hp['lb'][0], Trainable_vars = Trainable_Variables, Init_guess = (lambdasBM_star))

    print("D_Truth base" + str(sim_hpBM['D']/2))
    print("N2_Truth base" + str(sim_hpBM['N2']))
    
    
    # Check if base model exists 
    # if not os.path.isdir(os.path.join(ModelDirectory,pinn.logger.Name,'BaseModel')):
    #     os.mkdir(os.path.join(ModelDirectory,pinn.logger.Name,'BaseModel'))
        
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
    #pinn.training_data = (NN_hp['FROG_0'],NN_hp['FROG_1'],NN_hp['t'],NN_hp['w'])


    #FROG_0 = np.array(NN_hp['FROG_0'], dtype=np.float64)
    #FROG_1 = np.array(NN_hp['FROG_1'], dtype=np.float64)
    
    #noise_scale = 0.0  # 1% noise
    
    #FROG_0_noisy = FROG_0 + noise_scale * np.random.normal(size=FROG_0.shape)
    #FROG_1_noisy = FROG_1 + noise_scale * np.random.normal(size=FROG_1.shape)

    Frog0path = ModelDirectory + r'/data/FROG_data_8nm_pad2_6.mat'
    Frog1path = ModelDirectory + r'/data/FROG_data_3mw_padmatch2_6.mat'

    FROG_0_content = sio.loadmat(Frog0path)
    FROG_1_content = sio.loadmat(Frog1path)

    FROG_0 = FROG_0_content['Isig']
    FROG_1 = FROG_1_content['Isig']

    newsize = 128

    time_axis = FROG_0_content['t_exp'].squeeze()
    freq_axis = FROG_0_content['f_exp'].squeeze()

    
    time_axis = time_axis*pow(10,12)
    freq_axis = freq_axis*pow(10,-12)
    
    t_span_pre = np.linspace(time_axis[0], time_axis[-1], newsize)
    f_span_pre = np.linspace(freq_axis[0], freq_axis[-1], newsize)


    

    factor = 0.2
    t_new_pre = np.linspace(factor*NN_hp['t'][0], factor*NN_hp['t'][-1], newsize)
    #t_new_pre = np.linspace(-2.5, 2.5, newsize)
    f_new_pre = np.linspace(-1*6.4, 1*6.4, newsize)


    def interpolate_delay_axis(FROG, t_old, t_new):
        # Initialize new array with new delay length but same freq bins
        N_freq = FROG.shape[0]
        N_new_delay = len(t_new)
        FROG_new = np.zeros((N_freq, N_new_delay))
    
        for i in range(N_freq):
            interp_func = interp1d(t_old, FROG[i, :], kind='cubic', bounds_error=False, fill_value=0)
            FROG_new[i, :] = interp_func(t_new)

        return FROG_new

    def interpolate_freq_axis(FROG, f_span, f_new):
        """
        Interpolate the FROG trace along the frequency axis.
        
        Parameters:
            FROG    : 2D np.array, shape (N_freq, N_delay)
            f_span  : 1D np.array, original frequency axis (in THz)
            f_new   : 1D np.array, new frequency axis (in THz)
        
        Returns:
            FROG_new: 2D np.array, shape (len(f_new), N_delay)
        """
        N_delay = FROG.shape[1]
        N_new_freq = len(f_new)
        FROG_new = np.zeros((N_new_freq, N_delay))
    
        for j in range(N_delay):
            interp_func = interp1d(f_span, FROG[:, j], kind='cubic', bounds_error=False, fill_value=0)
            FROG_new[:, j] = interp_func(f_new)
    
        return FROG_new

    # Usage:
    FROG_0_interp = interpolate_delay_axis(FROG_0, t_span_pre, t_new_pre)
    FROG_0_both = interpolate_freq_axis(FROG_0_interp, f_span_pre, f_new_pre)
    
    FROG_1_interp = interpolate_delay_axis(FROG_1, t_span_pre, t_new_pre)
    FROG_1_both = interpolate_freq_axis(FROG_1_interp, f_span_pre, f_new_pre)

    


    # --- Resize with scipy.ndimage.zoom ---
    zoom_factors = (64 / FROG_0.shape[0], 64 / FROG_0.shape[1])
    FROG_0_zoom = zoom(FROG_0_both, zoom_factors, order=3)
    FROG_1_zoom = zoom(FROG_1_both, zoom_factors, order=3)

    # size 26, 256
    # --- Resize with skimage.transform.resize ---
    FROG_0_skimg = resize(FROG_0_both, (64, 64), order=3, preserve_range=True, anti_aliasing=True)
    FROG_1_skimg = resize(FROG_1_both, (64, 64), order=3, preserve_range=True, anti_aliasing=True)

    #FROG_0_skimg = pad_frog_to_512x512(FROG_0_skimg)
    #FROG_1_skimg = pad_frog_to_512x512(FROG_1_skimg)

    # Assume FROG_0_skimg and FROG_1_skimg are (512, 256)
    #pad_left  = (512 - 256) // 2  # = 128
    #pad_right = 512 - 256 - pad_left  # = 128
    
    # Pad columns (delay axis)
    #FROG_0_skimg = np.pad(FROG_0_skimg, pad_width=((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
    #FROG_1_skimg = np.pad(FROG_1_skimg, pad_width=((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)


    
    
    # --- Plot comparison ---
    # --- Plot comparison ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    im00 = axs[0, 0].imshow(FROG_0_zoom, aspect='auto', cmap='inferno')
    axs[0, 0].set_title("FROG_0: scipy.ndimage.zoom")
    fig.colorbar(im00, ax=axs[0, 0])
    
    im01 = axs[0, 1].imshow(FROG_0_skimg, aspect='auto', cmap='inferno')
    axs[0, 1].set_title("FROG_0: skimage.transform.resize")
    fig.colorbar(im01, ax=axs[0, 1])
    
    im10 = axs[1, 0].imshow(FROG_1_zoom, aspect='auto', cmap='inferno')
    axs[1, 0].set_title("FROG_1: scipy.ndimage.zoom")
    fig.colorbar(im10, ax=axs[1, 0])
    
    im11 = axs[1, 1].imshow(FROG_1_skimg, aspect='auto', cmap='inferno')
    axs[1, 1].set_title("FROG_1: skimage.transform.resize")
    fig.colorbar(im11, ax=axs[1, 1])
    
    for ax in axs.flat:
        ax.set_xlabel('Delay axis')
        ax.set_ylabel('Frequency axis')
    
    plt.tight_layout()
    plt.savefig('FROG0and1_interpolated.png')

    FROG_0 = FROG_0_skimg
    FROG_1 =  FROG_1_skimg

    # Step 1: Integrate over delay axis to get temporal autocorrelation
    auto_0 = np.trapz(FROG_0, axis=0)  # shape: [freq]
    auto_1 = np.trapz(FROG_1, axis=0)
    

    
    # Step 2: Integrate autocorrelations to get scalar energy estimate
    U2_0 = np.trapz(auto_0)
    U2_1 = np.trapz(auto_1)
    
    # Step 3: Compute correction factor based on FROG ∝ |U|^4
    correction_factor = (U2_0 / U2_1)  # since FROG ∝ U^2
    
    # Step 4: Normalize FROG_1 so its energy matches FROG_0

    FROG_0 = FROG_0/500.8537481523134

    FROG_1 = FROG_1/500.8537481523134

    FROG_1 = FROG_1 *  0.4738034399768933


    print("correction factor applied to FROG_PM " + str(correction_factor))

    FROG_0 = FROG_0.T
    FROG_1 = FROG_1.T

    
    Frog1SSFM_data = ModelDirectory + r'/data/SSFM_N2=2.mat'

    FROG1SSFM_content = sio.loadmat(Frog1SSFM_data)



    FROG_0_train = FROG_0_content['Ishg_ret']
    #FROG_1_train = FROG_1_content['Ishg_ret']
    FROG_1_train = FROG1SSFM_content['FROG_Lz']

    newsize = 64
    newfreq = 64

    t_span = np.linspace(time_axis[0], time_axis[-1], newsize)
    f_span = np.linspace(freq_axis[0], freq_axis[-1], newsize)

    t_span = t_span

    factor = 1
    t_new = np.linspace(factor*NN_hp['t'][0], factor*NN_hp['t'][-1], newsize)

    print("first time " + str(NN_hp['t'][0]))
    print("last time " + str(NN_hp['t'][-1]))
    #t_new = np.linspace(-6.4, 6.4, newsize)
    

    plt.figure(figsize=(6,4))
    plt.title("auto0")
    plt.plot(t_new, auto_0)
    plt.savefig('auto0')
    plt.close()

    plt.figure(figsize=(6,4))
    plt.title("auto1")
    plt.plot(t_new, auto_1)
    plt.savefig('auto1')
    plt.close()

    

    

    
    FROG_0_train_skimg = resize(FROG_0_train, (newfreq, newsize), order=3, preserve_range=True, anti_aliasing=True)
    FROG_1_train_skimg = resize(FROG_1_train, (newfreq, newsize), order=3, preserve_range=True, anti_aliasing=True)

    FROG_0_train = FROG_0_train_skimg.T
    FROG_1_train = FROG_1_train_skimg.T




    #E_0 = FROG_0_content['et_ret']
    #E_1 = FROG_1_content['et_ret']

    E_0 = FROG_0_content['et_ret'].flatten()
    
    #E_1 = FROG_1_content['et_ret'].flatten()
    U_1 = FROG1SSFM_content['u_end'].flatten()
    V_1 = FROG1SSFM_content['v_end'].flatten()

    E_1 = U_1 + 1j* V_1

    

    print("np.size(NN_hp['t']) " + str(np.size(NN_hp['t'])))
    

    # === 3. Interpolate real and imaginary parts separately ===
    interp_real_0 = interp1d(time_axis, np.real(E_0), kind='cubic', bounds_error=False, fill_value=0.0)
    interp_imag_0 = interp1d(time_axis, np.imag(E_0), kind='cubic', bounds_error=False, fill_value=0.0)

    interp_real_1 = interp1d(time_axis, np.real(E_1), kind='cubic', bounds_error=False, fill_value=0.0)
    interp_imag_1 = interp1d(time_axis, np.imag(E_1), kind='cubic', bounds_error=False, fill_value=0.0)
    
    # === 4. Apply interpolation ===
    E_0 = interp_real_0(t_new) + 1j * interp_imag_0(t_new)
    E_1 = interp_real_1(t_new) + 1j * interp_imag_1(t_new)

    #x_old = np.linspace(0, 1, 128)

    #x_new = np.linspace(0, 1, newsize)

    #E_real_0_func = interp1d(x_old, np.real(E_0), kind='cubic')
    #E_real_0 = E_real_0_func(x_new)  # shape (64,)
    E_real_0 = np.real(E_0)
    E_real_0 = E_real_0.reshape(1, newsize)
    #E_real_0 = [E_real_0]

    
    #E_imag_0_func = interp1d(x_old, np.imag(E_0), kind='cubic')
    #_imag_0 = E_imag_0_func(x_new)
    
    E_imag_0 = np.imag(E_0)
    E_imag_0 = E_imag_0.reshape(1, newsize)
    #E_imag_0 = [E_imag_0]

    
    #E_real_1_func = interp1d(x_old, np.real(E_1), kind='cubic')
    #_real_1 = E_real_1_func(x_new)
    E_real_1 = np.real(E_1)
    E_real_1 = E_real_1.reshape(1, newsize)
    #E_real_1 = [E_real_1]

    
    #E_imag_1_func = interp1d(x_old, np.imag(E_1), kind='cubic')
    #E_imag_1 = E_imag_1_func(x_new)
    E_imag_1 = np.imag(E_1)
    E_imag_1 = E_imag_1.reshape(1, newsize)
    #E_imag_1 = [E_imag_1]


    

    print("tspan min " + str(time_axis[0]))
    print("tspan max " + str(time_axis[-1]))
    
    freq_axis = FROG_0_content['f_exp']

    print("FROG_0 shape " + str(FROG_0.shape))
    #print("NN_hp['FROG_0'] shape " + str(NN_hp['FROG_0'].shape))

    print("NN_hp['FROG_0'] type:", type(NN_hp['FROG_0']))
    print("NN_hp['FROG_0'] length:", len(NN_hp['FROG_0']))

    #FROG_0_NN = NN_hp['FROG_0'][0]  # Extract the 2D array from the list
    #rint("FROG_0 shape:", FROG_0_NN.shape)  # Should now be (128, 128)


    print("time_axis_shape " + str(time_axis.shape))
    print("NN_hp['t] shape " + str(NN_hp['t'].shape))

    FROG_0_NN = [FROG_0] # Now a list of length 1
    FROG_1_NN = [FROG_1]

    print("NN_hpBM['t'] " + str(NN_hpBM['t'][0].shape))
    print("NN_hpBM['u_0'] " + str(NN_hpBM['u_0'][0].shape))
    
    #NN['t] shape (64,)
    #NN_hpBM['u_0'] (64, 1)
    


    print("shapecleanh0" + str(np.real(sim_data['Clean_h0']).shape))

    #pinn.training_data = (FROG_0_train, FROG_1_train, NN_hp['t'], freq_axis)
    pinn.training_data = (FROG_0_NN, FROG_1_NN, t_span, NN_hp['w'])


    #pinn.PerfectData = (FROG_0,FROG_1,np.real(sim_data['Clean_h0']),np.imag(sim_data['Clean_h0']),np.real(sim_data['Clean_h1']),np.imag(sim_data['Clean_h1']))
    #pinn.PerfectData = (FROG_0,FROG_1,np.real(E_0),np.imag(E_0),np.real(E_1),np.imag(E_1))
    pinn.PerfectData = (FROG_0_NN,FROG_1_NN,E_real_0,E_imag_0,E_real_1,E_imag_1)

    datapath = ModelDirectory + r'/data/PINN_FROG_modelN=2.0_dist=1.mat'

    data_content = sio.loadmat(datapath)
    data_struct = data_content['data']
    
    FROG0 = data_struct['FROG0'][0,0]
    FROGLz = data_struct['FROGLz'][0,0]

    #FROG0 = [FROG0]
    #FROGLz = [FROGLz]

    



    def getFROGtruth():
        shift_amount = 0  # Left circular shift

        truth_h0r = np.roll(np.real(sim_data['Clean_h0']), shift=shift_amount, axis=0)
        truth_h0i = np.roll(np.imag(sim_data['Clean_h0']), shift=shift_amount, axis=0)
        truth_h1r = np.roll(np.real(sim_data['Clean_h1']), shift=shift_amount, axis=0)
        truth_h1i = np.roll(np.imag(sim_data['Clean_h1']), shift=shift_amount, axis=0)

        #truth_h0r = np.roll(np.real(E_real_0), shift=shift_amount, axis=0)
        #truth_h0i = np.roll(E_imag_0, shift=shift_amount, axis=0).astype(np.float64)
        #truth_h1r = np.roll(np.real(E_real_1), shift=shift_amount, axis=0)
        #truth_h1i = np.roll(E_imag_1, shift=shift_amount, axis=0).astype(np.float64)
        #truth_h0r = np.real(sim_data['Clean_h0'])
        #truth_h0i = np.imag(sim_data['Clean_h0'])
        #truth_h1r = np.real(sim_data['Clean_h1'])
        #truth_h1i = np.imag(sim_data['Clean_h1'])

        truth_h0 = tf.complex(truth_h0r, truth_h0i)
        truth_h1 = tf.complex(truth_h1r, truth_h1i)

        #padding = hp['pad']   = int((hp['nwtot']-hp['nt'])/2) 
        #print("padding " +str(padding))
        #padding = padding - 32

        #rint("padding " +str(padding))

        

        
    
        #nw = hp['nw']    = int(2*hp['WinW']/hp['dw'])

        
        plt.figure(figsize=(8, 4))
        plt.plot(truth_h0r[0])
        plt.title("Rolled Real Part of E_real_0")
        plt.xlabel("Index")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('Ereal.png')
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.plot(truth_h0i[0])
        plt.title("Rolled imag Part of E_imag_0")
        plt.xlabel("Index")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('Eimag.png')
        plt.close()

        truth_h0r_clean = np.roll(np.real(sim_data['Clean_h0']), shift=shift_amount, axis=0)

        plt.figure(figsize=(8, 4))
        plt.plot(truth_h0r_clean[0])
        plt.title("Rolled Real Part of clean_h0")
        plt.xlabel("Index")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('Ereal_clean.png')
        plt.close()
        
        

        #print("truth_h0 shape "  + str(truth_h0.shape))
        
        #pad = 192  # number of points to pad on each side
        #truth_h0_pad = np.pad(truth_h0, pad_width=pad, mode='constant', constant_values=0)
        
        #truth_h1_pad = np.pad(truth_h1, pad_width=pad, mode='constant', constant_values=0)

        pad = 0  # number of zeros to add on each side of the time axis
        crop = int(0.0 * (128 + 2 * pad))  # or whatever padded length is

        truth_h0_pad = np.pad(truth_h0, pad_width=((0, 0), (pad, pad)), mode='constant', constant_values=0)
        truth_h1_pad = np.pad(truth_h1, pad_width=((0, 0), (pad, pad)), mode='constant', constant_values=0)

        
        ## Similarly for Egate if different or same shape
         #= np.pad(Egate, pad_width=pad, mode='constant', constant_values=0)
        

        


        
        #FROG0fromh0 = makeFROG(truth_h0,truth_h0,pad = hp['pad'],wcrop = hp['nw'])
        #ROG1fromh1 = makeFROG(truth_h1,truth_h1,pad = hp['pad'], wcrop = hp['nw'])
        #FROG0fromh0 = makeFROG(truth_h0,truth_h0,pad = 0,wcrop =0)
        #FROG1fromh1 = makeFROG(truth_h1,truth_h1,pad = 0, wcrop = 0)

        FROG0fromh0 = makeFROG(truth_h0_pad,truth_h0_pad,pad = 0,wcrop = crop)
        FROG1fromh1 = makeFROG(truth_h1_pad,truth_h1_pad,pad = 0, wcrop = crop)
        


        return FROG0fromh0, FROG1fromh1, truth_h0r, truth_h0i, truth_h1r, truth_h1i
    
    def Start_PINN_basemodel_fit():
        pinnBM.fit_basemodel(NN_hpBM['t'], NN_hpBM['u_0'], NN_hpBM['v_0'],\
                  NN_hpBM['u_1'], NN_hpBM['v_1'])
    
    def Start_PINN_fit(qvalue, grad, grad_flat, counter, weights, loss_value, all_h0r, all_h0i, all_h1r, all_h1i, phase_loss_shared, frog_loss_shared, barrier, lock, loadbasemodel = False):
        if loadbasemodel: 
            pinn.load_latest_checkpoint(basemodel = True)
            #nnhp['t'] NN_hp['FROG_0']
        #pinn.fit(t_span,\
        #         FROG0,\
        #         FROGLz, qvalue, grad, grad_flat, counter, weights, loss_value, all_h0r, all_h0i, all_h1r, all_h1i, phase_loss_shared, frog_loss_shared, barrier, lock)
        
        pinn.fit(t_span,\
                 FROG_0_NN,\
                  FROG_1_NN, qvalue, grad, grad_flat, counter, weights, loss_value, all_h0r, all_h0i, all_h1r, all_h1i, phase_loss_shared, frog_loss_shared, barrier, lock)
            
    def Continue_PINN_fit(numIters = hp['nt_epochs']):
        pinn.load_latest_checkpoint()
        # Extend the epochs too current training to redefined 
        pinn.nt_config.maxIter = int(numIters+pinn.logger.epoch_ax[-1])
        pinn.nt_config.startIter = int(pinn.logger.epoch_ax[-1])
        pinn.hp['nt_epochs'] = int(numIters+pinn.logger.epoch_ax[-1])
        pinn.fit(NN_hp['t_0'], NN_hp['u_0'], NN_hp['v_0'],\
                 NN_hp['t_1'], NN_hp['u_1'], NN_hp['v_1'])
        
 
    pinn.Start_fit = Start_PINN_fit
    pinn.Start_fit_basemodel = Start_PINN_basemodel_fit
    pinn.Continue_fit = Continue_PINN_fit
    pinn.getMakeFrog = getFROGtruth
    
    if Load_Trained_model:
        pinn.load_latest_checkpoint(indexnum = indexnum)
    
    return pinn
    # TRAINING!!!!