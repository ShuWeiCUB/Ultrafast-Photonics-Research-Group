import sys
import json
import numpy as np
import PI_FROG as PINN
from PI_FROG_util import plot_prediction
import matplotlib.pyplot as plt


  
pinn = PINN.get_PINN(D_trainable = False, N2_trainable = True)


# Getting the model predictions
D_pred, N_pred = pinn.get_params(numpy=True)

#pinn.Start_fit_basemodel()

#pinn.Start_fit(loadbasemodel = True)
#pinn.load_latest_checkpoint(indexnum=900, basemodel = True)
pinn.Start_fit_basemodel()


u0p, v0p,u1p,v1p,Up,Vp,FROG0,FROG1,z,t = pinn.get_predict(numpy = True)
FROG0_sim = pinn.PerfectData[0]
FROG1_sim = pinn.PerfectData[1]

# pinn.load_latest_checkpoint()

# u_0_pred, v_0_pred, u_1_pred, v_1_pred, U_pred, V_pred,sim_data, z, t, lambdas_star = pinn.get_predict

# print("D: ", D_pred)
# print("N: ", N_pred)
# print("fR: ", fR_pred)
import myplots

fig, axes = myplots.myfig('4Square_DW')

xla = '$Delay (T_0)$'
yla = '$\omega (1/T_0)$'

print(pinn.w.shape)
print(pinn.t.shape)
print(FROG0_sim[0].shape)

print(pinn.w.dtype)
print(pinn.t.dtype)
print(FROG0_sim[0].dtype)




myplots.myimshow(pinn.t,pinn.w,FROG0_sim[0],ax = axes[0],cbar = True,title = '$Truth_0$',xlabel = xla,ylabel = yla)
myplots.myimshow(pinn.t,pinn.w,FROG1_sim[0],ax = axes[2],cbar = True,title = '$Truth_{L_z}$',xlabel = xla,ylabel = yla)

myplots.myimshow(pinn.t,pinn.w,FROG0[0],ax = axes[1],cbar = True,title = '$Predicted_0$',xlabel = xla,ylabel = yla)
myplots.myimshow(pinn.t,pinn.w,FROG1[0],ax = axes[3],cbar = True,title = '$Predicted_{L_z}$',xlabel = xla,ylabel = yla)

myplots.savemyfig('4Square_DW.png')

myplots.myimshow(z,pinn.w,Vp.T,cbar = True, xlabel = 'space', ylabel = 'time')

myplots.savemyfig('Vp_figure.png')

Intensity = Up**2 + Vp**2

myplots.myimshow(z, t, Intensity.T, cbar=True, xlabel = 'space', ylabel = 'time')
myplots.savemyfig('Intensity.png')

pinn.getDandN()

plt.figure(figsize=(12, 6))

u0p = np.array(u0p)
v0p = np.array(v0p)
u1p = np.array(u1p)
v1p = np.array(v1p)
t = np.array(t)
z = np.array(z)

U0sim, V0sim, U1sim, V1sim = pinn.getMakeFROG()

U0sim = np.array(U0sim)
V0sim = np.array(V0sim)
U1sim = np.array(U1sim)
V1sim = np.array(V1sim)
Up = np.array(Up)
Vp = np.array(Vp)



zvalue = 0

plt.subplot(2, 3, 1)
plt.plot(t, u0p[0, :, zvalue], label='u0p')
#plt.plot(t, Up[:,0], label='Up')
plt.plot(t, U0sim[0, :], '--', label='U0sim')
plt.title('u0p vs U0sim z= ' + str(zvalue))
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(t, v0p[0, :, zvalue], label='v0p')
#plt.plot(t, Vp[:,0], label='Vp')
plt.plot(t, V0sim[0,:], '--', label='V0sim')
plt.title('v0p vs V0sim z= ' + str(zvalue))
plt.legend()

plt.subplot(2, 3, 3)
plt.plot(t, v0p[0, :, zvalue]**2+u0p[0, :, zvalue]**2, label='Ip')
plt.plot(t, V0sim[0,:]**2+U0sim[0,:]**2, '--', label='I0sim')
plt.title('IOsim= ' + str(zvalue))
plt.legend()

zvalue = 99

plt.subplot(2, 3, 4)
plt.plot(t, u1p[0, :, zvalue], label='u1p')
plt.plot(t, U1sim[0,:], '--', label='U1sim')
plt.title('u1p vs U1sim z= ' + str(zvalue))
plt.legend()

plt.subplot(2, 3, 5)
plt.plot(t, v1p[0, :, zvalue], label='v1p')
plt.plot(t, V1sim[0,:], '--', label='V1sim')
plt.title('v1p vs V1sim z= ' + str(zvalue))
plt.legend()


plt.subplot(2, 3, 6)
plt.plot(t, v1p[0, :, zvalue]**2+u1p[0, :, zvalue]**2, label='I1p')
plt.plot(t, V1sim[0,:]**2+U1sim[0,:]**2, '--', label='I1sim')
plt.title('I1sim= ' + str(zvalue))
plt.legend()

plt.tight_layout()
plt.show()


plt.savefig('pulse_plot')


E0sim = U0sim + 1j * V0sim
E1sim = U1sim + 1j * V1sim
# FFT and shift
#E0sim_fft = np.fft.fftshift(np.fft.fft(E0sim, axis=1), axes=1)
#E1sim_fft = np.fft.fftshift(np.fft.fft(E1sim, axis=1), axes=1)

# Create Hanning window matching time dimension
#N = E0sim.shape[1]
#window = np.hanning(N)  # 1D array

# Apply window along time axis for each batch
#E0sim_windowed = E0sim * window[np.newaxis, :]
#E1sim_windowed = E1sim * window[np.newaxis, :]

# FFT and shift
#E0sim_fft = np.fft.fftshift(np.fft.fft(E0sim_windowed, axis=1), axes=1)
#E1sim_fft = np.fft.fftshift(np.fft.fft(E1sim_windowed, axis=1), axes=1)

# Choose zero-padding factor, e.g., pad to 4 times original length
N = 64
pad_factor = 4
N_pad = N * pad_factor

# Zero-pad along time axis (axis=1)
# Pad equally on both sides to keep centered
pad_left = (N_pad - N) // 2
pad_right = N_pad - N - pad_left

E0sim_padded = np.pad(E0sim, ((0, 0), (pad_left, pad_right)), mode='constant')
E1sim_padded = np.pad(E1sim, ((0, 0), (pad_left, pad_right)), mode='constant')

# Create Hanning window for padded length
window = np.hanning(N_pad)

# Apply window along time axis for each batch
E0sim_windowed = E0sim_padded * window[np.newaxis, :]
E1sim_windowed = E1sim_padded * window[np.newaxis, :]

# FFT and shift
E0sim_fft = np.fft.fftshift(np.fft.fft(E0sim_windowed, axis=1), axes=1)
E1sim_fft = np.fft.fftshift(np.fft.fft(E1sim_windowed, axis=1), axes=1)



# Frequency axis
delta_t = t[1] - t[0]
#freqs = np.fft.fftshift(np.fft.fftfreq(N, d=delta_t))
freqs = np.fft.fftshift(np.fft.fftfreq(N_pad, d=delta_t))


# Extract real and imaginary parts
U0simfft = E0sim_fft.real
V0simfft = E0sim_fft.imag
U1simfft = E1sim_fft.real
V1simfft = E1sim_fft.imag

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(freqs, U0simfft[0, :], label='Real(E0sim_fft)')
plt.title('Real part of E0sim FFT')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(freqs, V0simfft[0, :], label='Imag(E0sim_fft)', color='orange')
plt.title('Imaginary part of E0sim FFT')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(freqs, U1simfft[0, :], label='Real(E1sim_fft)')
plt.title('Real part of E1sim FFT')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(freqs, V1simfft[0, :], label='Imag(E1sim_fft)', color='orange')
plt.title('Imaginary part of E1sim FFT')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.savefig('complexfreqfield')





# pinn.get_PINN_prediction_functions()
# pinn.plot_prediction()
# # Error = pinn.plot_error(compare = False, return_ErVals = True, IsAmpNoise = True, SNR = hp['SNR'])
# # pinn.plot_error(compare = True)