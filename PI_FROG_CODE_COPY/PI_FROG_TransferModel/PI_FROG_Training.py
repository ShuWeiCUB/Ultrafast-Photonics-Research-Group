import sys
import json
import numpy as np
import PI_FROG as PINN
from PI_FROG_util import plot_prediction


  
pinn = PINN.get_PINN(D_trainable = False, N2_trainable = True)


# Getting the model predictions
D_pred, N_pred = pinn.get_params(numpy=True)

#pinn.Start_fit_basemodel()

#pinn.Start_fit(loadbasemodel = True)
pinn.load_latest_checkpoint(indexnum=4995)


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
myplots.myimshow(pinn.t,pinn.w,FROG0_sim[0].T,ax = axes[0],cbar = True,title = '$Truth_0$',xlabel = xla,ylabel = yla)
myplots.myimshow(pinn.t,pinn.w,FROG1_sim[0].T,ax = axes[2],cbar = True,title = '$Truth_{L_z}$',xlabel = xla,ylabel = yla)

myplots.myimshow(pinn.t,pinn.w,FROG0[0],ax = axes[1],cbar = True,title = '$Predicted_0$',xlabel = xla,ylabel = yla)
myplots.myimshow(pinn.t,pinn.w,FROG1[0],ax = axes[3],cbar = True,title = '$Predicted_{L_z}$',xlabel = xla,ylabel = yla)

myplots.savemyfig('4Square_DW.png')

myplots.myimshow(z,pinn.w,Vp.T,cbar = True, xlabel = 'space', ylabel = 'time')

myplots.savemyfig('Vp_figure.png')

Intensity = Up**2 + Vp**2

myplots.myimshow(z, t, Intensity.T, cbar=True, xlabel = 'space', ylabel = 'time')
myplots.savemyfig('Intensity.png')

pinn.getDandN()





# pinn.get_PINN_prediction_functions()
# pinn.plot_prediction()
# # Error = pinn.plot_error(compare = False, return_ErVals = True, IsAmpNoise = True, SNR = hp['SNR'])
# # pinn.plot_error(compare = True)