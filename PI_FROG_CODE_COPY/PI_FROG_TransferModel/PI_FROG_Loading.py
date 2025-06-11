import sys
import json
import os
import PI_FROG as PINN
import pickle
import numpy as np
from PI_FROG_util import plot_prediction

NN_name = 'PIFROG_BaseModel'
# NN_name ='SNRINF_RK1_q400_pi2_T0150_Twin6_nt512'

# NN_name = 'DFPINF_Disc'
path = os.getcwd()
folder = os.getcwd()
with open(os.path.join( folder,NN_name, NN_name +'_hp.json')) as hpFile:
    hp = json.load(hpFile)
    
hp['NN_name'] = NN_name
pinn =  PINN.get_PINN(hp = hp,indexnum = 'last', ModelDirectory = os.path.join(os.getcwd(),folder),datafname = 'PI_FROG_SIM_v1.mat', Load_Trained_model = True)

u0p, v0p,u1p,v1p,Up,Vp,FROG0,FROG1,z,t = pinn.get_predict(numpy = True)
FROG0_sim = pinn.PerfectData[0]
FROG1_sim = pinn.PerfectData[1]
(FROG0_sim,FROG1_sim,u0_sim,v0_sim,u1_sim,v1_sim) = pinn.PerfectData
import myplots

fig, axes = myplots.myfig('4Square_DW')
xlabel= '$T$'; ylabel = '$\omega$'
myplots.myimshow(pinn.t,pinn.w, FROG0_sim[0], xlabel = xlabel, ylabel = ylabel,ax = axes[0],cbar = True)
myplots.myimshow(pinn.t,pinn.w,FROG1_sim[0], xlabel = xlabel, ylabel = ylabel,ax = axes[2],cbar = True)

myplots.myimshow(pinn.t,pinn.w,FROG0[0], xlabel = xlabel, ylabel = ylabel,ax = axes[1],cbar = True)
myplots.myimshow(pinn.t,pinn.w,FROG1[0], xlabel = xlabel, ylabel = ylabel,ax = axes[3],cbar = True)

fig, axes = myplots.myfig('4Square_DW')
lw = 1
xl = ('$t/T_0$')
yl = ('$u_0$','$v_0$','$u_1$','$v_1$')
tit = ('','','','')
alpha1 = 0.3;alpha2 = 0.5
myplots.myplot(pinn.t,u0p[0],ax=axes[0],linewidth = lw,alpha = alpha1)
myplots.myplot(pinn.t,u0_sim[0],ax=axes[0],linewidth = lw*2,linestyle='--',color = 'r',alpha = alpha2,\
               xlabel=xl,ylabel = yl[0],title = tit[0])

myplots.myplot(pinn.t,v0p[0],ax=axes[1],linewidth = lw,alpha = alpha1)
myplots.myplot(pinn.t,v0_sim[0],ax=axes[1],linewidth = lw*2,linestyle='--',color = 'r',alpha = alpha2,\
               xlabel=xl,ylabel = yl[1],title = tit[1])

myplots.myplot(pinn.t,u1p[0],ax=axes[2],linewidth = lw,alpha = alpha1)
myplots.myplot(pinn.t,u1_sim[0],ax=axes[2],linewidth = lw*2,linestyle='--',color = 'r',alpha = alpha2,\
               xlabel=xl,ylabel = yl[2],title = tit[2])

myplots.myplot(pinn.t,v1p[0],ax=axes[3],linewidth = lw,alpha = alpha1)
myplots.myplot(pinn.t,v1_sim[0],ax=axes[3],linewidth = lw*2,linestyle='--',color = 'r',alpha = alpha2,\
               xlabel=xl,ylabel = yl[3],title = tit[3])
    
myplots.savemyfig('Recovered_Pulse'+pinn.logger.Name+'.PNG')

xl = '$z/(L_D)$'
yl = '$t/(T_0)$'
fig, axes = myplots.myfig('3Vertx2')
FIELD = (Up,pinn.logger.sim_data['u_sim'].T,\
         Vp,pinn.logger.sim_data['v_sim'].T,\
         Vp**2+Up**2,(pinn.logger.sim_data['v_sim']**2+pinn.logger.sim_data['u_sim']**2).T)
titles = ('$u_p$','$u_s$','$v_p$','$v_s$','$|h|^2_p$','$|h|^2_s$')
tt = (t,pinn.logger.sim_data['t'],\
      t,pinn.logger.sim_data['t'],\
      t,pinn.logger.sim_data['t'])
zz = (z,pinn.logger.sim_data['z'],\
      z,pinn.logger.sim_data['z'],\
      z,pinn.logger.sim_data['z'])
for axs,t,z,tit,field in zip(axes,tt,zz,titles,FIELD):
    myplots.myimshow(t,z,field,aspect = 1/2.5,ax =axs,title = tit,\
                     xlabel = xl, ylabel = yl)
   
    

fig, axes = myplots.myfig('2Vert')
err, preds, truth = pinn.logger.error_fn()
myplots.myplot(np.array([0,pinn.logger.epoch_ax.max()]),np.array([truth[1],truth[1]]),ax = axes[0],linestyle = ('--'),color = 'k')
myplots.myplot((pinn.logger.epoch_ax),(pinn.logger.Vars_log['N2']),linestyle='-',marker='*',\
               yl = (0,1),ylabel = 'N2',\
                   xl = (0,pinn.logger.epoch_ax.max()),xlabel = 'epoch',\
                       xticks = (np.arange(0,pinn.logger.epoch_ax.max(),100)),\
                           ax = axes[0])
err_ep = (pinn.logger.Vars_log['N2'] - truth[1])/(truth[1])*100
myplots.myplot(pinn.logger.epoch_ax,err_ep,yl = [-10,10],\
               xl = (0,pinn.logger.epoch_ax.max()),xlabel = 'epoch',\
                   xticks = (np.arange(0,pinn.logger.epoch_ax.max(),100)))

myplots.mysuptitle('Prediction (left) and Simulation (right)')

myplots.savemyfig('Prediction vs Simulation')