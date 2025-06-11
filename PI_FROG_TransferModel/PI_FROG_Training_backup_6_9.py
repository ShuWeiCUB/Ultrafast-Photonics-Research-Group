import sys
import json
import numpy as np
import PI_FROG as PINN
from PI_FROG_util import plot_prediction
import tensorflow as tf
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
from multiprocessing import Barrier
from multiprocessing import sharedctypes

import matplotlib.pyplot as plt 

Dtrain = False

#print("Lol")
#pinn = PINN.get_PINN(D_trainable = False, N2_trainable = True)

#print(tf.config.list_physical_devices('GPU'))


# Getting the model predictions
#D_pred, N_pred = pinn.get_params(numpy=True)

#pinn.Start_fit_basemodel()

#pinn.Start_fit(loadbasemodel = True)


'''
def run_model(model_id):
    """Function to initialize and run a single model"""
    print(f"Starting model {model_id} on a separate CPU...")
    
    pinn = PINN.get_PINN(D_trainable=False, N2_trainable=True)
    D_pred, N_pred = pinn.get_params(numpy=True)
    pinn.Start_fit(loadbasemodel = True)
'''

    
    # Load model if needed
    #pinn.load_latest_checkpoint(basemodel=True)

    # Fit the model
    #pinn.fit(NN_hp['t'], NN_hp['FROG_0'], NN_hp['FROG_1'])

    # Get predictions
    #D_pred, N_pred = pinn.get_params(numpy=True)

    #return D_pred, N_pred  # Return model parameters
    
# Global variable to store the model in each worker process
'''
global_model = None  

def init_worker():
    """Initialize a separate ML model for each worker process."""
    global pinn
    pinn = PINN.get_PINN(D_trainable=False, N2_trainable=True)
    pinn.load_latest_checkpoint(basemodel=True)  # Load model only once

def run_model(model_id):
    """Function to run a single model without reloading every iteration."""
    global pinn  # Use the model initialized in the worker

    print(f"Running model {model_id} on CPU {mp.current_process().name}...")

    pinn.Start_fit(loadbasemodel = True)
    
'''

def run_model(model_id, grad, grad_flat, counter, weights, loss_value, all_h0r, all_h0i, all_h1r, all_h1i,  phase_loss_shared, frog_loss_shared, barrier, lock):
    """Each CPU initializes its own independent model and runs training."""
    print(f"Starting model {model_id} on CPU {mp.current_process().name}...")
    
    qvalue = model_id
    print("qvalue " + str(qvalue))


    # ✅ Create a new independent model inside the worker process
    pinn = PINN.get_PINN(D_trainable=Dtrain, N2_trainable=True) #Dtrain false
    pinn.load_latest_checkpoint(basemodel=True)  # Load weights for this instance
    

    # ✅ Run training independently for each process_
    #barrier.wait()  # Sync before any work
    #pinn.load_latest_checkpoint(indexnum=0)
    barrier.wait()
    pinn.Start_fit(qvalue, grad, grad_flat, counter, weights, loss_value, all_h0r, all_h0i, all_h1r, all_h1i, phase_loss_shared, frog_loss_shared, barrier,lock, loadbasemodel=True) 

    print(f"Finished model {model_id} on CPU {mp.current_process().name}")
    
    
    
    
    
    u0p, v0p,u1p,v1p,Up,Vp,FROG0,FROG1,z,t = pinn.get_predict(numpy = True)
    FROG0_sim = pinn.PerfectData[0]
    FROG1_sim = pinn.PerfectData[1]

    #pinn.PerfectData = (NN_hp['FROG_0'],NN_hp['FROG_1'],np.real(sim_data['Clean_h0']),np.imag(sim_data['Clean_h0']),np.real(sim_data['Clean_h1']),np.imag(sim_data['Clean_h1']))
    # pinn.load_latest_checkpoint()
    # u_0_pred, v_0_pred, u_1_pred, v_1_pred, U_pred, V_pred,sim_data, z, t, lambdas_star = pinn.get_predict

    # print("D: ", D_pred)
    # print("N: ", N_pred)
    # print("fR: ", fR_pred)
    import myplots

    if np.any(qvalue == 99):
        fig, axes = myplots.myfig('4Square_DW')
    
        xla = '$Delay (T_0)$'
        yla = '$\omega (1/T_0)$'
        myplots.myimshow(pinn.t,pinn.w,FROG0_sim[0].T,ax = axes[0],cbar = True,title = '$Truth_0$',xlabel = xla,ylabel = yla)
        myplots.myimshow(pinn.t,pinn.w,FROG1_sim[0].T,ax = axes[2],cbar = True,title = '$Truth_{L_z}$',xlabel = xla,ylabel = yla)
    
        myplots.myimshow(pinn.t,pinn.w,FROG0[0].T,ax = axes[1],cbar = True,title = '$Predicted_0$',xlabel = xla,ylabel = yla)
        myplots.myimshow(pinn.t,pinn.w,FROG1[0].T,ax = axes[3],cbar = True,title = '$Predicted_{L_z}$',xlabel = xla,ylabel = yla)

        #transpose to match shape
        #myplots.myimshow(pinn.w, pinn.t, FROG0[0].T, ax=axes[1], cbar=True, title='$Predicted_0$', xlabel=yla, ylabel=xla)
        #myplots.myimshow(pinn.w, pinn.t, FROG1[0].T, ax=axes[3], cbar=True, title='$Predicted_{L_z}$', xlabel=yla, ylabel=xla)
        
        myplots.savemyfig('4Square_DW.png')
    
    
        myplots.myimshow(z,pinn.w,Vp.T,cbar = True, title = 'Vp Figure', xlabel = 'z', ylabel = 'frequency')

        print("Shape of Vp.T:", Vp.T.shape)
        print("Width (x-axis, len(z)):", Vp.T.shape[1])
        print("Length (y-axis, len(pinn.w)):", Vp.T.shape[0])

        
        myplots.savemyfig('Vp_figure.png')

        Intensity = Up**2 + Vp**2

        myplots.myimshow(z, t, Intensity.T, cbar=True, title = 'Intensity',xlabel = 'z', ylabel = 'time')
        myplots.savemyfig('Intensity.png')


        
        fig, axes = myplots.myfig('4Square_DW')

        xla = '$Delay (T_0)$'
        yla = '$\omega (1/T_0)$'
        myplots.myimshow(pinn.t,pinn.w,FROG0_sim[0].T,ax = axes[0],cbar = True,title = '$SimTruth_0$',xlabel = xla,ylabel = yla)
        myplots.myimshow(pinn.t,pinn.w,FROG1_sim[0].T,ax = axes[2],cbar = True,title = '$SimTruth_{L_z}$',xlabel = xla,ylabel = yla)

        FROG0fromh0, FROG1fromh1, U0sim, V0sim, U1sim, V1sim = pinn.getMakeFrog()


        #FrogError0 = FROG0fromh0 - FROG0_sim[0]
        #FrogError1 = FROG1fromh1 - FROG1_sim[0]
        
        myplots.myimshow(pinn.t,pinn.w,tf.transpose(FROG0fromh0),ax = axes[1],cbar = True,title = '$fromh0truth_0$',xlabel = xla,ylabel = yla)
        myplots.myimshow(pinn.t,pinn.w,tf.transpose(FROG1fromh1),ax = axes[3],cbar = True,title = '$fromh1truth_{L_z}$',xlabel = xla,ylabel = yla)

        #myplots.myimshow(pinn.t,pinn.w,FrogError0,ax = axes[1],cbar = True,title = '$FrogError0$',xlabel = xla,ylabel = yla)
        #myplots.myimshow(pinn.t,pinn.w,FrogError1,ax = axes[3],cbar = True,title = '$FrogError1_{L_z}$',xlabel = xla,ylabel = yla)

        myplots.savemyfig('4Square_DW_makefrogtest')


        def normalize(F):
            return F / np.linalg.norm(F)

        FROG0fromh0_norm = normalize(FROG0fromh0)
        FROG1fromh1_norm = normalize(FROG1fromh1)
        FROG0_sim_norm   = normalize(FROG0_sim[0])
        FROG1_sim_norm   = normalize(FROG1_sim[0])
        FROG0_norm = normalize(FROG0[0])
        FROG1_norm = normalize(FROG1[0])
        
        
        FrogError0 = FROG0fromh0_norm - FROG0_sim_norm
        FrogError1 = FROG1fromh1_norm - FROG1_sim_norm


        # Assuming pinn.t and pinn.w are your time and frequency axes
        T, W = np.meshgrid(pinn.t, pinn.w)
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        # Plot FrogError0
        im0 = axes[0].imshow(FrogError0, extent=(pinn.t.min(), pinn.t.max(), pinn.w.min(), pinn.w.max()),
                             aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_title('FROG Error 0')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Frequency')
        plt.colorbar(im0, ax=axes[0])
        
        # Plot FrogError1
        im1 = axes[1].imshow(FrogError1, extent=(pinn.t.min(), pinn.t.max(), pinn.w.min(), pinn.w.max()),
                             aspect='auto', origin='lower', cmap='viridis')
        axes[1].set_title('FROG Error 1')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Frequency')
        plt.colorbar(im1, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig('makeFROG_error')
        #plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))


        
        
        FrogError0 = FROG0_norm - FROG0_sim_norm
        FrogError1 = FROG1_norm - FROG1_sim_norm
        
        # Plot FrogError0
        im0 = axes[0].imshow(FrogError0, extent=(pinn.t.min(), pinn.t.max(), pinn.w.min(), pinn.w.max()),
                             aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_title('FROG Error 0')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Frequency')
        plt.colorbar(im0, ax=axes[0])
        
        # Plot FrogError1
        im1 = axes[1].imshow(FrogError1, extent=(pinn.t.min(), pinn.t.max(), pinn.w.min(), pinn.w.max()),
                             aspect='auto', origin='lower', cmap='viridis')
        axes[1].set_title('FROG Error 1')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Frequency')
        plt.colorbar(im1, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig('makeFROG_error from basemodel')
        #plt.show()

        
        plt.figure(figsize=(12, 6))

        u0p = np.array(u0p)
        v0p = np.array(v0p)
        u1p = np.array(u1p)
        v1p = np.array(v1p)
        t = np.array(t)
        z = np.array(z)

        U0sim = np.array(U0sim)
        V0sim = np.array(V0sim)
        U1sim = np.array(U1sim)
        V1sim = np.array(V1sim)
        Up = np.array(Up)
        Vp = np.array(Vp)

        print("t shape:", np.shape(t))
        print("z shape:", np.shape(z)) # Expect something like (N,)
        print("u0p shape:", np.shape(u0p))   # Should be (len(t), len(z)) or similar
        print("u1p shape:", np.shape(u1p))   # Should be (len(t), len(z)) or similar
        print("U0sim shape:", np.shape(U0sim))   # Should be (len(t), len(z)) or similar
        print("Up shape:", np.shape(Up))   # Should be (len(t), len(z)) or similar

        zvalue = 0

        plt.subplot(2, 2, 1)
        plt.plot(t, u0p[0, :, zvalue], label='u0p')
        #plt.plot(t, Up[:,0], label='Up')
        plt.plot(t, U0sim[0, :], '--', label='U0sim')
        plt.title('u0p vs U0sim z= ' + str(zvalue))
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(t, v0p[0, :, zvalue], label='v0p')
        #plt.plot(t, Vp[:,0], label='Vp')
        plt.plot(t, V0sim[0,:], '--', label='V0sim')
        plt.title('v0p vs V0sim z= ' + str(zvalue))
        plt.legend()

        zvalue = 99
        
        plt.subplot(2, 2, 3)
        plt.plot(t, u1p[0, :, zvalue], label='u1p')
        plt.plot(t, U1sim[0,:], '--', label='U1sim')
        plt.title('u1p vs U1sim z= ' + str(zvalue))
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.plot(t, v1p[0, :, zvalue], label='v1p')
        plt.plot(t, V1sim[0,:], '--', label='V1sim')
        plt.title('v1p vs V1sim z= ' + str(zvalue))
        plt.legend()
        
        plt.tight_layout()
        plt.show()

        plt.savefig('pulse_plot')
        
                

    
    
    #plt.show()





if __name__ == "__main__":
    
    mp.set_start_method("spawn", force =True)
    
    num_cpus = min(100, cpu_count())  # Use up to 100 CPUs or the available count
    print(f"Using {num_cpus} CPUs for parallel execution.")
    
    with mp.Manager() as manager:
        grad = manager.list()  # Shared list for flattened gradients
        
        
        grad_flat = manager.list()  # Shared list for flattened gradients
        
        grad_flat.append(np.zeros(100, dtype=np.float64))  # Use NumPy array instead of TensorFlow tensor
        grad_flat.append(np.zeros(100, dtype=np.float64))  # Use NumPy array instead of TensorFlow tensor
        grad_flat.append(np.zeros(10000, dtype=np.float64))  # Use NumPy array instead of TensorFlow tensor
        grad_flat.append(np.zeros(100, dtype=np.float64))  # Use NumPy array instead of TensorFlow tensor
        grad_flat.append(np.zeros(10000, dtype=np.float64))  # Use NumPy array instead of TensorFlow tensor
        grad_flat.append(np.zeros(100, dtype=np.float64))  # Use NumPy array instead of TensorFlow tensor
        grad_flat.append(np.zeros(10000, dtype=np.float64))  # Use NumPy array instead of TensorFlow tensor
        grad_flat.append(np.zeros(100, dtype=np.float64))  # Use NumPy array instead of TensorFlow tensor
        grad_flat.append(np.zeros(20000, dtype=np.float64))  # Use NumPy array instead of TensorFlow tensor
        grad_flat.append(np.zeros(200, dtype=np.float64))  # Use NumPy array instead of TensorFlow tensor
        grad_flat.append(np.zeros(1, dtype=np.float64))  # Use NumPy array instead of TensorFlow tensor

        if Dtrain:
            grad_flat.append(np.zeros(1, dtype=np.float64))  # Use NumPy array instead of TensorFlow tensor
            
        
        
        '''
        grad_flat = [
        sharedctypes.RawArray('f', 100),      # 100 elements, float32
        sharedctypes.RawArray('f', 100),      
        sharedctypes.RawArray('f', 10000),    
        sharedctypes.RawArray('f', 100),      
        sharedctypes.RawArray('f', 10000),    
        sharedctypes.RawArray('f', 100),      
        sharedctypes.RawArray('f', 10000),    
        sharedctypes.RawArray('f', 100),      
        sharedctypes.RawArray('f', 200000),   
        sharedctypes.RawArray('f', 200),      
        sharedctypes.RawArray('f', 1)        
        ]
        '''
        



        counter = manager.Value('i', 0)

        if Dtrain:
            weights = mp.Array('d', np.zeros(50702))
        else:
            weights = mp.Array('d', np.zeros(50701))
            
            
         #mp Array is faster than manager.Array
        #weights = mp.Array('f', np.zeros(50701, dtype=np.float32))  # 'f' for float32

        loss_value = manager.Array('d', [0.0]*100)
        
        phase_loss_shared = manager.Array('d', [0.0]*100)
        frog_loss_shared = manager.Array('d', [0.0]*100)
        #bound_loss_shared = manager.Array('d', [0.0]*100)

        all_h0r = mp.Array('d', [0.0]*(100*64))
        all_h0i = mp.Array('d', [0.0]*(100*64))

        all_h1r = mp.Array('d', [0.0]*(100*64))
        all_h1i = mp.Array('d', [0.0]*(100*64))


        #all_h0r = tf.Variable(np.frombuffer(all_h0r_mp.get_obj(), dtype=np.float64).reshape(100, 64), trainable=False)
        #all_h0i = tf.Variable(np.frombuffer(all_h0i_mp.get_obj(), dtype=np.float64).reshape(100, 64), trainable=False)
        #all_h1r = tf.Variable(np.frombuffer(all_h1r_mp.get_obj(), dtype=np.float64).reshape(100, 64), trainable=False)
        #all_h1i = tf.Variable(np.frombuffer(all_h1i_mp.get_obj(), dtype=np.float64).reshape(100, 64), trainable=False)

        #all_h0r = tf.Variable(initial_value=np.zeros((100, 64), dtype=np.float64),trainable=False,synchronization=tf.VariableSynchronization.ON_READ,aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        #all_h0i = tf.Variable(initial_value=np.zeros((100, 64), dtype=np.float64),trainable=False,synchronization=tf.VariableSynchronization.ON_READ,aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        #all_h1r = tf.Variable(initial_value=np.zeros((100, 64), dtype=np.float64),trainable=False,synchronization=tf.VariableSynchronization.ON_READ,aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        #all_h1i = tf.Variable(initial_value=np.zeros((100, 64), dtype=np.float64),trainable=False,synchronization=tf.VariableSynchronization.ON_READ,aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

        
        #loss_value = manager.Array('f', np.zeros(100, dtype=np.float32))



        lock = mp.Lock()  # Lock for safe access
        
        
        
        qvalue = np.zeros((25,4), dtype=int)
        
        for i in range(25):
            for a in range(4):
                qvalue[i,a] = int(i + a*25)
                
        print(qvalue)
        
        for cord in qvalue[0, :]:
            print(cord)
            
        for cord in qvalue[1, :]:
            print(cord)
            
       
        
        #qvalue = np.arange(100, dtype=int)
            
        
            
        num_workers = 25  # Number of parallel processes
        barrier = Barrier(num_workers)  # Creates a synchronization point
        
        processes = []
        
        for i in range(num_workers):
            p = mp.Process(target=run_model, args= (qvalue[i, :], grad, grad_flat, counter, weights, loss_value, all_h0r, all_h0i, all_h1r, all_h1i, phase_loss_shared, frog_loss_shared, barrier, lock)) 
            #p = mp.Process(target=run_model, args= (qvalue[i], grad, grad_flat, counter, weights, loss_value, barrier)) 

            processes.append(p)
            p.start()
            
        for p in processes:
            p.join()

            

        #with Pool(processes=10) as pool:
            #pool.map(run_model(num_cpus, grad, grad_flat), range(num_cpus))  # Run models in parallel
        #    pool.starmap(run_model, [(qvalue[i, :], grad, grad_flat, counter, weights, loss_value, barrier) for i in range(10)])  # ✅ Correct
        
        
 

    
    #pinn = PINN.get_PINN(D_trainable=False, N2_trainable=True)
    #pinn.load_latest_checkpoint(basemodel=True)  # Load model only once


    
    #with Pool(processes=num_cpus, initializer=init_worker) as pool:
    #    results = pool.map(run_model, range(num_cpus)) 


    print("All models finished execution.")




# pinn.get_PINN_prediction_functions()
# pinn.plot_prediction()
# # Error = pinn.plot_error(compare = False, return_ErVals = True, IsAmpNoise = True, SNR = hp['SNR'])
# # pinn.plot_error(compare = True)