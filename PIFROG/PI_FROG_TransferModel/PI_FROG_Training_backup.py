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

Dtrain = True


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

def run_model(model_id, grad, grad_flat, counter, weights, loss_value, barrier, lock):
    """Each CPU initializes its own independent model and runs training."""
    print(f"Starting model {model_id} on CPU {mp.current_process().name}...")
    
    qvalue = model_id
    print("qvalue " + str(qvalue))

    # ✅ Create a new independent model inside the worker process
    pinn = PINN.get_PINN(D_trainable=False, N2_trainable=True)
    pinn.load_latest_checkpoint(basemodel=True)  # Load weights for this instance
    

    # ✅ Run training independently for each process
    
    barrier.wait()
    pinn.Start_fit(qvalue, grad, grad_flat, counter, weights, loss_value, barrier,lock, loadbasemodel=True) 

    print(f"Finished model {model_id} on CPU {mp.current_process().name}")
    
    
    
    
    
    u0p, v0p,u1p,v1p,Up,Vp,FROG0,FROG1,z,t = pinn.get_predict(numpy = True)
    FROG0_sim = pinn.PerfectData[0]
    FROG1_sim = pinn.PerfectData[1]
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
    
        myplots.myimshow(pinn.t,pinn.w,FROG0[0],ax = axes[1],cbar = True,title = '$Predicted_0$',xlabel = xla,ylabel = yla)
        myplots.myimshow(pinn.t,pinn.w,FROG1[0],ax = axes[3],cbar = True,title = '$Predicted_{L_z}$',xlabel = xla,ylabel = yla)
        
        myplots.savemyfig('4Square_DW.png')
    
    
        myplots.myimshow(z,pinn.w,Vp.T,cbar = True)
        
        myplots.savemyfig('Vp_figure.png')

    
    
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
            grad_flat.append(np.zeros(1, dtype=np.float64))
        
        
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
            p = mp.Process(target=run_model, args= (qvalue[i, :], grad, grad_flat, counter, weights, loss_value, barrier, lock)) 
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