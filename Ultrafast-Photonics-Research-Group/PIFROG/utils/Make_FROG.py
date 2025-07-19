import numpy as np
import tensorflow as tf

import time

import multiprocessing as mp
import numpy as np
import tensorflow as tf

#def roll_row(args):
#    row, shift = args
#    return np.roll(row, shift=shift, axis=0)

#def roll_rows_with_multiprocessing(rows, shifts):
#    """Perform rolling of rows using multiprocessing."""
#    with mp.Pool(processes=mp.cpu_count()) as pool:
#        rolled_rows = pool.map(roll_row, zip(rows, shifts))
#    return rolled_rows

#converts function into a tensorflow graph and uses vectorized operations to roll each row in parallel

@tf.function
def optimized_cFROG(cFROG_temp, Esig2):

    #print("tf.shape(Esig2)[0]) " + str(tf.shape(Esig2)[0]))
    #tf.print("Esig2 shape:", tf.shape(Esig2))

    def roll_fn(i):
        return tf.roll(cFROG_temp[i:i+1, :], shift=-i, axis=1)
    
    cFROG = tf.vectorized_map(roll_fn, tf.range(tf.shape(Esig2)[0]))
    cFROG = tf.ensure_shape(cFROG, [64, None, 64])
        
    return tf.concat(tf.unstack(cFROG), axis=0)

'''
@tf.function
def optimized_cFROG(cFROG_temp, Esig2):
    N = tf.shape(Esig2)[0]
    print(N)
    delay_indices = tf.range(-N // 2, N // 2)

    def roll_fn(i):
        shift = delay_indices[i]
        return tf.roll(cFROG_temp[i:i+1, :], shift=shift, axis=1)
    
    cFROG = tf.vectorized_map(roll_fn, tf.range(N))
    cFROG = tf.ensure_shape(cFROG, [512, None, 512])  # Optional: make this dynamic

    return tf.concat(tf.unstack(cFROG), axis=0)
'''


def makeFROG(Esig, Egate, pad = 0, wcrop = 0):

    pad = 0
    

    #print("shape Esig " + str(Esig.shape))
    
    #print("test")
    
    time0 = time.time()
    
    Esig0 = Esig
    Egate0 = Egate
    if wcrop % 2 == 1:
        wcrop = wcrop+1
    if type(Esig) == type(np.array([])):

        if pad > 0: 
            n = len(Esig.squeeze())
            Esig  = np.expand_dims(np.pad(Esig.squeeze(),pad),axis = 1)
            N = len(Esig.squeeze())
            Egate = np.expand_dims(np.pad(Egate.squeeze(),pad),axis = 1)
        elif pad == 0:
            n = len(Esig.squeeze())
            Esig = np.expand_dims(np.squeeze(Esig), axis=1)
            N = len(Esig.squeeze())
            Egate = np.expand_dims(np.squeeze(Egate), axis=1)
            
        
        cFROG = Esig*Egate.T
        
        for i in range(1,len(Esig)):
            #cFROG[i,:] = np.roll(cFROG[i,:],shift = -2+i)
            cFROG[i,:] = np.roll(cFROG[i,:],shift = -i)
        
        ax = 0
       # cFROG = np.fft.fftshift(np.fft.fft(np.fft.fftshift(cFROG, axes = ax),axis = ax), axes = ax)/n
        # cFROG = np.fft.fftshift(cFROG,axes = 1)
        # cFROG = np.fft.fft(np.fft.fftshift(cFROG, axes = ax),axis = ax)
        
        cFROG = np.fft.fftshift(np.fft.fft(np.fft.fftshift(cFROG),axis = ax), axes = ax)/n
        
        if wcrop == 0: 
            FROG  = (np.abs(cFROG[pad:N-pad,:])**2) # Output is [nw,ndelay]
        else:
            mid = int(cFROG.shape[1]/2)
            FROG = (np.abs(cFROG[pad:N-pad,int(mid-wcrop/2):int(mid+wcrop/2)])**2)
        
    
    if type(Esig) == type(tf.constant([])):

        pad = 0
        
        n = len(tf.squeeze(Esig))
        #print("n " + str(n))
        #print("shape Esig2" + str(Esig2.shape))



        if pad > 0:
            paddings  = tf.constant([[pad,pad]])
            Esig2 = tf.expand_dims(tf.pad(tf.squeeze(Esig), paddings = paddings), axis=1)
            Egate2 = tf.expand_dims(tf.pad(tf.squeeze(Egate), paddings = paddings), axis=1)
        elif pad ==0:
            Esig2 = tf.expand_dims(tf.squeeze(Esig), axis=1)
            Egate2 = tf.expand_dims(tf.squeeze(Egate), axis=1)

        N = len(tf.squeeze(Esig2))

        
        
        cFROG_temp = Esig2*tf.transpose(Egate2)
        
  

        time1 = time.time()
    
        #cFROG = cFROG_temp[0:1,:]

        #for i in range(1,len(Esig2)):
        #    cFROG = tf.concat([cFROG,tf.roll(cFROG_temp[i:i+1,:],shift = -2-i,axis = 1)],axis = 0)
  
           
        #################
        cFROG = optimized_cFROG(cFROG_temp, Esig2)
        ################
     
 
                                                       
        ax = 1
        
        time2 = time.time()
        
        #cFROG = tf.signal.fftshift(tf.signal.fft(tf.transpose(tf.signal.fftshift(cFROG,axes = ax))),axes = ax)/n
        #cFROG = tf.signal.fftshift(tf.signal.ifft(tf.transpose(tf.signal.ifftshift(cFROG, axes=ax))), axes=ax) * N
        #cFROG = tf.signal.fftshift(tf.signal.ifft(tf.signal.ifftshift(cFROG, axes=ax)), axes=ax) * N
        #cFROG = tf.transpose(tf.signal.fftshift(tf.signal.ifft(tf.transpose(tf.signal.ifftshift(cFROG, axes=ax))), axes=ax))  #working!
        cFROG = tf.transpose(tf.signal.fftshift(tf.signal.fft(tf.transpose(tf.signal.fftshift(cFROG, axes=ax))), axes=ax)) / n #working?

        #cFROG = tf.transpose(tf.signal.fftshift(tf.signal.ifft(tf.signal.ifftshift(cFROG, axes=ax)), axes=ax)) * N
        #cFROG = tf.transpose(tf.signal.fftshift(tf.transpose(tf.signal.ifft(tf.transpose(tf.signal.ifftshift(cFROG, axes=ax))), axes=ax))) * N
        #cFROG = tf.signal.fftshift(tf.transpose(tf.signal.ifft(tf.transpose(tf.signal.ifftshift(cFROG, axes=ax))), axes=ax)) * N


        




        # cFROG = tf.signal.fftshift(tf.signal.fft(tf.signal.fftshift(cFROG,axes = ax)),axes = ax)/n
        if wcrop ==0:
            FROG = tf.square(tf.abs(cFROG[pad:N-pad,:]))
        else:
            mid = int(cFROG.shape[1]/2)
            FROG = tf.square(tf.abs(cFROG[pad:N-pad,int(mid-wcrop/2):int(mid+wcrop/2)]))
        time3 = time.time()
        
        time01 = time1-time0
        time12 = time2-time1
        time23 = time3-time2
        
        #print("time before tensor roll " + str(time01))
        #print("time of tensor roll " + str(time12))
        #print("after tensor roll " + str(time23))
        
     
        
            
    return FROG