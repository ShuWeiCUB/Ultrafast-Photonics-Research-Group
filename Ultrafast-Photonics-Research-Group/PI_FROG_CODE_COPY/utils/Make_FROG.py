import numpy as np
import tensorflow as tf
def makeFROG(Esig, Egate, pad = 0, wcrop = 0):
    Esig0 = Esig
    Egate0 = Egate
    if wcrop % 2 == 1:
        wcrop = wcrop+1
    if type(Esig) == type(np.array([])):
        n = len(Esig.squeeze())
        Esig  = np.expand_dims(np.pad(Esig.squeeze(),pad),axis = 1)
        N = len(Esig.squeeze())
        Egate = np.expand_dims(np.pad(Egate.squeeze(),pad),axis = 1)
        
        cFROG = Esig*Egate.T
        
        for i in range(1,len(Esig)):
            cFROG[i,:] = np.roll(cFROG[i,:],shift = -i)

            #cFROG[i,:] = np.roll(cFROG[i,:],shift = -2+i)
            
        
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
        n = len(tf.squeeze(Esig))
        paddings  = tf.constant([[pad,pad]])
        Esig2 = tf.expand_dims(tf.pad(tf.squeeze(Esig), paddings = paddings), axis=1)
        N = len(tf.squeeze(Esig2))
        Egate2 = tf.expand_dims(tf.pad(tf.squeeze(Egate), paddings = paddings), axis=1)
        cFROG_temp = Esig2*tf.transpose(Egate2)
        cFROG = cFROG_temp[0:1,:]
        for i in range(1,len(Esig2)):
            cFROG = tf.concat([cFROG,tf.roll(cFROG_temp[i:i+1,:],shift = -i,axis = 1)],axis = 0)

        
        
        ax = 1
        #cFROG = tf.signal.fftshift(tf.signal.fft(tf.transpose(tf.signal.fftshift(cFROG,axes = ax))),axes = ax)/n
        cFROG = tf.transpose(tf.signal.fftshift(tf.signal.fft(tf.transpose(tf.signal.fftshift(cFROG, axes=ax))), axes=ax)) / n
        # cFROG = tf.signal.fftshift(tf.signal.fft(tf.signal.fftshift(cFROG,axes = ax)),axes = ax)/n
        if wcrop ==0:
            FROG = tf.transpose(tf.square(tf.abs(cFROG[pad:N-pad,:])))
        else:
            mid = int(cFROG.shape[1]/2)
            FROG = tf.transpose(tf.square(tf.abs(\
                                                 cFROG[pad:N-pad,int(mid-wcrop/2):int(mid+wcrop/2)])))
            
    return FROG