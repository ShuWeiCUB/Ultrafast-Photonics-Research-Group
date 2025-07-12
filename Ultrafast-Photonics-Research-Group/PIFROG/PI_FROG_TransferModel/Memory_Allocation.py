# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:46:20 2024

@author: ECEE1B79
"""

# Memory testing 

import tensorflow as tf
import numpy as np
import sys
import os
import time
import myplots
import matplotlib.pyplot as plt
GlobalPath = os.getcwd()
sys.path.append(os.path.join(GlobalPath,'..', "utils"))
from Make_FROG import makeFROG

from tqdm import trange
from time import sleep

for i in trange(10, desc='1st loop'):
    for j in trange(5, desc='2nd loop', leave=False):
        for k in trange(100, desc='3nd loop'):
            sleep(0.01)

for i in range(0,100):
    print(f"{i/100:.0%}")
    time.sleep(0.1)
    #print('\b', end="\r", flush=True) 
    sys.stdout.write("\033[K")
    time.sleep(0.1)
    # sys.stdout.write('\010')
    

# nx = 2**6
# dx = 10/nx
# dX = np.pi/(dx*nx)
# xx = np.arange(-nx/2,nx/2)*dx
# XX = np.arange(-nx/2,nx/2)*dX
# yy = np.exp(-(xx**2))

# Tyy = tf.convert_to_tensor(yy,dtype = 'complex128')

# with tf.GradientTape() as tape:
#     tape.watch(Tyy)
#     FROG = makeFROG(yy,yy) #[x, X]
    
# tape.gradient(FROG, sources)
# myplots.myimshow(XX,xx, FROG, xl = (-2,2), yl = (-2,2))
# myplots.savemyfig()
    
    
    
    
