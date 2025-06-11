import json
import numpy as np
# Default parameters
hp = {}
# Data size on the solution u
hp["N"] = 2**9
# DeepNN topology (1-sized input [x], 3 hidden layer of 50-width, q-sized output defined later [u_1^n(x), ..., u_{q+1}^n(x)]
hp["layers"] = [1, 100, 100, 100, 100, 0]
# Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
hp["tf_epochs"] = 0000
hp["tf_lr"] = 0.003
hp["tf_b1"] = 0.9
hp["tf_b2"] = 0.999
hp["tf_eps"] = 1e-9 


# Setting up the quasi-newton LBGFS optimizer (set nt_epochs=0 to cancel it)
hp["nt_epochs"] = 20000
hp["nt_lr"] = 0.5
hp["nt_ncorr"] = 50

# logger
hp["log_frequency"] = 10
hp['log_checkpoints'] = True
hp["log_checkpoint_freq"] = 10
hp['NN_name'] = 'SNRINF_pi2_Discovery_20000'
hp['RKsteps'] = 1
hp['q'] =400
hp['SNR'] = np.inf
hp['datafname'] = 'GNLSE_Disc_pi2.mat'

hp_file = hp['NN_name']+'_params.json'
with open(hp_file,'w') as fp:
    json.dump(hp, fp)