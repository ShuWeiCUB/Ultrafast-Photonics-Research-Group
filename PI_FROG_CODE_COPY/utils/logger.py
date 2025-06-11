import json
import os
import sys
import tensorflow as tf
import time
from datetime import datetime, date
import numpy as np
import pickle
import matplotlib.pyplot as plt

class Logger(object):
    def __init__(self, hp,Directory = os.getcwd()):
        print("Hyperparameters:")

        # print(json.dumps(hp[k] for k in hp.keys(), indent=2))
        print()

        print("TensorFlow version: {}".format(tf.__version__))
        print("Eager execution: {}".format(tf.executing_eagerly()))
        print("GPU-accerelated: {}".format(tf.test.is_gpu_available()))

        self.start_time = time.time()
        self.prev_time = self.start_time
        self.frequency = hp["log_frequency"]
        self.loss_log  = np.array([])
        self.epoch_ax = np.array([])
        self.loss_log_fig, self.loss_log_ax = plt.subplots(1,1)
        self.plot_loss_log = True
        self.total_epochs = 0
        self.SaveVars = False
        
        if "NN_name" not in hp:
            today = date.today()
            d4 = today.strftime("%b-%d-%Y")
            hp['NN_name'] = 'NN_created_' + d4
        self.Name = hp['NN_name']
        
        if not os.path.isdir(os.path.join(Directory,self.Name)):
            os.mkdir(os.path.join(Directory,self.Name))
        self.SaveDir = os.path.join(Directory,self.Name)
        
        if "log_checkpoints" not in hp:
            hp['log_checkpoints'] = False
        self.log_checkpoints= hp['log_checkpoints']
        
        if hp['log_checkpoints']:
            self.log_checkpoint_freq = hp["log_checkpoint_freq"]
        
            if "log_checkpoint_dir" not in hp:
                hp['log_checkpoint_dir'] = self.SaveDir + '/Checkpoints'
            else:
                hp['log_checkpoint_dir'] = os.path.join(self.SaveDir,'Checkpoints')
            self.Checkpoint_dir = hp['log_checkpoint_dir']
            self.Checkpoint_name = self.Name + '_chk_epoch0'
            if not os.path.isdir(self.Checkpoint_dir):
                os.mkdir(self.Checkpoint_dir)
            
                    
        
    def set_total_epoch(self,total_epochs):
        self.total_epochs = total_epochs
        
    def get_total_epoch(self):
        return self.total_epochs
        

    def get_epoch_duration(self):
        now = time.time()
        edur = datetime.fromtimestamp(now - self.prev_time) \
            .strftime("%S.%f")[:-5]
        self.prev_time = now
        return edur

    def get_elapsed(self):
        return datetime.fromtimestamp(time.time() - self.start_time) \
                .strftime("%M:%S")

    def get_error_u(self):
        return self.error_fn()

    def set_error_fn(self, error_fn):
        self.error_fn = error_fn

    def log_train_start(self, model, model_description=False):
        print("\nTraining started")
        print("================")
        if model_description:
            print(model.summary())
        if self.log_checkpoints and len(self.epoch_ax) < 1:
            self.log_save_checkpoints(0,model,save = True)
                
    def log_train_epoch(self, epoch, loss, NeurNet, custom="", is_iter=False, saveVars = False, Vars2save = None):
        if epoch % self.frequency == 0:
            name = 'nt_epoch' if is_iter else 'tf_epoch'
            print(f"{name} = {epoch:6d}  " +
                  f"elapsed = {self.get_elapsed()} " +
                  f"(+{self.get_epoch_duration()})  " +
                  f"loss = {loss:.4e}  " + custom)
            self.loss_log = np.append(self.loss_log,loss)
            self.epoch_ax = np.append(self.epoch_ax,epoch)
            if saveVars and epoch>self.frequency:
                self.log_custom_variables(Vars2save = Vars2save)
            if self.log_checkpoints:
                self.log_save_checkpoints(epoch,NeurNet,save = True)
        if epoch == NeurNet.hp['nt_epochs']:
            self.log_save_checkpoints(epoch,NeurNet,save = True)
            
        
    def log_save_checkpoints(self, epoch, NN_model,save = False):
        if epoch % self.log_checkpoint_freq == 0 or save:
            name ='Save Checkpoint at Epoch '
            print(f"{name} = {epoch:6d}")
            NN_model.model.save_weights(self.Checkpoint_dir+'/'+str(epoch))
            self.log_save_hp(NN_model.hp)
            self.log_save_logged_data()
            

    def log_train_opt(self, name):
        print(f"-- Starting {name} optimization --")

    def log_train_end(self, epoch, NeurNet, custom=""):
        print("==================")
        print(f"Training finished (epoch {epoch}): " +
              f"duration = {self.get_elapsed()}  " )
              #+
              #f"error = {self.get_error_u():.4e}  " + custom)
        if self.plot_loss_log:
            self.loss_log_ax.cla()
            self.loss_log_ax.plot(self.epoch_ax,self.loss_log)
            plt.xlabel('Epoch')
            plt.ylabel('loss')
            plt.title('Training Record')
        if self.log_checkpoints:
            self.log_save_checkpoints(epoch,NeurNet,save = True)
        self.log_save_hp(NeurNet.hp)
        self.log_save_logged_data()
    
    def log_save_hp(self, hp, filename = str('')):
        if not bool(filename):
            filename = self.Name
        hp_file = self.SaveDir + '/' + filename + '_hp.json'
        if bool(hp):
            with open(hp_file,'w') as fp:
                json.dump(hp, fp)
            print('HP saved to file: ' + hp_file)
        else:
            print('HP file empty non saved')
            
            
    def log_save_logged_data(self, filename = str(''),saveVars = False):
        if not bool(filename):
            filename = self.Name
            data_file = self.SaveDir + '/' + filename + '_logger_data.pkl'
        else:
            data_file = filename
        # write logger data to dict file for ease of saving
        logger_data = {}
        logger_data['epoch_ax'] = self.epoch_ax
        logger_data['loss_log'] = self.loss_log
        if self.SaveVars:
            for i, j in zip(self.Names_of_vars,self.Vars_log):
                    logger_data[i] = self.Vars_log[j]
        with open(data_file,'wb') as fp:
            pickle.dump(logger_data, fp)
            fp.close()
        print('Data saved to file: ' + data_file )
     
    def log_get_logged_data(self, filename = str('')):
        if not bool(filename):
            filename = self.Name
            data_file = self.SaveDir + '/' + filename + '_logger_data.pkl'
        else:
            data_file = filename
     
    def log_load_logged_data(self, filename = str('')):
        if not bool(filename):
            filename = self.Name
            data_file = self.SaveDir + '/' + filename + '_logger_data.pkl'
        else:
            data_file = filename
        # write logger data to dict file for ease of saving
        if os.path.isfile(data_file):
            with open(data_file,'rb') as fp:
                logger_data = pickle.load(fp)
                fp.close()
                print('Data loaded from file: ' + data_file )
                self.logger_data = {}
                for i in logger_data.keys():
                    self.logger_data[i] = logger_data[i]
                    if i != 'epoch_ax' and i != 'loss_log':
                        self.Vars_log[i] = logger_data[i]
                
                # for i in logger_data.keys() not == 'epoch_ax' or 'loss_log':
                #     print(logger_data.keys()[i])
                plt.rcParams.update({
                'font.family': 'Arial',  # Font family
                'font.size': 11,          # Font size
                'axes.labelweight': 'normal',  # Label weight
                'axes.labelcolor': 'black',   # Label color
                'axes.labelsize': 11,         # Label size
                'axes.titlesize': 11,         # Title size
                'axes.titleweight': 'bold',   # Title weight
                })
                fig, ax = plt.subplots(1,1, figsize = (4,2))
                ax.plot(self.logger_data['epoch_ax'],self.logger_data['loss_log'],'sg',linewidth = 3)
                ax.set_title('Training Loss Log' + self.Name)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
#                ax.set_yscale('log')
 #               ax.set_xlim((0,self.logger_data['epoch_ax'].max()))
                
                plt.tight_layout()
                plt.savefig('Loss_fx',dpi = 600)
                self.epoch_ax = self.logger_data['epoch_ax']
                self.loss_log = self.logger_data['loss_log']
                plt.savefig('loss_log.png',dpi = 500)
        else:
            
            print('No logger data file found continuing without')
            
        
            
        
            
    def log_custom_variables(self, Names_of_vars = None, Vars2save = None, Initialize = False):
            if Initialize:
                self.Vars_log ={}
                self.Names_of_vars = Names_of_vars
                self.SaveVars = True
                for i, j in zip(Names_of_vars,Vars2save):
                        self.Vars_log[i] = j
            else:
                for i, j in zip(self.Names_of_vars,Vars2save):
                    if len(self.Vars_log[i].shape)>1:
                        self.Vars_log[i] = np.append(self.Vars_log[i],j,axis = 1)
                    else:
                        self.Vars_log[i] = np.append(self.Vars_log[i],j)
            

            
            
            
            
        