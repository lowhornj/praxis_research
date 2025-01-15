import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary
import numpy as np
from auto_encoder.network_matrix import *
torch.set_num_threads(10)  # Limit to 4 threads
torch.set_num_interop_threads(5)
from utils.utils import set_generator_seed, set_seed,masked_loss
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
from config import RCA_FRAMEWORK_PATH, remote_server_uri
import math
import os

class lstm_autoencoder():
    def __init__(self,
                 cell_mat,
                 n_epochs = 5,
                 step_window=30,
                 size_window=50,
                 batch_size=100,
                 n_layers = 5,
                 hidden_size = 512,
                 lr=.01,
                 name_model='lstm_model',
                 path_model = '../models/',
                 device = torch.device('cpu')):
        self.n_epochs = n_epochs
        self.n_cat = cell_mat.shape[1]
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.x_train = torch.tensor(cell_mat.astype(np.float64)).to(device)
        self.y_train = torch.tensor(cell_mat.astype(np.float64)).to(device)
        self.predictions = None
        self.actual = self.x_train.reshape(self.x_train.shape[0],self.x_train.shape[2],self.x_train.shape[1])
        self.step_window = 30
        self.size_window = 50
        self.batch_size = 100
        self.lr = lr
        self.criterion = masked_loss
        self.name_model = name_model
        self.path_model = path_model
        self.device = torch.device('cpu')
        set_seed()
        self.train_loader = DataLoader(self.x_train, batch_size=self.batch_size, shuffle=False)
        self.seq_length, self.size_data, self.nb_feature = self.x_train.data.shape
        self.test_loader = DataLoader(self.y_train, batch_size=self.batch_size, shuffle=False)
        self.model = LSTMAutoEncoder(self.n_layers, hidden_size=self.hidden_size, nb_feature=self.nb_feature, device=self.device)
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.loss_checkpoint_train = LossCheckpoint()
        self.loss_checkpoint_valid = LossCheckpoint()
        self.model_management = ModelManagement(self.path_model, self.name_model)
        self.earlyStopping = EarlyStopping(patience=5)
        self.run_id = None
        

    def load_checkpoint(self,model, optimizer, filename):
        # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
        start_epoch = 0
        #checkpoint = torch.load(filename)
        start_epoch = filename['epoch']
        model.load_state_dict(filename['state_dict'])
        optimizer.load_state_dict(filename['optimizer'])
        return model, optimizer, start_epoch
    
    def compare_mod_signature(self):
        """
        Performs feature comparison on the input data and previous model state. 
        """

        return (self.champion_signature[0]['tensor-spec']['shape'][2] == self.x_train.shape[2]) & (self.champion_signature[0]['tensor-spec']['shape'][1] == self.x_train.shape[1])
    
    def get_champion(self,client):
        """
        Fetches the Chamion model for a given model group, loads the weights of the model, and peforms signature comparison to ensure that schemas match. 
        """
        run = client.get_model_version_by_alias(self.name_model, "champion")
        mod_run = run.source
        model_metadata = run.source + '/MLmodel'
        artifacts_url = "/".join(mod_run.split("/")[:-2])
        statefile = artifacts_url + '/statefile'
        loaded_model = mlflow.pytorch.load_model(mod_run)
        logged_mod =mlflow.artifacts.download_artifacts(statefile)
        model_info = mlflow.models.get_model_info(mod_run)
        self.champion_signature =  model_info._signature_dict['inputs']
        self.champion_signature = eval(self.champion_signature.replace("true", "True"))
        if self.compare_mod_signature():
            mod = torch.load(logged_mod)
            self.model, self.optimizer, self.start_epoch = self.load_checkpoint(self.model, self.optimizer, mod)
        
    def evaluate(self,loader, validation=False, epoch=0):
        set_seed()
        self.model.eval()
        eval_loss = 0
        with torch.no_grad():
            for id_batch, data in enumerate(loader):
                data = data.to(self.device)
                output = self.model.forward(data)
                loss = self.criterion(data, output.to(self.device))
                eval_loss += loss.item()
            print('\r', 'Eval [{}/{} ({:.0f}%)] \tLoss: {:.6f})]'.format(
                id_batch + 1, len(loader),
                (id_batch + 1) * 100 / len(loader),
                loss.item()), sep='', end='', flush=True)
        avg_loss = eval_loss / len(loader)
        print('====> Validation Average loss: {:.6f}'.format(avg_loss))
        # Checkpoint
        if validation:
            self.loss_checkpoint_valid.losses.append(avg_loss)
            self.model_management.checkpoint(epoch, self.model, self.optimizer, avg_loss)
            return self.earlyStopping.check_training(avg_loss)
        
    def train(self,epoch):
        set_seed()
        self.model.train()
        train_loss = 0
        for id_batch, data in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            # forward
            data = data.to(self.device)
            output = self.model.forward(data)
            loss = self.criterion(data, output.to(self.device))
            # backward
            loss.backward()
            train_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.3)
            self.optimizer.step()

            print('\r', 'Training [{}/{} ({:.0f}%)] \tLoss: {:.6f})]'.format(
                id_batch + 1, len(self.train_loader),
                (id_batch + 1) * 100 / len(self.train_loader),
                loss.item()), sep='', end='', flush=True)
        try:
            avg_loss = train_loss / len(self.train_loader)
        except: 
            avg_loss = 0
        print('====> Epoch: {} Average loss: {:.6f}'.format(epoch, avg_loss))
        self.loss_checkpoint_train.losses.append(avg_loss)


    def promote_champion(self,client,versions,test_mape,mape):
        if versions == 0:
            client.set_registered_model_alias(name=self.name_model
                                              , alias="champion"
                                              , version='1')
        elif test_mape > float(mape):
            current_model_metadata = client.get_latest_versions(self.name_model
                                                                , stages=["None"])
            latest_model_version = current_model_metadata[0].version
            client.delete_registered_model_alias(self.name_model, "champion")
            client.set_registered_model_alias(name=self.name_model
                                              , alias="champion"
                                              , version=latest_model_version)
            
            
    def evaluate_champion(self):
        '''
        Evaluates the input data against the Champion to determine the MAPE. 
        '''
        try:
            test_mape = []
            eval_loss=0
            self.model.eval()
            predict = torch.zeros(size=self.y_train.shape, dtype=torch.float64)
            with torch.no_grad():
                for id_batch, data in enumerate(self.test_loader):
                    data = data.to(self.device)
                    output = self.model.forward(data)
                    predict[id_batch*data.shape[0]:(id_batch+1)*data.shape[0], :, :] = output.reshape(data.shape[0],self.size_data, -1)
                    loss = self.criterion(data, output.to(self.device))
                    eval_loss += loss.item()
        except Exception as e: 
            print('Error in forward pass: ' + str(e))

        try:
            mus = torch.abs(((data - output)/data))
            mus[mus == np.inf] = 0
            mus[mus >1 ] = 1
            test_mape.append(float(torch.nanmean(mus)))
            test_mape = np.mean(test_mape)  
        except Exception as e:
            print("Error in mape calculation: " + str(e))
        
        return test_mape
        
    def fit(self):
        set_seed()
        self.model = self.model.to(self.device) 
        mlflow.set_tracking_uri(remote_server_uri)
        mlflow.set_experiment("/"+self.name_model)
        path_exists = os.path.exists(self.path_model)
        if not path_exists:
            os.makedirs(self.path_model)
            
        client = MlflowClient()
        versions = []
        try:
            for mv in client.search_model_versions(f"name='{self.name_model}'"):
                versions.append(mv)
            versions = len(versions)
        except:
            print('No versions exist')
            versions = 0 
        
        if versions > 0:
            try:
                self.get_champion(client)
                print('Successfully retrieved Champion')
            except Exception as e: 
                print('Issue retrieving champion: ' + str(e))

            try:
                test_mape = self.evaluate_champion()
                print('Successfully evaluated Champion')
            except Exception as e: 
                test_mape = 1
                print('Issue evaluating champion: ' + str(e))

        else:
            test_mape = 1

        with mlflow.start_run() as run:
            self.run_id = run.info.run_id
            params = {
                    "epochs": self.n_epochs,
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                    "loss_function": self.criterion.__class__.__name__,
                    "optimizer": self.optimizer.__class__.__name__,
                    'step_window':self.step_window,
                    'size_window':self.size_window,
                    'batch_size':self.batch_size
                }
            # Log training parameters.
            mlflow.log_params(params)
        
            with open(self.path_model+"model_summary.txt", "w", encoding="utf-8") as f:
                f.write(str(summary(self.model)))
            mlflow.log_artifact(self.path_model+"model_summary.txt")

            for epoch in range(1, self.n_epochs):
                self.train(epoch)
                if self.evaluate(self.test_loader, validation=True, epoch=epoch):
                    break
                # Lr on plateau
                if self.earlyStopping.patience_count == 5:
                    print('lr on plateau ', self.optimizer.param_groups[0]['lr'], ' -> ', self.optimizer.param_groups[0]['lr'] / 10)
                    self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr'] / 10
            
        
            eval_loss=0
            self.model.eval()
            predict = torch.zeros(size=self.y_train.shape, dtype=torch.float64)
            with torch.no_grad():
                for id_batch, data in enumerate(self.test_loader):
                    data = data.to(self.device)
                    output = self.model.forward(data)
                    predict[id_batch*data.shape[0]:(id_batch+1)*data.shape[0], :, :] = output.reshape(data.shape[0],self.size_data, -1)
                    loss = self.criterion(data, output.to(self.device))
                    eval_loss += loss.item()

            avg_loss = eval_loss / len(self.test_loader)
            print('====> Prediction Average loss: {:.6f}'.format(avg_loss))
            self.predictions  = predict.reshape(predict.shape[0],predict.shape[2],predict.shape[1])

            signature = infer_signature(self.x_train.numpy(), self.x_train.numpy())
        
            torch.save(self.model, self.path_model+"modelfile")
            state = {'epoch': epoch + 1, 'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()}
            torch.save(state, self.path_model+"statefile")
            
            mus = torch.abs(((data - output)/data))
            mus[mus == np.inf] = 0
            mus[mus >1 ] = 1
            mape = torch.nanmean(mus)
            mlflow.log_metric("MAPE", f"{mape:3f}", step=epoch)
            
            mlflow.pytorch.log_model(
                self.model,
                artifact_path='model/metadata',
                signature=signature,
                registered_model_name=self.name_model
                )

            mlflow.log_artifact(self.path_model+"modelfile")
            mlflow.log_artifact(self.path_model+"statefile")
            mlflow.end_run()
            
        self.promote_champion(client,versions,test_mape,mape)

