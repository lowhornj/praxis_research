from mpge import Optim, MPGE
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import set_seed
from mpge.utils import Data_utility
import numpy as np
import polars as pl

class rca_arguments:
    def __init__(self
                ,input_dimensions
                ,batch_size = 32
                ,channel_size1 =5
                ,channel_size2 = 10
                ,hid1 = 512
                ,hid2 = 128
                ,hid3 = 1
                ,k_size = [5,3]
                ,window = 10
                ,L1Loss = True
                ,optim = 'adam'
                ,learning_rate = 0.001
                ,clip = 10
                ,epochs = 50
                ,lamda=0.15
                ,cuda=True):
            self.input_dimensions=input_dimensions
            self.n_e=input_dimensions
            self.batch_size=batch_size
            self.channel_size1=channel_size1
            self.channel_size2=channel_size2
            self.hid1=hid1
            self.hid2=hid2
            self.hid3=hid3
            self.k_size=k_size
            self.window=window
            self.L1Loss=L1Loss
            self.optim=optim
            self.learning_rate=learning_rate
            self.clip=clip
            self.epochs=epochs
            self.lamda=lamda
            self.cuda = cuda
            if self.cuda:
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        
class mpge_root_cause_diagnosis(rca_arguments):
    def __init__(self,input_df,cols_to_exclude,lamda=0.15):
        self.cols_to_exclude=cols_to_exclude
        self.data = Data_utility(0.95,0.05,True,1,10,0,input_df=input_df,cols_to_exclude=self.cols_to_exclude)
        self.X=self.data.train[0]
        self.Y=self.data.train[1]
        self.lamda=lamda
        self.args=rca_arguments(input_dimensions=self.data.dat.shape[1])
        self.model = eval('MPGE').Model(self.args)
        self.nParams = sum([p.nelement() for p in self.model.parameters()])
        if self.args.L1Loss:
            self.criterion = nn.L1Loss(reduction='sum')
        else:
            self.criterion = nn.MSELoss(reduction='sum')
        
        self.optim = Optim.Optim(
            self.model.parameters(),self.args.optim,self.args.learning_rate,self.args.clip
        )
        self.adjacency_matrix=None
        self.root_rank_score=None

    def train(self):
        self.model.train()
        total_loss=0
        n_samples=0
        for X, Y in self.data.get_batches(self.X,self.Y,self.args.batch_size,True):
            set_seed()
            if X.shape[0]!=self.args.batch_size:
                break
            X.to(self.args.device)
            Y.to(self.args.device)
            self.model.zero_grad()
            output=self.model(X)
            loss1=self.criterion(output.to(self.args.device),Y)
            loss_reg=self.lamda*torch.sum(torch.abs(self.model.A0))+(0.001)*torch.sum((self.model.A0)**2)
            loss=loss1+loss_reg
            loss.backward()
            grad_norm=self.optim.step()
            total_loss+=loss.data.item()
            n_samples+=(output.size(0)*self.data.m)

        #print('total_loss: '+str(total_loss))
        #print('n_samples: '+str(n_samples))
        return total_loss/n_samples

    def evaluate(self):
        self.model.eval()
        total_loss=0
        n_samples=0
        for X, Y in self.data.get_batches(self.X,self.Y,self.args.batch_size,True):
            set_seed()
            if X.shape[0]!=self.args.batch_size:
                break
            output=self.model(X)
            loss=self.criterion(output,Y)
            total_loss += loss.data.item()
            n_samples += (output.size[0]*self.data.m)

        return total_loss/n_samples

    def root_rank(self):
        """Extract the adjacency matrix from the network"""
        set_seed()
        a = torch.abs(self.model.A0).cpu().data.numpy()
        a_sum=np.sum(a,axis=1)
        for i in range(a.shape[0]):
            a[i,:]=a[i,:]/a_sum[i]
        delta=5

        """HAP mechanism, adjacency pruning with softmax"""
        b = a.copy()
        b=torch.from_numpy(b)
        b_s1=F.softmax(delta*b)
        b_s2=F.softmax(delta*b_s1)
        self.adjacency_matrix=torch.mm(torch.mm(b_s2,b_s1),b).data.numpy()

        """Eigenvalue decomposition of the adjacency matrix for root rank scores"""
        l, v = np.linalg.eig(self.adjacency_matrix.T)
        score = (v[:,0]/np.sum(v[:,0])).real
        score=np.around(score/np.sum(score),decimals=3)
        scores_df = pl.DataFrame({'Column':self.data.column_names,'Value':score})
        self.root_rank_score = scores_df.sort(["Value"], descending=True)

    def fit(self):

        for epoch in range(1,self.args.epochs+1):
            train_loss = self.train()
        self.root_rank()