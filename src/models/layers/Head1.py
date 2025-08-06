
import torch
import torch.nn as nn

from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np


#class PredictionHead(nn.Module):
#    def __init__(self, classes, dim,forecast_len,n_vars,batch_size2, individual, head_dropout=0.4, flatten=False): #
#        super().__init__()
#        self.individual = individual
#        self.n_vars = n_vars
#        self.classes = classes
#        self.flatten = flatten
#        self.batch_size2=batch_size2
#        self.forecast_len=forecast_len

#        #head_dim = dim*num_patch
        
#        if self.individual:
#            self.linears = nn.ModuleList()
#            self.dropouts = nn.ModuleList()
#            self.flattens = nn.ModuleList()
#            for i in range(self.n_vars):
#                #self.flattens.append(nn.Flatten(start_dim=-2))
#                self.linears.append(nn.Linear(384, forecast_len))
#                self.dropouts.append(nn.Dropout(head_dropout))
#        else:
#            self.dropout = nn.Dropout(head_dropout)
#            self.linear = nn.Linear(dim, classes) # you should adjust the input features to match the input matrix
#            self.layerNorm = nn.LayerNorm(dim)
#            self.gelu = nn.GELU()
#            self.relu = nn.ReLU()
#            self.leaky_relu = nn.LeakyReLU()

#            self.drop = nn.Dropout(0.4) # was 0.3
#            #self.linear2a = nn.Linear(192, 30)
#            self.linear2a = nn.Linear(384, 30)
#            self.linear2b = nn.Linear(30, classes)
#            self.linear2 = nn.Linear(dim*64, classes) # you should adjust the input features to match the input matrix
#            #self.linear2 = nn.Linear(dim*256, classes)
#            self.linear3 = nn.Linear(384, classes) # you should adjust the input features to match the input matrix
        
#    def forward(self, x):
#       #print(x.shape)#torch.Size([128, 192])

#       if self.individual:
#            x_out = []
#            for i in range(self.n_vars):
#                #print('first',x.shape)
#                #z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * num_patch]
#                z = self.linears[i](x)                    # z: [bs x forecast_len]
#                #print('linears',z.shape)
#                z = self.dropouts[i](z)
#                #print('dropouts',z.shape)
#                x_out.append(z)
#                #print(x_out.shape)
#            x = torch.stack(x_out, dim=1)         # x: [bs x nvars x forecast_len]
#            #print('stack',x.shape)
#            x =x.permute(0,2,1)
#            #print('final',x.shape)
#       else:
#                # Reshape x to match the expected shape
#                #x = x.view(32, 2, 96)
#                #x= self.leaky_relu(x)
#                self.batch_size2, _ = x.shape  

    
#                #x = self.layerNorm(x)
#                #x = self.gelu(x)
#                x = self.drop(x)
#                # x = x.view(32, 96 * 8)  # Flatten the tensor to shape [32, 768]
#                x = self.linear2a(x)
#                x = self.linear2b(x)
#                #x = self.linear3(x)
#                #print(x.shape)
#                #x = x.view(32, 5040,2)
#                x = x.view(self.batch_size2, self.forecast_len, self.n_vars)
#                ##x = x.unsqueeze(3)
#                #print(x.shape)
#       return x

   
class PredictionHead(nn.Module):
    def __init__(self, classes, dim,target_points,n_vars,batch_size2,individual,head_dropout=0.4):
        super().__init__()
        #self.custom_activation = CustomActivation(floor_value=-10.0)
        self.n_vars=n_vars
        self.target_points=target_points
        self.batch_size2=batch_size2
        self.classes = classes
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(dim, classes) # you should adjust the input features to match the input matrix
        self.layerNorm = nn.LayerNorm(dim*8)
        #self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.prelu = nn.PReLU()
        self.drop = nn.Dropout(0.4) # was 0.3
        self.drop2=nn.Dropout(0.3)
        self.linear2a = nn.Linear(dim*8, 60)
        self.tanh=nn.Tanh()
        self.linear2b = nn.Linear(60, classes)
        #self.linear2a = nn.Linear(dim*8,classes )
        #self.linear2b = nn.Linear(60, target_points)

        #self.linear2 = nn.Linear(dim*64, classes) # you should adjust the input features to match the input matrix




    def forward(self, x):
        #print (x.shape)
        
        #x = self.layerNorm(x)
        #x = self.custom_activation(x)
        #x = self.prelu(x)
        #x = self.leaky_relu(x)
        #x=self.prelu (x)
        #self.relu = nn.ReLU()
        #x = x.mean(dim=-2)
        #x = self.drop(x)
        #x=self.drop2(x)
       # print (x.shape)
        self.batch_size2, _ = x.shape       
        #x = x.view(self.batch_size2 , 4,96 ) #
        #x= self.layerNorm(x)
        #x = x.view(self.batch_size2 , 4*96 )
        #x = self.linear2(x)
        #x= self.tanh(x)
        x = self.linear2a(x)
        #x = self.leaky_relu(x)
        #x= self.tanh(x)
        #self.gelu = nn.GELU()
        x = self.drop(x)
        x = self.linear2b(x)
        #print (x.shape)
        x = x.view(self.batch_size2,self.target_points,self.n_vars)
        #print (x.shape)
        #if x.shape ==[64, 672]:
        #    x = x.view(self.batch_size2,96,7)
        #else :
        #    x = x.view(self.batch_size2,96,7)
        return x


