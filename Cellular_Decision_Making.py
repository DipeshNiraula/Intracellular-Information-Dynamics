import Intracellular_Information_Dynamics as iid
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cpu")

class RC(iid.CellReservoir):
    def __init__(self, memory_retention_rate=0.9, *args, **kw):
        super().__init__(*args, **kw)
        self.memory_retention_rate = memory_retention_rate   
        # build cell organelle structures 
        self.cm_surface, self.pc_surface, self.co_surface, self.cs,\
            self.cm_idx, self.pc_idx, self.co_idx, self.cs_idx = self.cell_organelles() 
        
        #Collect organelle index    
        self.organelle_idx = np.vstack((self.cm_idx, self.pc_idx, self.co_idx, self.cs_idx))        
        self.vertex_idx = self.vertex_index()
        #initialize conductance map
        self.conductance_map = self.initiate_conductance()
    
    def initiate_statemap(self):
        '''
        Initialize memory state map 

        '''
        self.state_map = self.empty_Cell()
    
    def initiate_signalmap(self):
        '''
        Initialize phyical signal map        


        '''
        self.signal_map = self.empty_Cell()

    def get_co_state(self):
        '''
        Get central organelle surface vertex memory state

        Returns
        -------
        1D numpy array of size m
            central organelle surface vertex memory state

        '''
        return self.state_map[self.co_idx[:,0], self.co_idx[:,1], self.co_idx[:,2]]
    
    def get_co_signal(self):
        '''
        Get central organelle surface vertex physical signal

        Returns
        -------
        1D numpy array of size m
            central organelle surface vertex physical signal

        '''
        return self.signal_map[self.co_idx[:,0], self.co_idx[:,1], self.co_idx[:,2]]
    
    def forward_one_step(self, cellreservoir, input_signal_idx):
        '''
        Carries out one forward step of information flow in a cell 
        from source until saturation

        Parameters
        ----------
        cellreservior : 3D numpy array of size 2n-1 by 2n- by 2n-1 
            cellreservior vertices contain the potential map and edges contain the edge-current at time t
        input_signal_idx : 2D array of size rows by 3
            index of vertices that has received signal at time t 

        Returns
        -------
        None.

        '''
        #initiate signal tracker
        # vertex indices in cell reservoir is 2*voxel index of cell  
        signal_time_tracker = [2*input_signal_idx]
        #initiate visted set
        visited = set()
               
        signal_time = 0
        while True:
            cache = np.copy(self.signal_map)
            # one forward step of information dynamics
            signaled_idx, cellreservoir, self.signal_map, self.state_map  =\
                self.forward(cellreservoir, 
                             self.conductance_map, 
                             self.signal_map, 
                             self.state_map, 
                             signal_time_tracker[signal_time], 
                             visited,
                             memory_retention_rate=self.memory_retention_rate)
                
            visited = visited.union(set(tuple(i) for i in signal_time_tracker[signal_time]))
            
            signal_time += 1
            if np.array_equal(cache, self.signal_map):
                print("saturated")
                break

            if signaled_idx.shape[0] > 0:
                #store signaled vertex
                signal_time_tracker.append(signaled_idx)
            else:
                print('Signal cant flow anymore')
                break    
    
    def forward_complete(self, X, source='point'):
        '''
        Carries out complete forward step of information flow in a cell 
        from source until saturation for time-series perturbation

        Parameters
        ----------
        X : numpy array of size t by 1
            Strength of the environmental perturbation as a function of time.
        source: str, categorical, optional
            extracellular ion distribution type. The default is "point".

        Returns
        -------
        S : numpy array of size t by m
            Array of central organelle memory state corresponding to t environmental perturbation

        '''
        S = []
        #initate memory state map
        self.initiate_statemap()
        if source == "point":            
            input_signal_cord = np.array([[0, 0, self.r_pc]])        
            input_signal_idx = self.coordinate_to_index(input_signal_cord)  
        
        elif source == "spherical":
            input_signal_idx = self.pc_idx
            input_signal_cord = self.index_to_coordinate(input_signal_idx) 

            
        for t, x_t in enumerate(X):
            print(f'timestep: {t} X: {x_t}')
            #using random source at each time point

            #initiate a cell reservior
            cr = self.empty_CellReservoir_grid()  
            #initate signal map
            self.initiate_signalmap() 
            # calculate potential map for given input at time t
            potential_map = self.potential(self.organelle_idx, 
                                           input_signal_cord, 
                                           charge = [x_t])
            # set the vertex with potential_map
            cr[self.vertex_idx] = potential_map     
            #bias_correction = 1 - beta**(t+1)
            self.forward_one_step(cr, input_signal_idx)             
            s_t = self.get_co_state()      
            print(s_t.shape)
            S.append(s_t)
        S = np.array(S)
        return S


class Readout(nn.Module):
    def __init__(self,
                 input_dim,
                 out_dim,
                 node1 = 128,
                 node2 = 128,
                 node3 = 64,
                 node4 = 32,
                 dropout=0.0):
                    
        super(Readout, self).__init__()
        
        self.actFnIn = nn.ELU()
        self.actFn1 = nn.ELU()
        self.actFn2 = nn.ELU()
        self.actFn3 = nn.ELU()
        
        self.linearIn = nn.Linear(input_dim, node1)
        self.linearMid1 = nn.Linear(node1, node2)
        self.linearMid2 = nn.Linear(node2, node3)
        self.linearMid3 = nn.Linear(node3, node4)
        self.linearOut = nn.Linear(node4, out_dim)
        
        self.p = dropout

    def forward(self,x):
        x = self.linearIn(x)
        x = self.actFnIn(x)
        x = F.dropout(x, p=self.p)
        
        x = self.linearMid1(x)
        x = self.actFn1(x)
        x = F.dropout(x, p=self.p)
        
        x = self.linearMid2(x)
        x = self.actFn2(x)
        x = F.dropout(x, p=self.p)
        
        x = self.linearMid3(x)
        x = self.actFn3(x)
        x = F.dropout(x, p=self.p)
        
        return self.linearOut(x)    

#######___________helper functions___________#######
def to_torch(x):
    return torch.from_numpy(x).to(device).float()

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())    

def min_max(arr):
    return np.divide(arr-np.min(arr), np.max(arr)-np.min(arr))   

def min_max2(arr, min_, max_):
    return np.divide(arr-min_, max_-min_)

def subplot(ax, title, X, y, y_pred, legend=False):
    rmse_ = np.round(rmse(y, y_pred), 3) 
    ax.set_title(title)
    ax.grid(False)
    if legend:
        ax.plot(X, 'g', label='Input')
        ax.plot(y, 'b', label='Truth')
        ax.plot(y_pred, 'r:', label=f'Prediction\n RMSE: {rmse_}')
    else:        
        ax.plot(X, 'g')#, label='Input')
        ax.plot(y, 'b')#, label='Truth')
        ax.plot(y_pred, 'r:', label=f'RMSE: {rmse_}')    
    ax.set_xlabel('Time')
    ax.legend()
    min_ = np.min(y)-0.1
    max_ = np.max(y)+0.1
    ax.set_ylim(min_, max_)    
    return rmse_
#######_____________________________________#######

def Decision_Maker(S_train, 
                   y_train, 
                   L1_alpha=1E-6, 
                   L2_alpha=1E-6, 
                   learning_rate=1E-3, 
                   epoch=1000):
    '''
    Train 4 types of intracellular decision-makers via regression:
        Linear, Lasso, Ridge, and Artificial Neural Network (ANN)
    
    Parameters
    ----------
    S_train : numpy array of size n x m
        n Flattened central organelle memory state of length m
    y_train : numpy array of size n x 1
        n cell reponse of length 1
    L1_alpha : TYPE, optional
        Coefficient for L1 regularization. The default is 1E-6.
    L2_alpha : TYPE, optional
        Coefficient for L2 regularization. The default is 1E-6.
    learning_rate : TYPE, optional
        Learning rate from backpropagation. The default is 1E-3.
    epoch : int, optional
        Training epochs for ANN. The default is 1000.

    Returns
    -------
    linear : sklearn linear regression model
        Trained Linear Model
    lasso : sklearn Lasso regression model
        Trained Lasso Model
    ridge : sklearn Ridge regression model
        Trained Ridge Model
    NN : pytorch ANN model
        Trained ANN model

    '''
    S_train = min_max(S_train)
    
    linear = LinearRegression()
    linear.fit(S_train, y_train)  
    
    lasso = Lasso(alpha=L1_alpha)
    lasso.fit(S_train, y_train)  
    
    ridge = Ridge(alpha=L2_alpha)
    ridge.fit(S_train, y_train)  
    
    NN = Readout(S_train.shape[1], y_train.shape[1])
    NN.train()
    S_train_t = torch.tensor(S_train).float()
    y_train_t = torch.tensor(y_train).float()
                    
    optimizer = torch.optim.Adam(NN.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    
    for _ in range(epoch):
        optimizer.zero_grad()     
        out = NN(S_train_t) 
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step() 
    
    return linear, lasso, ridge, NN

def cell_response(X, y_list, y_mode_list, source='point'):
    '''
    Training Cell Response to the CellReservoir via Intracellular Information Dynamics
    for a given environental perturbation. Saves plots.

    Parameters
    ----------
    X : numpy array of size t by 1
        Strength of the environmental perturbation as a function of time.
    y_list : list of numpy array of size t by 1
        list of cell responses.
    y_mode_list : list of str
        title for file name.
    source: str, categorical, optional
        extracellular ion distribution type. The default is "point".

    Returns
    -------
    rmse_dict : dictionary
        Dictionary of RMSE values for Linear, Lasso, Ridge, and ANN model calculated
        in between grouth truth and model prediction for training and testing set.

    '''
    X_train = X[:X.shape[0]//4, :]
    X_test = X[X.shape[0]//4:, :]
    
    rc = RC()
 
    S_train = rc.forward_complete(X_train, source=source)
    S_test = rc.forward_complete(X_test, source=source)
    
    #min-max normalize
    min_ = np.min(np.vstack((S_train, S_test))) 
    max_ = np.max(np.vstack((S_train, S_test))) 
    
    S_train = min_max2(S_train, min_, max_)
    S_test = min_max2(S_test, min_, max_)
    
    rmse_dict = {}
    
    for y, y_mode in zip(y_list, y_mode_list):
        y_train = y[:y.shape[0]//4, :]
        y_test = y[y.shape[0]//4:, :]
        
        linear, lasso, ridge, NN = Decision_Maker(S_train, y_train)

        fig = plt.figure(figsize=(7, 7), dpi=150)            
               
        ax1 = fig.add_subplot(4,2,1)
        rmse_linear_train = subplot(ax1, 
                                    'Linear Regression Train', 
                                    X_train, 
                                    y_train, 
                                    linear.predict(S_train))
        ax2 = fig.add_subplot(4,2,2)
        rmse_linear_test = subplot(ax2, 
                                   'Linear Regression Test', 
                                   X_test, 
                                   y_test, 
                                   linear.predict(S_test), 
                                   legend=True)
        
        ax3 = fig.add_subplot(4,2,3)
        rmse_lasso_train = subplot(ax3, 
                                   'Lasso Regression Train', 
                                   X_train, 
                                   y_train, 
                                   lasso.predict(S_train))
        ax4 = fig.add_subplot(4,2,4)
        rmse_lasso_test = subplot(ax4, 
                                  'Lasso Regression Test', 
                                  X_test, 
                                  y_test, 
                                  lasso.predict(S_test))
        
        ax5 = fig.add_subplot(4,2,5)
        rmse_ridge_train = subplot(ax5, 
                                   'Ridge Regression Train', 
                                   X_train, 
                                   y_train, 
                                   ridge.predict(S_train))
        ax6 = fig.add_subplot(4,2,6)
        rmse_ridge_test = subplot(ax6, 
                                  'Ridge Regression Test', 
                                  X_test, 
                                  y_test, 
                                  ridge.predict(S_test))
        
        NN.eval()
        ax7 = fig.add_subplot(4,2,7)   
        rmse_NN_train = subplot(ax7, 
                                'ANN Regression Train', 
                                X_train, 
                                y_train, 
                                NN(torch.tensor(S_train).float()).detach().numpy())
        ax8 = fig.add_subplot(4,2,8)
        rmse_NN_test = subplot(ax8, 
                               'ANN Regression Test', 
                               X_test, 
                               y_test, 
                               NN(torch.tensor(S_test).float()).detach().numpy()) 
        
        plt.tight_layout()
        plt.savefig(f'./plot/{y_mode}.svg', bbox_inches="tight")
        plt.savefig(f'./plot/{y_mode}.png', bbox_inches="tight")
        
        rmse_dict[y_mode] = [rmse_linear_train, 
                             rmse_linear_test, 
                             rmse_lasso_train, 
                             rmse_lasso_test, 
                             rmse_ridge_train, 
                             rmse_ridge_test, 
                             rmse_NN_train, 
                             rmse_NN_test]
        
    return rmse_dict

def noise_analysis(source='point'):
    '''
    Analyse CellReservoir robustness to noise in environental perturbation signal.
    Saves dataframe.

    Parameters
    ----------
    source: str, categorical, optional
        extracellular ion distribution type. The default is "point".

    Returns
    -------
    None.

    '''
    x = np.cos(np.arange(0, 60, 0.2))[:, np.newaxis]
    y = np.sin(np.arange(0, 60, 0.2))[:, np.newaxis]     

    y_minmax = min_max(y)
    y_step = np.array([1 if i > 0 else 0 for i in x])[:, np.newaxis] 
    
    rmse_dict = {}
    for noise_level in [0.05, 0.1, 0.25, 0.5, 1.0]:
        
        #Noise
        n = np.random.normal(scale=noise_level , size=x.size)[:, np.newaxis]
        x_noisy = x+n        
        x_noisy_minmax = min_max(x_noisy)
    
        rmse_dict.update(
            cell_response(x_noisy_minmax,
                          [y_minmax, y_step],
                          [f'{source}_source_noise_cos_to_sin_noise_level_{noise_level}', 
                           f'{source}_source_noisy_cos_to_step_noise_level_{noise_level}'],
                          source=source)
            )
    df = pd.DataFrame(rmse_dict)
    df.to_csv(f'./plot/Noise_Analysis_RMSE_{source}_source.csv')

def plot_noise_analysis(source = "point"):
    '''
    Function to plot results from Noise Analysis

    Parameters
    ----------
    source: str, categorical, optional
        extracellular ion distribution type. The default is "point".


    Returns
    -------
    None.

    '''
    df = pd.read_csv(f'./plot/Noise_Analysis_RMSE_{source}_source.csv', index_col=0)
    df['Model Type'] = ["Linear + Train", 
                        "Linear + Test", 
                        "Lasso + Train", 
                        "Lasso + Test",
                        "Ridge + Train",
                        "Ridge + Test",
                        "ANN + Train",
                        "ANN + Test"]

    for y_type, tag in zip(['sin', "step"], ["Sine", "Square"]):    
        dff = pd.DataFrame(columns = ['Model Type'])
        dff['Model Type'] = df['Model Type']
        for i in df.columns:
            if y_type in i:
                dff[i]=df[i]
    
        melted = pd.melt(dff, id_vars=['Model Type'])
        fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
        ax = sns.barplot(data = melted, 
                         x = "Model Type", 
                         y="value", 
                         hue="variable",
                         ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), 
                           rotation=45, 
                           horizontalalignment='right')
        ax.set_ylim([0,1.3])
        ax.set_ylabel("RMSE")
        legend_handles, _ = ax.get_legend_handles_labels()
        ax.legend(legend_handles, ['0.05', '0.1', '0.25', '0.5', '1.0'], 
                  bbox_to_anchor=(1,1), 
                  title='Noise Level')
        ax.set_title(f"{tag.title()} Response for {source.title()} Source Cosine Perturbation ")
        
        plt.savefig(f'./plot/Noise_Analysis_{source}_source_{tag}_response.svg', bbox_inches="tight")
        plt.savefig(f'./plot/Noise_Analysis_{source}_source_{tag}_response.png', bbox_inches="tight")
 
if __name__ == "__main__": 
    #Environmental Perturbation Signal and Cell Response Cycle
    x = np.cos(np.arange(0, 60, 0.2))[:, np.newaxis]
    y = np.sin(np.arange(0, 60, 0.2))[:, np.newaxis]     
    x_minmax = min_max(x)
    y_minmax = min_max(y)
    y_step = np.array([1 if i > 0 else 0 for i in x])[:, np.newaxis] 

    # ---------------Point Source
    # cos -> [sin, step]
    cell_response(x_minmax,
                  [y_minmax, y_step],
                  ['spherical_soruce_cos_to_sin', 'spherical_source_cos_step'],
                  source = "point")
    # #---------------spherical Source
    cell_response(x_minmax,
                  [y_minmax, y_step],
                  ['spherical_soruce_cos_to_sin', 'spherical_source_cos_step'],
                  source = "spherical")

    
    # #-----------different noise level
    noise_analysis(source='point')
    noise_analysis(source='spherical')
    
   
    plot_noise_analysis(source="point")
    plot_noise_analysis(source="spherical")