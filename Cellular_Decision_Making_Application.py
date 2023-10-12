import Cellular_Decision_Making as cdm
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D # For points and lines

from matplotlib.legend import Legend
from matplotlib import cm

def Decision_Maker_NN(S_train, 
                      y_train,
                      node1=128,
                      node2=128,
                      node3=64,
                      epoch=1000, 
                      dropout=0,
                      lambda1=0.005,
                      lambda2=0.001):
    '''
    Train an artificial neural network (ANN) decision-maker.
    Elastic Net approach (L1+L2 regularization) with dropout is used for 
        improving generalization. 

    Parameters
    ----------
    S_train : torch tensor of size n by m
        n Flattened central organelle memory state of length m
    y_train : torch tensor of size n by 1
        n cell reponse of length 1
    node1 : int, optional
        No of NN nodes for Layer 1. The default is 128.
    node2 : int, optional
        No of NN nodes for Layer 2. The default is 128.
    node3 : int, optional
        No of NN nodes for Layer 3. The default is 64.
    epoch : int, optional
        Epoch. The default is 1000.
    dropout : float, optional
        Probaility of droping a node in a layer. The default is 0.
    lambda1 : float, optional
        Coefficient for L1 regularization. The default is 0.005.
    lambda2 : float, optional
        Coefficient for L2 regularization. The default is 0.001.

    Returns
    -------
    NN : pytorch model
        Artificial Neural Network.

    '''
    NN = cdm.Readout(S_train.shape[1], 
                 y_train.shape[1],
                 node1 = node1,
                 node2 = node2,
                 node3 = node3,
                 dropout=dropout)
    NN.train()
    S_train_t = torch.tensor(S_train).float()
    y_train_t = torch.tensor(y_train).float()
                    
    optimizer = torch.optim.Adam(NN.parameters())
    criterion = torch.nn.MSELoss()
    
    for _ in range(epoch):
        optimizer.zero_grad()     
        out = NN(S_train_t)
        loss = criterion(out, y_train_t)
        #Elastic Net
        # Compute L1 and L2 loss component        
        l1= sum([p.abs().sum() for p in NN.parameters()])
        l2= sum([(p**2).sum() for p in NN.parameters()])

        # Add L1 and L2 loss components
        loss += lambda1*l1
        loss += lambda2*l2

        loss.backward()
        optimizer.step() 
    
    return NN


def main(data, activation_type="CD3", K_int_mM=150, source="spherical", ensemble_count=20):
    '''
    Learn CD8 behavior in the presence of extracellular K+. 
    CD8 activated by CD3 or CD3+CD28.
    
    Parameters
    ----------
    data : pandas DataFrame
        Dataframe with atleast CD3 or CD3_CD28, K_mM, and "IFNg+IL2+ Mean" columns
    activation_type : str, optional
        CD8 activator column name. The default is "CD3".
    K_int_mM: float, optional
        Intracellular K+ concentration in mM. The default is 150.0.
    source: str, categorical, optional
        extracellular K+ distribution type. The default is "spherical".
    ensemble_count: int, optional
        Number of ensemble. The default is 20.
    
    
    Returns
    -------
    None.
        Plots the results in two plots: training/validation and prediction.
        The results are from statistical ensembling with 20 trials.

    '''
 
    K_conc = data['K_mM'].to_numpy()
    K_len = K_conc.shape[0]
    activator_conc = data[activation_type].to_numpy().reshape(-1, 1)
    y = data["IFNg+IL2+ Mean"].to_numpy().reshape(-1, 1)
    
    #to avoid repeated calculation of S
    unique_K_conc = np.unique(data['K_mM'].to_numpy())
    unique_K_len = unique_K_conc.shape[0]
    
    # [K+]_int = 150mM
    In_K = np.array([K_int_mM-i for i in unique_K_conc]).reshape(-1,1)    
    
    
    df_long = pd.DataFrame(columns=[activation_type, 'K_mM', 'IFNg+IL2+ Mean', "NN", "Trial"])
    df_wide = pd.DataFrame()
    
    df_long_test = pd.DataFrame(columns=[activation_type, 'K_mM', "NN", "Trial"])
    
    test_activator_conc = np.array([0.05, 0.1, 0.2, 0.4, 0.5, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    test_K_conc = np.array([0, 5, 10, 15, 25, 35, 45, 50, 55, 60, 65])
    
    test_activator_len = test_activator_conc.shape[0]
    test_K_len = test_K_conc.shape[0]
    
    In_K_test = np.array([K_int_mM-i for i in test_K_conc]).reshape(-1,1)
    
    test_K_conc = np.hstack([test_K_conc for _ in range(test_activator_len)])
    test_activator_conc = np.repeat(test_activator_conc, test_K_len)
    
    #train
    for ensemble in range(ensemble_count):
        rc = cdm.RC() 
        
        S_K = rc.forward_complete(In_K, source=source)
        S_K_test = rc.forward_complete(In_K_test, source=source)
    
        #stacking S_K to obtain original data shape 
        S_K = np.vstack([S_K for i in range(int(K_len/unique_K_len))])
    
        In_activator = np.repeat(activator_conc, S_K.shape[1], axis=1)
        In = np.hstack((S_K, In_activator))
        
        S_K_test = np.vstack([S_K_test for _ in range(test_activator_len)])
        In_activator_test = np.repeat(test_activator_conc.reshape(-1,1), S_K_test.shape[1], axis=1)    
        In_test = np.hstack((S_K_test, In_activator_test))
        
        NN = Decision_Maker_NN(In, y)
            
        df_temp = pd.DataFrame(columns=[activation_type, 'K_mM', 'IFNg+IL2+ Mean', "NN", "Trial"])
        df_temp[activation_type] = data[activation_type]
        df_temp['K_mM'] = data['K_mM']
        df_temp['IFNg+IL2+ Mean'] = data['IFNg+IL2+ Mean']
        
        df_temp_test = pd.DataFrame(columns=[activation_type, 'K_mM', "NN", "Trial"])
        df_temp_test[activation_type] = test_activator_conc
        df_temp_test['K_mM'] = test_K_conc 
        
        NN.eval()
        df_temp['NN']  = NN(torch.tensor(In).float()).detach().numpy()
        df_temp["Trial"] = ensemble
        df_long = pd.concat([df_long, df_temp], axis= 0)
        
        df_wide[f"NN{ensemble}"] = NN(torch.tensor(In).float()).detach().numpy().reshape(-1,)
        
        df_temp_test['NN']  = NN(torch.tensor(In_test).float()).detach().numpy()
        df_long_test = pd.concat([df_long_test, df_temp_test], axis= 0)
        
    df_long.to_csv(f"./plot/{activation_type}_preprocess_training_NN.csv")
    df_long_test.to_csv(f"./plot/{activation_type}_preprocess_prediction_NN.csv")
    
    rmse = np.round(cdm.rmse(data['IFNg+IL2+ Mean'], df_wide.mean(axis=1)), 3)
    
    #plot 
    df_long.K_mM = pd.Categorical(df_long.K_mM)
    df_long_test.K_mM = pd.Categorical(df_long_test.K_mM)
    
    fig = plt.figure(figsize=(8,4), dpi=150)    
    ax = fig.add_subplot(1,2,1)    
    
    sns.lineplot(data = df_long,  
                 x = activation_type, 
                 y = "IFNg+IL2+ Mean", 
                 hue = "K_mM", 
                 style = "K_mM", 
                 errorbar = "sd", 
                 ax = ax, 
                 markers = True)
    
    ax.legend(title = "K_mM Data", bbox_to_anchor=(1.1, 1), borderaxespad=0.)
    ax.set_title(f"{activation_type} + K | IFNg+IL2+ Mean | NN Learning| RMSE: {rmse}")
    ax.set(xlabel=f'{activation_type} Concentration ', ylabel='Frequency of Parent CD8+ [%]')
    
    labels_ = ["0", "20", "30", "40"]
    markers = ['<', '>', '^', 'D']
    colors = cm.get_cmap('Dark2').colors
    sns.lineplot(data = df_long,  
                 x = activation_type, 
                 y = "NN", 
                 hue = "K_mM", 
                 style = "K_mM", 
                 errorbar = "sd", 
                 ax = ax,
                 markers = markers[:len(labels_)],
                 palette = colors[:len(labels_)],
                 legend = False)
    
    patches = []
    for lst in zip(labels_, colors[:len(labels_)], markers[:len(labels_)]):
        patches.append(Line2D([0], 
                              [0], 
                              linewidth = 0.1, 
                              linestyle = '', 
                              color = lst[1], 
                              markerfacecolor = lst[1], 
                              marker= lst[2], 
                              label = lst[0], 
                              alpha = 1.0))

    leg = Legend(ax, 
                 patches, 
                 labels = labels_, 
                 title = 'K_mM Model', 
                 bbox_to_anchor = (1.1, 0.5), 
                 borderaxespad = 0.)
    ax.add_artist(leg)

    ax = fig.add_subplot(1,2,2)
    sns.lineplot(data = df_long_test,  
                 x = activation_type, 
                 y = "NN", 
                 hue = "K_mM", 
                 style = "K_mM", 
                 errorbar = "sd", 
                 ax = ax, 
                 markers = True)
    
    ax.legend(title = "K_mM Model", bbox_to_anchor=(1.1, 1), borderaxespad=0.)
    ax.set_title("NN Prediction")
    ax.set(xlabel=f'{activation_type} Concentration ', ylabel='Frequency of Parent CD8+ [%]')
    plt.tight_layout()
    plt.savefig(f'./plot/{activation_type}_K.svg')
    plt.savefig(f'./plot/{activation_type}_K.png')

if __name__ == "__main__": 
    main(pd.read_csv("./Data/CD3_preprocess.csv"), activation_type="CD3")
    main(pd.read_csv("./Data/CD3_CD28_preprocess.csv"), activation_type="CD3_CD28")