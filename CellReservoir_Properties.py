import Intracellular_Information_Dynamics as iid
import Cellular_Decision_Making as cdm
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import pandas as pd


def equilibrium_potential(R = 8.314, #J/(K mol)
    T = 21,#c
    z = 1,#no of elementary charges
    F = 96485,#C/mol
    conc_ext = 150,#mM
    conc_int = 1) :#mM
    '''
    Nernst equation for equilibrium potential
    '''
    return ((R*(T+273.15)*np.log(conc_ext/conc_int))/(z*F))*1000#mV

def print_equilibrium_potential():
    '''
    Calculate equilibrium potential for Na, Cl, K, Ca, Mg, HCO3
    and save the results in dataframe

    '''
    df = pd.DataFrame(columns=['ions', 
                               'z', 
                               'int_conc_mM', 
                               'ext_conc_mM',
                               'eq_pot_21', 
                               'eq_pot_37'])
    df['ions'] = ['Na', 'Cl', 'K', 'Ca', 'Mg', 'HCO3']
    df['z'] = [1, -1, 1, 2, 2, -1]
    df['int_conc_mM'] = [13, 5, 150, 0.0001, 1, 8]
    df['ext_conc_mM'] = [142, 120, 4, 1, 0.5, 27]
    for temp in [21, 37]:
        for n, [in_, ex_, z] in enumerate(zip(df['int_conc_mM'].values, 
                                              df['ext_conc_mM'].values, 
                                              df['z'].values)):
            df.loc[n, f'eq_pot_{temp}'] = equilibrium_potential(T = temp, 
                                                                conc_ext = ex_, 
                                                                conc_int = in_, 
                                                                z = z)
    print(df)
    df.to_csv('./plot/equilibirum_potential.csv')


def visualize_3D_cell_with_umap(n_neighbors=8, min_dist=0.0, metric='manhattan'):
    
    '''
    
    Visualize 3D cell in 2D via UMAP projection
    
    Note:
    as of 9/6/2023 (umap-learn 0.5.3) umap.plot.connectivity() does not 
        take ax as argument and recreates a new plot
    
    I manually added ax=None as function argument in my local repo
    and modified 
    
    dpi = plt.rcParams["figure.dpi"]
    fig = plt.figure(figsize=(width / dpi, height / dpi))
    ax = fig.add_subplot(111)
    
    to
    
    if ax is None:
        dpi = plt.rcParams["figure.dpi"]
        fig = plt.figure(figsize=(width / dpi, height / dpi))
        ax = fig.add_subplot(111)
    '''
    import umap
    import umap.plot
    
    for cs_vol in [0.01, 0.05, 0.07, 0.08, 0.10, 0.15]:
            
        fig, axes = plt.subplots(3, 2, figsize=(8, 10), dpi=300, tight_layout=True)
        #fig.suptitle("UMAP Visualization and Connectivity")
        #axes.set_title("UMAP Visualization and Connectivity")
        row = 0
        cr = iid.CellReservoir(cs_frac = cs_vol)
        cm_surface, pc_surface, co_surface, cs, cm_idx, pc_idx, co_idx, cs_idx = cr.cell_organelles() 
        
        all_idx = np.vstack((cm_idx, pc_idx, cs_idx, co_idx))
        max_ = np.max(all_idx)
        min_ = np.min(all_idx)
        all_idx = (all_idx - min_)/(max_-min_)
        
        all_label = np.hstack((np.array(['1.CM' for _ in range(cm_idx.shape[0])]),
                               np.array(['2.PC' for _ in range(pc_idx.shape[0])]),
                               np.array(['3.CS' for _ in range(cs_idx.shape[0])]),
                               np.array(['4.CO' for _ in range(co_idx.shape[0])])))
                               
        mapper = umap.UMAP(n_neighbors=n_neighbors, 
                           min_dist=min_dist,
                           metric=metric).fit(all_idx)
        umap.plot.points(mapper, labels=all_label, theme='fire', ax=axes[row][0])
        umap.plot.connectivity(mapper, theme='fire', ax=axes[row][1])#, show_points=True)
        #umap.plot.diagnostic(mapper, diagnostic_type='pca')
        
        all_idx = np.vstack((pc_idx, cs_idx, co_idx))
        all_idx = (all_idx - min_)/(max_-min_)

        all_label = np.hstack((np.array(['2.PC' for _ in range(pc_idx.shape[0])]),
                               np.array(['3.CS' for _ in range(cs_idx.shape[0])]),
                               np.array(['4.CO' for _ in range(co_idx.shape[0])])))
        mapper = umap.UMAP(n_neighbors=n_neighbors, 
                           min_dist=min_dist,
                           metric=metric).fit(all_idx)
        umap.plot.points(mapper, labels=all_label, theme='fire', ax=axes[row+1][0])
        umap.plot.connectivity(mapper, theme='fire', ax=axes[row+1][1])

        
        all_idx = np.vstack((cs_idx, co_idx))
        all_idx = (all_idx - min_)/(max_-min_)

        all_label = np.hstack((np.array(['3.CS' for _ in range(cs_idx.shape[0])]),
                               np.array(['4.CO' for _ in range(co_idx.shape[0])])))
 
        mapper = umap.UMAP(n_neighbors=n_neighbors, 
                           min_dist=min_dist,
                           metric=metric).fit(all_idx)
        umap.plot.points(mapper, labels=all_label, theme='fire', ax=axes[row+2][0])
        umap.plot.connectivity(mapper, theme='fire', ax=axes[row+2][1])#, show_points=True)       
        
        fig.savefig(f'./plot/cell_visualization_umap_{round(cs_vol*100)}.svg', bbox_inches="tight")
        fig.savefig(f'./plot/cell_visualization_umap_{round(cs_vol*100)}.png', bbox_inches="tight")

def plot_signal_flow_vs_cs_vol(source='point', fig_size=(12, 9), dpi=150):
    
    '''
    Plot CellResrvoir Signal Map for Cytoskeleton Volume of 1%, 5%, 7%, 8%, 10%, 15%
    Parameters
    ----------
    source: str, categorical, optional
        extracellular K+ distribution type. The default is "spherical".
    '''
    
    # initialization
    sns.set_style("whitegrid", {'axes.grid' : False})

    fig, axes = plt.subplots(3, 4, figsize=fig_size, dpi=dpi)
    rows = 0
    COL = 0
    for cs_vol in [0.01, 0.05, 0.07, 0.08, 0.10, 0.15]:
        cr = cdm.RC(cs_frac = cs_vol)
        cellreservoir = cr.empty_CellReservoir_grid()        
            
        #plot cell structures    
        cellreservoir_plot = cr.empty_CellReservoir_grid()
        cellreservoir_plot[cr.vertex_idx] = 1*cr.cm_surface + \
            2*cr.pc_surface + 3*cr.cs+ 4*cr.co_surface 
        axes[rows][COL].imshow(cellreservoir_plot[cr.vertex_idx]\
                               [int(cellreservoir_plot.shape[0]/4),:,:], 
                               vmin=1E-5, 
                               vmax=4)
        #im.set_cmap('fire')
        axes[rows][COL].set_title(f'Geometry CS Vol {round(cs_vol*100)}%', 
                                  fontsize=15)   
        axes[rows][COL].tick_params(labelsize=30)
        axes[rows][COL].set_xticks([]) #tick_params(labelsize=15)
        axes[rows][COL].set_yticks([])
        
        COL += 1
        
        #plot information flow
        if source == "point":            
            input_signal_cord = np.array([[0, 0, cr.r_pc]])        
            input_signal_idx = cr.coordinate_to_index(input_signal_cord)  
        
        elif source == "spherical":
            input_signal_idx = cr.pc_idx
            input_signal_cord = cr.index_to_coordinate(input_signal_idx) 
        
        cellreservoir = cr.empty_CellReservoir_grid()              
        potential_map = cr.potential(cr.organelle_idx, input_signal_cord)
        cellreservoir[cr.vertex_idx] = potential_map
        cr.initiate_signalmap() 
        cr.initiate_statemap()
        cr.forward_one_step(cellreservoir, input_signal_idx)  
        
        #subplots
        axes[rows][COL].imshow(cr.signal_map[int(cr.signal_map.shape[0]/2),:,:], 
                                    norm = colors.LogNorm(vmin=1E-5, vmax=50))#vmin=1E-5, ))
        #im.set_cmap('nipy_spectral')
        axes[rows][COL].set_title(f'Signal Map CS {round(cs_vol*100)}%', fontsize=15)   
        axes[rows][COL].tick_params(labelsize=30)
        axes[rows][COL].set_xticks([]) #tick_params(labelsize=15)
        axes[rows][COL].set_yticks([])
        
        if COL == 3:
            COL = 0
            rows += 1
        else:
            COL += 1
    
    plt.tight_layout()
    plt.savefig(f'./plot/SignalDistribution_CSVol_{source}_source.svg', bbox_inches="tight")
    plt.savefig(f'./plot/SignalDistribution_CSVol_{source}_source.png', bbox_inches="tight")

def percolation_analysis(source = 'point',
                         no_trials = 100, 
                         lower_vol = 1, 
                         upper_vol = 15):
    '''
    Analyze the minimum cytoskeleton volume required for signal percolation in between
    peripherial cytoplasm and cell organelle. Information dyanmics experiment is repeated
    100 times on each random configuration of cytoskeleton volume ranging from 1% to 15%. 

    Parameters
    ----------
    source: str, categorical, optional
        extracellular K+ distribution type. The default is "spherical".

    '''
  
    df = pd.DataFrame(columns = range(no_trials), 
                      index = range(lower_vol, upper_vol+1))
    
    for cs_vol in range(lower_vol, upper_vol+1):
        for repeat in range(no_trials):
            print(f'CS Vol: {cs_vol} | Trial: {repeat}')
            cr = cdm.RC(cs_frac = cs_vol/100)
            
            if source == "point":            
                input_signal_cord = np.array([[0, 0, cr.r_pc]])        
                input_signal_idx = cr.coordinate_to_index(input_signal_cord)  
            
            elif source == "spherical":
                input_signal_idx = cr.pc_idx
                input_signal_cord = cr.index_to_coordinate(input_signal_idx) 
   
            cellreservoir = cr.empty_CellReservoir_grid()              
            potential_map = cr.potential(cr.organelle_idx, input_signal_cord)
            cellreservoir[cr.vertex_idx] = potential_map
            cr.initiate_signalmap() 
            cr.initiate_statemap()
            cr.forward_one_step(cellreservoir, input_signal_idx)  
            cell_with_co_current = cr.get_co_signal()    
            
            if np.amax(cell_with_co_current) == 0:            
                df.at[cs_vol, repeat] = 0
            else:
                df.at[cs_vol, repeat] = 1

    df.to_csv(f"./plot/Percolation_Analysis_{source}.csv")
    #df= pd.read_csv(f"./plot/Percolation_Analysis_{source}.csv", index_col=0)
    #plot
    df["Probability of PC-CO Connection"] = df.sum(axis=1)/no_trials
    df["Cytoskeleton Volume [%]"] = df.index
    unique_xs = sorted(df["Cytoskeleton Volume [%]"].unique())
    fig, ax = plt.subplots(figsize=(4.5,3), dpi=150)
    sns.lineplot(data = df, 
                 x = df["Cytoskeleton Volume [%]"].map(unique_xs.index), 
                 y = "Probability of PC-CO Connection", 
                 ax = ax, 
                 color = 'red')
    sns.barplot(data = df, 
                x = "Cytoskeleton Volume [%]", 
                y = "Probability of PC-CO Connection", 
                ax = ax, 
                color ='black')
    sns.lineplot(data = df, 
                 x = df["Cytoskeleton Volume [%]"].map(unique_xs.index), 
                 y = 0.5, 
                 linestyle = 'dashdot', 
                 color = "green", 
                 ax = ax)
    ax.set_title(f"Chance of Signal Percolation  for {source.title()} Source")
    plt.savefig(f'./plot/Percolation_Analysis_{source}.svg', bbox_inches="tight")
    plt.savefig(f'./plot/Percolation_Analysis_{source}.png', bbox_inches="tight")

  
    
if __name__ == "__main__": 
    print_equilibrium_potential()
    visualize_3D_cell_with_umap()
    plot_signal_flow_vs_cs_vol(source='point')
    plot_signal_flow_vs_cs_vol(source='spherical')
    percolation_analysis(source='point')
    percolation_analysis(source='spherical')
    
    
