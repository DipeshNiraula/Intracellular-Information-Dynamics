import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import copy    

class CellGraph():
    
    def __init__(self,
                 grid_size = 0.5,#[um]
                 r_cell_membrane = 10.0,#[um]
                 r_peripheral_cytoplasm = 9.0,#[um]
                 r_central_organelle = 2.5,#[um]
                 cs_frac = 0.2,
                 empty_shape = (1,3),
                 eps = 0.1,
                 center = [0,0,0],                 
                 ):
        self.grid_size = grid_size         
        self.no_grid = int(((2*r_cell_membrane)/grid_size)+1)
        
        self.r_cm = r_cell_membrane
        self.r_pc = r_peripheral_cytoplasm
        self.r_co = r_central_organelle 
        self.cs_frac = cs_frac        
        
        self.index_origin_in_coordinate = np.array([-1*r_cell_membrane, -1*r_cell_membrane, -1*r_cell_membrane])
    
        self.emptyshape = empty_shape
   
        self.eps = eps
        self.cell_center = center       
    
    def empty_CellGraph_grid(self):
        '''  
        Initialize cellgraph CG(V,E)
        
        Returns
        -------
        cellgraph: 3D numpy array of size 2n-1 by 2n-1 by 2n-1  
            Initial cellgraph is a zero numpy array          
        '''
        return np.zeros((2*self.no_grid-1, 2*self.no_grid-1, 2*self.no_grid-1))
    
    def empty_Cell(self):
        '''  
        Initialize cell for vertex properties such as potential and current signal
        
        Returns
        -------
        cell: 3D numpy array of size n by n by n           
            Initial cell is a zero numpy array   
        '''
        return np.zeros((self.no_grid, self.no_grid, self.no_grid))
    
    def initiate_conductance(self, mu=0.05, sigma=0.5):
        '''
        Initiate log-normally distributed  cellgraph conductance
        
        Parameters
        ----------
        mu : float, optional
            mean of log normal distribution. The default is 0.05.
        sigma : float, optional
            deviation of log normal distribution. The default is 0.5.

        Returns
        -------
        conductance map: 3D numpy array of size 2n-1  by 2n-1 by 2n-1  
           log-normally distributed conductance for a cellgraph

        '''
        return np.random.lognormal(mu, sigma, (2*self.no_grid-1, 2*self.no_grid-1, 2*self.no_grid-1))
    
    def vertex_index(self):
        ''' 
        Return grid indices of cellgraph vertices. Vertex live in even indices (2i,2j,2k).
        
        Returns
        -------
        I_vertex, J_vertex, K_vertex : 3D numpy array of size n by n by n
            I_vertex, J_vertex, and K_vertex are cell grid indices in x, y, and z directions

        '''
        I_vertex, J_vertex, K_vertex = 2*np.indices((self.no_grid, self.no_grid, self.no_grid))
        return I_vertex, J_vertex, K_vertex 
    
    def sphere_surface(self, X, Y, Z, radius):
        '''
        Return cell with spherical cell organelle surfaces and their indices.

        Parameters
        ----------
        X, Y, Z : 3D numpy array of size n by n by n
            X, Y, and Z are cell grid co-ordinates in x, y, and z directions
        radius : float
            radius of a spherical surface.

        Returns
        -------
        sphere_surface_ : 3D numpy array of size n by n by n
            A n by n by n cell masked with spherical surface voxel as True and remaining voxels as False.
        idx : 2D numpy aray of size rows by 3 where rows is the number of True voxels
            Indices of spherical surface voxel in 2D.

        '''
        sphere = (X-self.cell_center[0])**2 + (Y-self.cell_center[1])**2 + (Z-self.cell_center[0])**2 
        outer_sphere = sphere < (radius+self.grid_size/2)**2
        inner_sphere = sphere < (radius-self.grid_size/2)**2
        sphere_surface_ = np.bitwise_xor(outer_sphere, inner_sphere)
        x,y,z = np.where(sphere_surface_== True)
        idx = np.hstack((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)))
        return sphere_surface_ , idx
    
    def cytoskeleton(self, X, Y, Z):
        '''
        Return cell with frac percentage of randomly distributed cytoskeleton by volume in between r_pc and r_co. 
        
        Parameters
        ----------
        X, Y, Z : 3D numpy array of size n by n by n
            X, Y, and Z are cell grid co-ordinates in x, y, and z directions


        Returns
        -------
        cs : 3D numpy array of size n by n by n
            A n by n by n cell masked with randomly chosen voxels as True and remaining voxels as False.
        idx : 2D numpy array of size rows by 3 where rows is the number of True voxels
            Indices of cytoskeleton voxel in 2D.

        '''
        sphere = (X-self.cell_center[0])**2 + (Y-self.cell_center[1])**2 + (Z-self.cell_center[0])**2 
        outer_sphere = sphere < (self.r_pc-self.grid_size/2)**2
        inner_sphere = sphere < (self.r_co+self.grid_size/2)**2
        cs = np.bitwise_xor(outer_sphere, inner_sphere)
        # randomly assign false to 1-frac vertices
        x,y,z = np.where(cs==True)
        idx = np.hstack((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)))  
        cs_idx = idx[np.random.choice(np.arange(idx.shape[0]), 
                                 size = np.rint((1-self.cs_frac)*idx.shape[0]).astype(int), 
                                 replace = False)]
        cs[cs_idx[:,0], cs_idx[:,1], cs_idx[:,2]] = False 
        # select cytoskeleton vertices
        x,y,z = np.where(cs==True)
        idx = np.hstack((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)))  
        return cs, idx
    
    def coordinate_to_index(self, co_ordinate):
        '''
        Co-ordinate to index Transformation
        
        Parameters
        ----------
        co_ordinates: 2D array of size rows by 3
            {(x,y,z)}

        Returns
        -------
        index : 2D array of size rows by 3
            index transform of coordinates
    
        '''
        transformation = np.identity(3)
        transformation[0,0] = 1/self.grid_size
        transformation[1,1] = 1/self.grid_size
        transformation[2,2] = 1/self.grid_size
        
        index = (co_ordinate - self.index_origin_in_coordinate) @ transformation
        #rint: Rounded to Nearest Integer
        index = (np.rint(index)).astype(int)
        return index

    def index_to_coordinate(self, index):
        '''
        Index to co-ordinate Transformation
        
        Parameters
        ----------
        index: 2D Array of size rows by 3
            {(i,j,k)}
        Returns
        -------
        co_ordinate : 2D array of size rows by 3
            co-ordinate transform of grid index
    
        '''
        transformation = np.identity(3)
        transformation[0,0] = self.grid_size
        transformation[1,1] = self.grid_size
        transformation[2,2] = self.grid_size
        co_ordinate = index @ transformation + self.index_origin_in_coordinate
        return co_ordinate
    
    def index_to_coordinate_grid(self, I, J, K):
        '''
        Index to coordinate grid transformation
        Parameters
        ----------
        I, J, K: 3D Array of size n by n by n
            I J and K are arrays of n 2D arrays of size n by n that contains grid indices in x, y, and z directions
        Returns
        -------
        X, Y, Z : 3D Array of size n by  by n
            X, Y, and Z are arrays of n 2D arrays of size n by n that contains grid co-ordinates in x, y, and z directions
    
        '''
        X = I* self.grid_size + self.index_origin_in_coordinate[0]
        Y = J* self.grid_size + self.index_origin_in_coordinate[1]
        Z = K* self.grid_size + self.index_origin_in_coordinate[2]
        return X, Y, Z

    
    def cell_organelles(self):  
        '''
        Return cell with spherical cell organelles and randomly distribute cytoplasm

        Returns
        -------
        cm_surface, pc_surface, co_surface : 3D numpy array of size n by n by n
            An n by n by n cell masked with spherical surfaces representing cell membrane, peripheral cytoplasm, and central organelle as True and remaining voxels as False.
        cs : 3D numpy array of size nxnxn.
            An n by n by n cell masked with randomly chosen cytoskeleton voxels as True and remaining voxels as False.
        cm_idx, pc_idx, co_idx, cs_idx : 2D numpy aray of size rows by 3 where rows is the number of True voxels
            Index of organelles surface and random cytoskeleton voxels in 2D.

        '''
        I, J, K = np.indices((self.no_grid, self.no_grid, self.no_grid))
        X, Y, Z = self.index_to_coordinate_grid(I, J, K)   
        cm_surface, cm_idx = self.sphere_surface(X, Y, Z, self.r_cm)
        pc_surface, pc_idx = self.sphere_surface(X, Y, Z, self.r_pc)
        co_surface, co_idx = self.sphere_surface(X, Y, Z, self.r_co)
        cs, cs_idx = self.cytoskeleton(X, Y, Z)        
        return cm_surface, pc_surface, co_surface, cs, cm_idx, pc_idx, co_idx, cs_idx 
    
    def vector_length(self, vector):
        '''
        Calculate vector length |v| = (x*x+y*y+z*z)**(1/2)
        
        Parameters
        ----------
        vector : 2D array of size rows by n
            vector in 3D space
        Returns
        -------
        vector_length: 1D array of size rows
            vector length calculated by einstein-summation function

        '''
        return np.sqrt(np.einsum('ij,ij->i', vector, vector))

    def potential(self, cell, vertices, source, charge=1):
        '''
        Return cell potential map for a source charge distribution following 1/d law
        
        Parameters
        ----------
        cell : 3D numpy array of size nxnxn
            A n by n by n cell masked with organelle voxels as True and remaining voxels as False.
        vertices : 2D array of size rows by 3
            Index of organelles surface and random cytoskeleton voxels in 2D.
        source : array of size 1 by 3
            Co-ordinate of source charge
        charge : float, optional
            charge strength. The default is 1.

        Returns
        -------
        cell : 3D numpy array of size nxnxn
            An n by n by n cell with potential value on voxels with cell organelles.

        '''
        co_ordinate = self.index_to_coordinate(vertices) 
        distance = self.vector_length(co_ordinate - source)
        potential = np.divide(charge, distance)
        cell[vertices[:,0], vertices[:,1], vertices[:,2]] = potential
        
        #for the source grid, to avoid infinity potential, we consider the potential at r = grid_size/2
        source_idx = self.coordinate_to_index(source)
        cell[source_idx[:,0], source_idx[:,1], source_idx[:,2]] = np.divide(charge, self.grid_size/2)
        return cell
    
    def ohms_law(self, conductance, source_pot, target_pot):
        '''
        Return edge current value I=GdV. Current can only flow from higher potential to lower potential.
        
        Parameters
        ----------
        conductance : 2D array of size rows by 1
            Conductance of edges that lies in between source potential and target potential.
        source_pot : array of size 1 by 1
            Potential value of the source voxel
        target_pot : 2D array of size rows by 1
            Potential value of the target voxels

        Returns
        -------
        current: 2D array of size rows by 1
            Nearest neighbour edge current value in between source and target voxel.

        '''
        current =  conductance*(source_pot-target_pot)
        #only take positive current- forward process
        return np.clip(current, a_min=0, a_max=None)       
        
    def r_ball(self, cellgraph, source_idx, step=2):    
        '''
        Return 1-voxel nearest neighbour of a vertex. In cellgraph step = 2 units.
        
        Parameters
        ----------
        cellgraph : 3D numpy array of size 2n-1 by 2n-1 by 2n-1
            cellgraph are masked with organelle vertices as True.
        source_idx : 1D numpy array of size 3
            index of the source voxel whose immediate nearest neighbour is to be found
        step : float, optional
            no of steps around the source considered as the radius of r-ball. The default is 2.

        Returns
        -------
        NN_idx : 2D array of size rows by 3
            Vertex index of the nearest neighbours of the source. 
            Only returns the nearest neighbours that has been occupied by a voxel.

        '''
        #check for endcase
        start_i = np.where(source_idx[0]-step < 0, 0, source_idx[0]-step)
        start_j = np.where(source_idx[1]-step < 0, 0, source_idx[1]-step)
        start_k = np.where(source_idx[2]-step < 0, 0, source_idx[2]-step)
        #plus one because python uses half open convention, i.e [start, end)
        end_i = np.where(source_idx[0]+1+step > cellgraph.shape[0], cellgraph.shape[0]+1, source_idx[0]+1+step)
        end_j = np.where(source_idx[1]+1+step > cellgraph.shape[1], cellgraph.shape[1]+1, source_idx[1]+1+step)
        end_k = np.where(source_idx[2]+1+step > cellgraph.shape[2], cellgraph.shape[2]+1, source_idx[2]+1+step)
        
        #select grid with stepsize 2        
        I_NN, J_NN, K_NN = np.mgrid[start_i:end_i:step, start_j:end_j:step, start_k:end_k:step]    
        NN_idx = np.hstack((I_NN.reshape(-1,1),
                            J_NN.reshape(-1,1), 
                            K_NN.reshape(-1,1))) 
        #check if a vertex exist in the grid location
        NN_idx = np.array([i for i in NN_idx if cellgraph[i[0], i[1], i[2]] and not np.equal(i, source_idx).all()]).reshape(-1,3)
        return NN_idx
    
    def get_central_organelle_current(self, current_map, co_idx):
        '''
        Returns surface current map of central organelle
        
        Parameters
        ----------
        current_map : 3D numpy array of size n by n by n
            current map of cell at time t
        co_idx : 2D array of size rows by 3
            cell index for central organelle surface
        
        Returns
        -------
        co_surface_current_map: 3D numpy array of size n by n by n
            current map of central organelle surface at a given time t

        '''
        co_surface_current_map = self.empty_Cell()
        co_surface_current_map[co_idx[:,0], co_idx[:, 1], co_idx[:, 2]] = current_map[co_idx[:,0], co_idx[:,1], co_idx[:,2]]
        
        return co_surface_current_map
            
    def forward(self, cellgraph, conductance, current_map, signal_idx):
        '''
        Returns one forward step of information flow
        
        Parameters
        ----------
        cellgraph : 3D numpy array of size 2n-1 by 2n- by 2n-1 
            Cellgraph vertices contain the potential map and edges contain the edge-current at time t
        conductance : 3D numpy array of size 2n-1 by 2n-1 by 2n-1
            Randomly generated conductance values for edges 
        current_map : 3D numpy array of size n by n by n
            current map of cell at time t
        signal_idx : 2D array of size rows by 3
            index of vertices that has received signal at time t 

        Returns
        -------
        signaled_vertex : 2D array of size rows by 3
            index of vertices that has received signal at time t+1 
        cellgraph : 3D numpy array of size 2n-1 by 2n- by 2n-1 
            Cellgraph vertices contain the potential map and edges contain the edge-current at time t+1
        current_map : 3D numpy array of size n by n by n
            current map of cell at time t+1
        '''
        # initialization of signaled vertex with np.array((-100,-100,-100)), will be removed at the end
        signaled_vertex = np.array((-100,-100,-100))
        for i in range(signal_idx.shape[0]):
            #find nearest nodes and their potential
            NN_vertex_idx = self.r_ball(cellgraph, signal_idx[i]).astype(int) 
            source_pot = cellgraph[signal_idx[i][0], signal_idx[i][1], signal_idx[i][2]]
            NN_target_pot =  cellgraph[NN_vertex_idx[:,0], NN_vertex_idx[:,1], NN_vertex_idx[:,2]]
            
            #find edges between two nodes
            NN_edge_idx = np.divide(signal_idx[i]+NN_vertex_idx, 2).astype(int) 
            NN_conductance = conductance[NN_edge_idx[:,0], NN_edge_idx[:,1], NN_edge_idx[:,2]]            
            
            #ohms law
            NN_edge_current = self.ohms_law(NN_conductance, source_pot, NN_target_pot)

            #an edge can represent multiple conductance, in that case we simply replace edge currents value with new one
            #although there is no need for storing edge_current           
            cellgraph[NN_edge_idx[:,0], NN_edge_idx[:,1], NN_edge_idx[:,2]] += NN_edge_current
            
            #current at a vertex is sum of all incoming currents
            current_idx = np.divide(NN_vertex_idx, 2).astype(int) 
            current_map[current_idx[:,0], current_idx[:,1], current_idx[:,2]] += NN_edge_current
            
            signaled_vertex = np.vstack((signaled_vertex, NN_vertex_idx))        
        # store signaled vertex only once
        signaled_vertex = np.unique(signaled_vertex, axis=0).astype(int) 
        # remove source vertex
        signaled_vertex = np.array([signaled_vertex[i] for i in range(signaled_vertex.shape[0]) if not any(np.equal(signal_idx, signaled_vertex[i]).all(axis=1))]).reshape(-1,3)
        # remove np.array((-100,-100,-100))
        signaled_vertex = signaled_vertex[1:]
        return signaled_vertex, cellgraph, current_map    
        
##--------------plot helper functions----------------------------------------##
def remove(string):
    return string.replace(" ", "_")

def plot_2D(array_2D, title, log=False, vmin=1E-5, vmax=50):
    fig, ax = plt.subplots(figsize = (10,10), dpi=150) 
    if log:
        im = plt.imshow(array_2D, norm = colors.LogNorm(vmin=vmin, vmax=vmax))
    else:
        im = plt.imshow(array_2D)
    im.set_cmap('nipy_spectral')
    plt.title(title, fontsize = 40)
    cbar_ax = fig.add_axes([0.95, 0.1, 0.05, 0.8])#[left, bottom, width, height]
    cb = plt.colorbar(im, cax=cbar_ax)
    ax.tick_params(labelsize=30)
    cb.ax.tick_params(labelsize=30)
    plt.show()
    plt.savefig('./plot/{}.svg'.format(remove(title)))
    plt.savefig('./plot/{}.png'.format(remove(title)))

def plot_2D_cellgraph(array_2D, title):
    ticks=['','Cell Membrane', 'Peripheral Cytoplasm', 'Cytoskeleton', 'Central Organelle']
    fig, ax = plt.subplots(figsize = (10,10), dpi=150) 
    plt.title(title, fontsize = 40)
    
    im = plt.imshow(array_2D)
    im.set_cmap('nipy_spectral')
    cbar_ax = fig.add_axes([0.95, 0.1, 0.05, 0.8])#[left, bottom, width, height]
    cb = plt.colorbar(im, cax=cbar_ax, ticks=np.linspace(0,4,5), boundaries=np.linspace(0,4,5)+0.5)
    cb.ax.set_yticklabels(["{}".format(i) for i in ticks])
    
    ax.tick_params(labelsize=30)
    cb.ax.tick_params(labelsize=30)
    plt.show()
    plt.savefig('./plot/{}.svg'.format(remove(title)))
    plt.savefig('./plot/{}.png'.format(remove(title)))

def plot_cell_graph_minus_vertex_conductance_map(conductance_map, no_grid):
    
    conductance_plot = copy.deepcopy(conductance_map)
    conductance_plot[vertex_idx] = 0
    index_x, index_y, index_z = np.indices((2*no_grid-1, 2*no_grid-1, 2*no_grid-1))
    #for cellgraph center of the sphere [x_0, y_0, z_0] = [0,0,0] translates to [i_0, j_0, k_0] = [cg.no_grid, cg.no_grid, cg.no_grid]
    sphere = (index_x-no_grid)**2 + (index_y-no_grid)**2 + (index_z-no_grid)**2 
    #for cellgraph cell membrane radius = cg.no_grid
    outer_sphere = sphere <= (no_grid)**2
    x,y,z = np.where(outer_sphere == False)
    idx = np.hstack((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)))
    conductance_plot[idx[:,0], idx[:,1], idx[:,2]] = 0
    
    plot_2D(conductance_plot[int(conductance_map.shape[0]/2),:,:],
            title = 'CellGraph Conductance Map',
            log=True, vmin=np.amin(conductance_plot.flatten()[conductance_plot.flatten()>0]), 
            vmax=np.max(conductance_plot))
##---------------------------------------------------------------------------##


if __name__ == "__main__": 
    # initialization
    cg = CellGraph()
    cellgraph = cg.empty_CellGraph_grid()        
    
    # build cell organelle structures 
    cm_surface, pc_surface, co_surface, cs, cm_idx, pc_idx, co_idx, cs_idx = cg.cell_organelles()
    cell = cm_surface + pc_surface + co_surface + cs    
    vertex_idx = cg.vertex_index()
    cellgraph[vertex_idx] = cell
        
    #plot cell structures    
    cellgraph_plot = cg.empty_CellGraph_grid()
    cell_plot = 1*cm_surface + 2*pc_surface + 3*cs+ 4*co_surface 
    cellgraph_plot[vertex_idx] = cell_plot
    
    sns.set_style("whitegrid", {'axes.grid' : False})
    plot_2D_cellgraph(cellgraph_plot[vertex_idx][int(cellgraph_plot.shape[0]/4),:,:],
                      title = 'Cell Geometry')
    
    #Define source charge    
    input_signal = np.array([[0,0,cg.r_pc]])
    #Collect organelle idx    
    organelle_idx = np.vstack((cm_idx, pc_idx, co_idx, cs_idx))    
    #initialize a cell
    potential_map = cg.empty_Cell()
    # calculate potential map
    potential_map = cg.potential(potential_map, organelle_idx, input_signal)
    # set the vertex with potential_map
    cellgraph[vertex_idx] = potential_map
    conductance_map = cg.initiate_conductance()
    
    
    # plot potential and conductance map
    plot_2D(cellgraph[int(cellgraph.shape[0]/2),:,:],
            title = 'CellGraph Potential Map',
            log=True, vmin=np.amin(cellgraph.flatten()[cellgraph.flatten()>0]), 
            vmax=np.max(cellgraph))
    
    plot_2D(potential_map[int(potential_map.shape[0]/2),:,:], 
            title = 'Potential Map',
            log=True, vmin=np.amin(potential_map.flatten()[potential_map.flatten()>0]), 
            vmax=np.max(potential_map))    
    
    plot_2D(conductance_map[int(conductance_map.shape[0]/2),:,:],
            title = 'CellGraph Conductance Map',
            log=True, vmin=np.amin(conductance_map.flatten()[conductance_map.flatten()>0]), 
            vmax=np.max(conductance_map))
    
    plot_cell_graph_minus_vertex_conductance_map(conductance_map, cg.no_grid)
    
    
    # Information Dynamics 
    # initialize current map
    current_map = cg.empty_Cell()
    input_signal_idx = cg.coordinate_to_index(input_signal)    
    # vertex indices in cell graph is 2*voxel index of cell        
    signal_tracker_0 = 2*input_signal_idx
    #initiate signal tracker
    signal_time_tracker = [signal_tracker_0]
    TOTAL_TIME_STEP = 40
    FIG_COL = 5
    import math
    fig_rows = math.ceil(TOTAL_TIME_STEP/FIG_COL)
    fig, axes = plt.subplots(fig_rows, FIG_COL, figsize=(15, 24), dpi=150)
    
    #sub-figure counter
    rows = 0
    COL = 0
    
    for signal_time in range(TOTAL_TIME_STEP):
        print('timestep: ', signal_time)
        # one forward step of information dynamics
        signaled_idx, cellgraph, current_map  = cg.forward(cellgraph, conductance_map, current_map, signal_time_tracker[signal_time])    
        if signaled_idx.shape[0] > 0:
            #store signaled vertex
            signal_time_tracker.append(signaled_idx)
            #subplots
            im = axes[rows][COL].imshow(current_map[int(current_map.shape[0]/2),:,:], norm = colors.LogNorm(vmin=1E-5, vmax=50))
            im.set_cmap('nipy_spectral')
            axes[rows][COL].set_title('t = {}'.format(signal_time+1), fontsize=25)   
            axes[rows][COL].tick_params(labelsize=30)
            axes[rows][COL].set_xticks([]) #tick_params(labelsize=15)
            axes[rows][COL].set_yticks([])
            if COL == FIG_COL-1:
                rows+=1
                COL=0
            else:
                COL += 1
        else:
            print('Signal cant flow anymore')
            break    
    
    #plot information flow per time step
    fig.subplots_adjust(right=0.9, top=0.95)
    cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.8])#[left, bottom, width, height]
    cb = fig.colorbar(im, cax=cbar_ax)
    fig.suptitle('Information Flow', fontsize = 40)
    cb.ax.tick_params(labelsize=30)
    
    plt.savefig('./plot/Information_Dynamics.svg')
    plt.savefig('./plot/Information_Dynamics.png')
    
    #plot information flow at time =  TOTAL_TIME_STEP
    plot_2D(current_map[int(current_map.shape[0]/2),:,:], 
            title= 'Information Signal Map at 40 time steps', 
            log=True)
    
    #plot central organelle surface current at time =  TOTAL_TIME_STEP
    cell_with_co_current = cg.get_central_organelle_current(current_map, co_idx)    
    plot_2D(cell_with_co_current[int(cell_with_co_current.shape[0]/2),:,:],
            title = 'Organelle Surface Signal Map',
            log=True, vmin=np.amin(cell_with_co_current.flatten()[cell_with_co_current.flatten()>0]), 
            vmax=np.max(cell_with_co_current))

    # plot histograms
    g = sns.displot(cell_with_co_current.flatten()[cell_with_co_current.flatten() > 0])#, kde=True)

    g.set_axis_labels("Signal Strength", "Count", fontsize=20)
    g.fig.suptitle('Organelle Surface Signal Histogram', fontsize=20)  
    g.ax.tick_params(labelsize=20)
    
    h = sns.displot(conductance_map.flatten()[conductance_map.flatten() > 0])#, kde=True)
    
    h.set_axis_labels("Conductance", "Count", fontsize=20)
    h.fig.suptitle('LogNormal Conductance', fontsize=20)
    h.ax.tick_params(labelsize=20)
