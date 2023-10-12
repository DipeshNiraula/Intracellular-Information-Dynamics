import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import copy    
import math
#np.random.seed(12345678)
class CellReservoir:    
    def __init__(self,
                 grid_size = 0.5,#[um]
                 r_cell_membrane = 10.0,#[um]
                 r_peripheral_cytoplasm = 9.0,#[um]
                 r_central_organelle = 2.5,#[um]
                 cs_frac = 0.2,
                 empty_shape = (1,3),
                 center = [0,0,0],                 
                 ):       
        self.r_cm = r_cell_membrane
        self.r_pc = r_peripheral_cytoplasm
        self.r_co = r_central_organelle 
        self.cs_frac = cs_frac        
        self.grid_size = grid_size         
        self.no_grid = self._calculate_no_grid()
        self.index_origin_in_coordinate = self._index_origin_in_coordinate()
        self.emptyshape = empty_shape
        self.cell_center = center  
    
    def _calculate_no_grid(self):
        '''
        Returns
        -------
        Integer
            Calculates no of grid for a given cell radius and unit grid size

        '''
        return int(((2*self.r_cm)/self.grid_size)+1)
    
    def _index_origin_in_coordinate(self):
        '''
        Returns
        -------
        Numpy Array of size 3 by 1
            initiates the origin of the cell graph as (-r_cm, -r_cm, -r_cm)
        '''
        return np.array([-1*self.r_cm, -1*self.r_cm, -1*self.r_cm])
    
    def empty_CellReservoir_grid(self):
        '''  
        Initialize cellreservoir cr(V,E)
        
        Returns
        -------
        cellreservoir: 3D numpy array of size 2n-1 by 2n-1 by 2n-1  
            Initial cellreservoir is a zero numpy array          
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
    
    def initiate_conductance(self, mu=0.1, sigma=0.5):
        '''
        Initiate log-normally distributed  cellreservoir conductance
        
        Parameters
        ----------
        mu : float, optional
            mean of log normal distribution. The default is 0.05.
        sigma : float, optional
            deviation of log normal distribution. The default is 0.5.

        Returns
        -------
        conductance map: 3D numpy array of size 2n-1  by 2n-1 by 2n-1  
           log-normally distributed conductance for a cellreservoir

        '''
        return np.random.lognormal(mu, sigma, (2*self.no_grid-1, 2*self.no_grid-1, 2*self.no_grid-1))
    
    def vertex_index(self):
        ''' 
        Return grid indices of cellreservoir vertices. Vertex live in even indices (2i,2j,2k).
        
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

    def potential(self, vertices, source_cord, charge=[1], k=1):
        '''
        Return cell potential map for a source charge distribution following 1/d law
        
        Parameters
        ----------
        vertices : 2D array of size rows by 3
            Index of organelles surface and random cytoskeleton voxels in 2D.
        source_cord : array of size rows by 3
            Co-ordinate of source charge
        charge : list of float, optional
            charge strength. The default is 1.
        k : float, optional
            inverse debye length. The default is 1 um^-1.
        Returns
        -------
        cell : 3D numpy array of size nxnxn
            An n by n by n cell with potential value on voxels with cell organelles.

        '''
        cell = self.empty_Cell()
        co_ordinate = self.index_to_coordinate(vertices)
        #populating charge list with the number of source
        charge = charge*source_cord.shape[0]
        
        if source_cord.shape[0] != len(charge):
            print('Error: Source co-ordinate and charge have different size')
            return
        
        for s_cord, c in zip(source_cord, charge):
            distance = self.vector_length(co_ordinate - s_cord.reshape(1,-1)) + 1E-1 #to avoid infinity potential,
            potential = np.divide(c*np.exp(-k*distance), distance)
            cell[vertices[:,0], vertices[:,1], vertices[:,2]] += potential

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
        #return current
        
    def r_ball(self, cellreservoir, source_idx, visited, step=2):    
        '''
        Return 1-voxel nearest neighbour of a vertex. In cellreservoir step = 2 units.
        
        Parameters
        ----------
        cellreservoir : 3D numpy array of size 2n-1 by 2n-1 by 2n-1
            cellreservoir are masked with organelle vertices as True.
        source_idx : 1D numpy array of size 3
            index of the source voxel whose immediate nearest neighbour is to be found
        visited : set of 3-tuples
            set of all incides visited during previous time steps
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
        end_i = np.where(source_idx[0]+1+step > cellreservoir.shape[0], 
                         cellreservoir.shape[0]+1, 
                         source_idx[0]+1+step)
        end_j = np.where(source_idx[1]+1+step > cellreservoir.shape[1], 
                         cellreservoir.shape[1]+1, 
                         source_idx[1]+1+step)
        end_k = np.where(source_idx[2]+1+step > cellreservoir.shape[2], 
                         cellreservoir.shape[2]+1, 
                         source_idx[2]+1+step)
        
        #select grid with stepsize 2        
        I_NN, J_NN, K_NN = np.mgrid[start_i:end_i:step, 
                                    start_j:end_j:step, 
                                    start_k:end_k:step]    
        NN_idx = np.hstack((I_NN.reshape(-1,1),
                            J_NN.reshape(-1,1), 
                            K_NN.reshape(-1,1))) 
        #check if a vertex exist in the grid location
        NN_idx = np.array([i for i in NN_idx if cellreservoir[i[0], i[1], i[2]] \
                           and not np.equal(i, source_idx).all() \
                               and tuple(i) not in visited]).reshape(-1,3)
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
        co_surface_current_map[co_idx[:,0], co_idx[:, 1], co_idx[:, 2]] = current_map[co_idx[:,0], 
                                                                                      co_idx[:,1], 
                                                                                      co_idx[:,2]]
        
        return co_surface_current_map
            
    def tanh(self, current_map_arr):
        '''
        Returns activated current map array squashed in between -1 to 1
        
        Parameters
        ----------
        arr : 1D numpy array of size row
            Array of current signal

        Returns
        -------
        1D numpy array of size row
            Array of hyperpolic tangent of current signal calcluated as exp(2i-1)/exp(2i+1)

        '''
        return np.divide(np.exp(2*current_map_arr)-1, np.exp(2*current_map_arr)+1)
    
    def forward(self, 
                cellreservoir, 
                conductance, 
                current_map, 
                state_map, 
                signal_idx, 
                visited, 
                memory_retention_rate=0.9): 
        '''
        Returns one forward step of information flow from vertex to its nearest neighbour
        
        Parameters
        ----------
        cellreservoir : 3D numpy array of size 2n-1 by 2n- by 2n-1 
            cellreservoir vertices contain the potential map and edges contain the edge-current at time t
        conductance : 3D numpy array of size 2n-1 by 2n-1 by 2n-1
            Randomly generated conductance values for edges 
        current_map : 3D numpy array of size n by n by n
            current map of cell at time t
        state_map : 3D numpy array of size n by n by n
            state map of cell at time t that aggregates signals via exponential moving averaging
        signal_idx : 2D array of size rows by 3
            index of vertices that has received signal at time t 
        visited : set of 3-tuples
            set of all incides visited during previous time steps

        Returns
        -------
        signaled_vertex : 2D array of size rows by 3
            index of vertices that has received signal at time t+1 
        cellreservoir : 3D numpy array of size 2n-1 by 2n- by 2n-1 
            cellreservoir vertices contain the potential map and edges contain the edge-current at time t+1
        current_map : 3D numpy array of size n by n by n
            current map of cell at time t+1
        '''
        # initialization of signaled vertex with np.array((-100,-100,-100)), will be removed at the end
        signaled_vertex = np.array((-100,-100,-100))
        for i in range(signal_idx.shape[0]):
            #find nearest nodes and their potential
            NN_vertex_idx = self.r_ball(cellreservoir, 
                                        signal_idx[i], 
                                        visited).astype(int) 
            source_pot = cellreservoir[signal_idx[i][0], 
                                       signal_idx[i][1], 
                                       signal_idx[i][2]]
            NN_target_pot = cellreservoir[NN_vertex_idx[:,0], 
                                          NN_vertex_idx[:,1], 
                                          NN_vertex_idx[:,2]]
            
            #find edges between two nodes
            NN_edge_idx = np.divide(signal_idx[i]+NN_vertex_idx, 2).astype(int) 
            NN_conductance = conductance[NN_edge_idx[:,0], 
                                         NN_edge_idx[:,1], 
                                         NN_edge_idx[:,2]]            
            
            #ohms law
            NN_edge_current = self.ohms_law(NN_conductance, 
                                            source_pot,
                                            NN_target_pot)

            #an edge can represent multiple conductance, in that case we simply replace edge currents value with new one
            #although there is no need for storing edge_current           
            cellreservoir[NN_edge_idx[:,0], NN_edge_idx[:,1], NN_edge_idx[:,2]] += NN_edge_current
            
            #current at a vertex is sum of all incoming currents
            current_idx = np.divide(NN_vertex_idx, 2).astype(int) 
            current_map[current_idx[:,0], current_idx[:,1], current_idx[:,2]] += NN_edge_current
            
            #update state map as s(t) = beta*s(t-1) + (1-beta)*tanh(current(t))
            #note the connections are determined by the NN
            state_map[current_idx[:,0], current_idx[:,1], current_idx[:,2]] = \
                memory_retention_rate*state_map[current_idx[:,0], 
                                                current_idx[:,1], 
                                                current_idx[:,2]] + \
                 (1-memory_retention_rate)* self.tanh(current_map[current_idx[:,0], 
                                                                  current_idx[:,1], 
                                                                  current_idx[:,2]])
            
            signaled_vertex = np.vstack((signaled_vertex, NN_vertex_idx))        
        # store signaled vertex only once
        signaled_vertex = np.unique(signaled_vertex, axis=0).astype(int) 
        # remove source vertex
        signaled_vertex = np.array([signaled_vertex[i] for i in \
                                    range(signaled_vertex.shape[0]) if not \
                                        any(np.equal(signal_idx, 
                                                     signaled_vertex[i]).all(axis=1))]).reshape(-1,3)
        # remove np.array((-100,-100,-100))
        signaled_vertex = signaled_vertex[1:]
        return signaled_vertex, cellreservoir, current_map, state_map    
        
##--------------plot helper functions----------------------------------------##
def remove(string):
    return string.replace(" ", "_").replace(":", "_")

def plot_2D(array_2D, title, log=False, vmin=1E-5, vmax=50):
    fig, ax = plt.subplots(figsize = (10,10), dpi=150) 
    if log:
        im = plt.imshow(array_2D, norm = colors.LogNorm(vmin=vmin, vmax=vmax))
    else:
        im = plt.imshow(array_2D, norm = colors.Normalize(vmin=vmin, vmax=vmax))
    #im.set_cmap('nipy_spectral')
    plt.title(title, fontsize = 40)
    cbar_ax = fig.add_axes([0.95, 0.1, 0.05, 0.75])#[left, bottom, width, height]
    cb = plt.colorbar(im, cax=cbar_ax)
    ax.tick_params(labelsize=30)
    cb.ax.tick_params(labelsize=30)
    plt.show()
    fig.tight_layout()
    plt.savefig('./plot_publication/{}.svg'.format(remove(title)), bbox_inches="tight")
    plt.savefig('./plot_publication/{}.png'.format(remove(title)), bbox_inches="tight")

def plot_2D_cellreservoir(array_2D, title):
    ticks=['',
           'Cell Membrane', 
           'Peripheral Cytoplasm', 
           'Cytoskeleton', 
           'Central Organelle']
    fig, ax = plt.subplots(figsize=(10,10), dpi=150) 
    plt.title(title, fontsize=40)
    
    im = plt.imshow(array_2D)
    #im.set_cmap('nipy_spectral')
    cbar_ax = fig.add_axes([0.95, 0.1, 0.05, 0.75])#[left, bottom, width, height]
    cb = plt.colorbar(im, 
                      cax = cbar_ax, 
                      ticks = np.linspace(0,4,5), 
                      boundaries = np.linspace(0,4,5)+0.5)
    cb.ax.set_yticklabels(["{}".format(i) for i in ticks])
    
    ax.tick_params(labelsize=30)
    cb.ax.tick_params(labelsize=30)
    plt.show()
    fig.tight_layout()
    plt.savefig('./plot_publication/{}.svg'.format(remove(title)), 
                bbox_inches="tight")
    plt.savefig('./plot_publication/{}.png'.format(remove(title)), 
                bbox_inches="tight")

def plot_cell_graph_edge_conductance_map(conductance_map, no_grid, vertex_idx):    
    conductance_plot = copy.deepcopy(conductance_map)
    conductance_plot[vertex_idx] = 0
    index_x, index_y, index_z = np.indices((2*no_grid-1, 2*no_grid-1, 2*no_grid-1))
    #for cellreservoir center of the sphere [x_0, y_0, z_0] = [0,0,0] 
    #translates to [i_0, j_0, k_0] = [cr.no_grid, cr.no_grid, cr.no_grid]
    sphere = (index_x-no_grid)**2 + (index_y-no_grid)**2 + (index_z-no_grid)**2 
    #for cellreservoir cell membrane radius = cr.no_grid
    outer_sphere = sphere <= (no_grid)**2
    x,y,z = np.where(outer_sphere == False)
    idx = np.hstack((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)))
    conductance_plot[idx[:,0], idx[:,1], idx[:,2]] = 0
    
    plot_2D(conductance_plot[int(conductance_map.shape[0]/2),:,:],
            title = 'CellReservoir Conductance Map',
            log=True, vmin=np.amin(conductance_plot.flatten()[conductance_plot.flatten()>0]), 
            vmax=np.max(conductance_plot))
    
def plot_dist(map_, x_label, ylabel, title):
    ax = sns.displot(map_.flatten()[map_.flatten() > 0], color="red")#, kde=True)    
    ax.set_axis_labels(x_label, ylabel, fontsize=20)
    ax.fig.suptitle(title, fontsize=20)
    ax.ax.tick_params(labelsize=20)
    tx = ax.ax.xaxis.get_offset_text()
    tx.set_fontsize(20)
    ax.ax.ticklabel_format(axis='x', style='scientific', scilimits=(0,0))
    plt.tight_layout()
    plt.savefig('./plot_publication/{}.svg'.format(remove(title)), bbox_inches="tight")
    plt.savefig('./plot_publication/{}.png'.format(remove(title)), bbox_inches="tight")
##---------------------------------------------------------------------------##

def main():
    # initialization
    cr = CellReservoir()
    cellreservoir_point = cr.empty_CellReservoir_grid()        
    cellreservoir_spherical = cr.empty_CellReservoir_grid()  
    
    # build cell organelle structures 
    cm_surface, pc_surface, co_surface, cs, cm_idx, pc_idx, co_idx, cs_idx = cr.cell_organelles() 
    vertex_idx = cr.vertex_index()
    
    #plot cell structures    
    cellreservoir_plot = cr.empty_CellReservoir_grid()
    cell_plot = 1*cm_surface + 2*pc_surface + 3*cs+ 4*co_surface 
    cellreservoir_plot[vertex_idx] = cell_plot  
    
    sns.set_style("whitegrid", {'axes.grid' : False})
    plot_2D_cellreservoir(cellreservoir_plot[vertex_idx][int(cellreservoir_plot.shape[0]/4),:,:],
                      title = 'Cell Geometry')

    #plot conductance map
    conductance_map = cr.initiate_conductance()
    plot_cell_graph_edge_conductance_map(conductance_map, cr.no_grid, vertex_idx)
    
    #plot conductance histogram
    plot_dist(conductance_map, "Conductance", "Count", 'LogNormal Conductance')

    def plot_potential_map(cellreservoir, 
                           input_signal, 
                           source_type:str, 
                           charge=[1], 
                           log_type =True):    
        #Collect organelle idx    
        organelle_idx = np.vstack((cm_idx, pc_idx, co_idx, cs_idx))  
        # calculate potential map
        potential_map = cr.potential(organelle_idx, input_signal, charge=charge)
        # set the vertex with potential_map
        cellreservoir[vertex_idx] = potential_map         
        # plot potential and conductance map        
        plot_2D(potential_map[int(potential_map.shape[0]/2), :, :], 
                title = f'Potential Map: {source_type}',
                log=log_type, 
                vmin=np.amin(potential_map.flatten()[potential_map.flatten()>0]), 
                vmax=np.max(potential_map))  
        plot_2D(cellreservoir[int(cellreservoir.shape[0]/2),:,:],
                title = f'CellReservoir Potential Map: {source_type}',
                log=log_type, 
                vmin=np.amin(cellreservoir.flatten()[cellreservoir.flatten()>0]), 
                vmax=np.max(cellreservoir))
        return cellreservoir
    #plot potential map
    
    #point source
    input_signal_point = np.array([[0, 0, cr.r_pc]])
    cellreservoir_point = plot_potential_map(cellreservoir_point, 
                                             input_signal_point, 
                                             "Point Source")
    
    #spherical source 
    input_signal_spherical = cr.index_to_coordinate(pc_idx)   
    cellreservoir_spherical = plot_potential_map(cellreservoir_spherical, 
                                             input_signal_spherical, 
                                             "Spherical Source", 
                                             charge = [1],
                                             log_type = True)
    
    def information_dynamics(cellreservoir, 
                             input_signal, 
                             source_type:str, 
                             total_time_step=40, 
                             fig_col=5, 
                             fig_size=(15,25), 
                             log_type =True):
        # initialize current map
        current_map = cr.empty_Cell()
        state_map = cr.empty_Cell()

        input_signal_idx = cr.coordinate_to_index(input_signal)    
        # vertex indices in cell graph is 2*voxel index of cell        
        signal_tracker_0 = 2*input_signal_idx
        #initiate signal tracker
        signal_time_tracker = [signal_tracker_0]
        #initiate visted set
        visited = set()
        
        #here total_time_step is explicitly defined for plotting purpose
        #alternatively we can condition with while loop which will stop at saturation
        #see RC file
        fig_rows = math.ceil(total_time_step/fig_col)
        fig, axes = plt.subplots(fig_rows, fig_col, figsize=fig_size, dpi=150)
        
        #sub-figure counter
        rows = 0
        col = 0
        
        for signal_time in range(total_time_step):
            print('timestep: ', signal_time)
            # one forward step of information dynamics
            signaled_idx, cellreservoir,\
                current_map, state_map  = cr.forward(cellreservoir, 
                                                    conductance_map, 
                                                    current_map, 
                                                    state_map, 
                                                    signal_time_tracker[signal_time], 
                                                    visited)    
            visited = visited.union(set(tuple(i) for i in signal_time_tracker[signal_time]))
            if signaled_idx.shape[0] > 0:
                #store signaled vertex
                signal_time_tracker.append(signaled_idx)
                #subplots
                im = axes[rows][col].imshow(current_map[int(current_map.shape[0]/2),:,:], 
                                            norm = colors.LogNorm(vmin=1E-5, vmax=50))
                #im.set_cmap('nipy_spectral')
                axes[rows][col].set_title('t = {}'.format(signal_time+1), fontsize=25)   
                axes[rows][col].tick_params(labelsize=30)
                axes[rows][col].set_xticks([]) #tick_params(labelsize=15)
                axes[rows][col].set_yticks([])
                if col == fig_col-1:
                    rows+=1
                    col=0
                else:
                    col+=1
            else:
                print('Signal cant flow anymore')
                break    
        
        #plot information flow per time step
        fig.subplots_adjust(right=0.9, top=0.95)
        cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.75])#[left, bottom, width, height]
        cb = fig.colorbar(im, cax=cbar_ax)
        fig.suptitle(f'Information Flow {source_type}', fontsize = 40)
        cb.ax.tick_params(labelsize=30)
        #fig.tight_layout()
        plt.savefig(f'./plot_publication/Information_Dynamics_{source_type}.svg', 
                    bbox_inches="tight")
        plt.savefig(f'./plot_publication/Information_Dynamics_{source_type}.png', 
                    bbox_inches="tight")
        
        #plot information flow at time =  TOTAL_TIME_STEP
        plot_2D(current_map[int(current_map.shape[0]/2),:,:], 
                title = f'Information Signal Map: {source_type}', 
                log = log_type)
        
        #plot central organelle surface current at time =  TOTAL_TIME_STEP
        cell_with_co_current = cr.get_central_organelle_current(current_map, co_idx)    
        plot_2D(cell_with_co_current[int(cell_with_co_current.shape[0]/2),:,:],
                title = f'Organelle Surface Signal Map: {source_type}',
                log = log_type, 
                vmin = np.amin(cell_with_co_current[int(cell_with_co_current.shape[0]/2),:,:]\
                               [cell_with_co_current[int(cell_with_co_current.shape[0]/2),:,:]>0]), 
                vmax = np.max(cell_with_co_current[int(cell_with_co_current.shape[0]/2),:,:]))
        
        #plot central organelle state at time =  TOTAL_TIME_STEP
        cell_with_co_state = cr.get_central_organelle_current(state_map, co_idx)    
        plot_2D(cell_with_co_state[int(cell_with_co_state.shape[0]/2),:,:],
                title = f'Organelle Surface State Map: {source_type}',
                log = log_type, 
                vmin = np.amin(cell_with_co_state[int(cell_with_co_state.shape[0]/2),:,:]\
                               [cell_with_co_state[int(cell_with_co_state.shape[0]/2),:,:]>0]), 
                vmax = np.max(cell_with_co_state[int(cell_with_co_state.shape[0]/2),:,:]))
   
        # plot histograms
        # co current map        
        plot_dist(cell_with_co_current, 
                  "Signal Strength", 
                  "Count", 
                  f'Organelle Surface Signal Histogram: {source_type}')
        #co state map
        plot_dist(cell_with_co_state, 
                  "Signal Strength", 
                  "Count", 
                  f'Organelle Surface State Histogram: {source_type}')

        
    information_dynamics(cellreservoir_point, 
                         input_signal_point, 
                         "Point Source", 
                         total_time_step = 40, 
                         fig_col = 5, 
                         fig_size = (15,25))   
    information_dynamics(cellreservoir_spherical, 
                         input_signal_spherical, 
                         "Spherical Source", 
                         total_time_step = 12, 
                         fig_col = 4, 
                         fig_size = (12,12),
                         log_type = True)  
    
if __name__ == "__main__": 
    main()