from read_pdb_file import read_pdb
import numpy as np

def read_pro_lig_pairs():
    pro_list = []
    lig_list = []
    for index in range(1, 2):
        digit = format(index, '04d')
        file_name_pro = 'training_data/'+digit+'_pro_cg.pdb'
        file_name_lig = 'training_data/'+digit+'_lig_cg.pdb'
        pro = read_pdb(file_name_pro)
        lig = read_pdb(file_name_lig)
        pro_list.append(pro)
        lig_list.append(lig)
    return pro_list, lig_list

# lig is [x_list, y_list, z_list, type_list]
# return centroid of the ligand
def get_lig_centroid(lig):
    x_list = lig[0]
    y_list = lig[1]
    z_list = lig[2]
    return [np.mean(x_list), np.mean(y_list), np.mean(z_list)]

# initilized the grid with centroid being the center of the box
def get_initialized_3D_grid(centroid, box_width, grid_size):
    n_points = box_width/grid_size
    x_axis = [centroid[0]-box_width/2+i*grid_size for i in range(n_points)]
    y_axis = [centroid[1] - box_width / 2 + i * grid_size for i in range(n_points)]
    z_axis = [centroid[2] - box_width / 2 + i * grid_size for i in range(n_points)]
    return [[[0 for i in x_axis] for j in y_axis] for k in z_axis]

def get_mapped_3D_grid(initial_grid, pro, lig):
    for zip(pro[0], pro[1], pro[2])
    pass

def get_3D_grid(box_width, grid_size , pro, lig):
    # get centroid of the ligand
    centroid = get_lig_centroid(lig)
    # centroid of the ligand will become the centroid of the 3D grid box
    initial_grid = get_initialized_3D_grid(centroid, box_width, grid_size)
    # map pro and ligand data points to the grid
    mapped_grid = get_mapped_3D_grid(initial_grid, pro, lig, centroid)


def create_training_data(pro_list, lig_list, box_width, grid_size):
    # for each ligand, get certain percentage of proteins
    for pro, lig in zip(pro_list, lig_list):
        get_3D_grid()


def get_model():
    pass

def train():
   pass

