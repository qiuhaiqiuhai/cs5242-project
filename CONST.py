class VOXEL:
    size = 19
    step = 1
    ch_pro_p, ch_lig_p, ch_pro_h, ch_lig_h = 0, 1, 2, 3

class DATA:
    unbind_count = 5 # For one ligand atom, there are how many unbind processed data
    processed_amount = 3000



class DIR:
    preprocess_base = '../preprocessed_data/'
    bind_data = '../preprocessed_data/bind_data'
    unbind_data = '../preprocessed_data/unbind_data_%02d'
