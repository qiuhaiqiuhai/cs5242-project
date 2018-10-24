class VOXEL:
    size = 19
    step = 1
    ch_pro_p, ch_lig_p, ch_pro_h, ch_lig_h = 0, 1, 2, 3

class DATA:
    unbind_count = 2 # For one ligand atom, there are how many unbind processed data
    processed_amount = 2400



class DIR:
    preprocess_base = '../preprocessed_data/'
    bind_data = '../preprocessed_data/voxelise_%d/bind_data'
    unbind_data = '../preprocessed_data/voxelise_%d/unbind_data_%02d'
    voxelise_base = '../preprocessed_data/voxelise_%d/'
    # dir 'training' to save training data
    training_base = '../preprocessed_data/training/voxelise_%d/'
    training_bind_data = '../preprocessed_data/training/voxelise_%d/bind_data'
    training_unbind_data = '../preprocessed_data/training/voxelise_%d/unbind_data_%02d'

