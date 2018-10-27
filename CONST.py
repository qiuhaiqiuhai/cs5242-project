class VOXEL:
    size = 19
    step = 1
    ch_pro_p, ch_lig_p, ch_pro_h, ch_lig_h = 0, 1, 2, 3

class DATA:
    test_total = 824
    unbind_count = 4 # For one ligand atom, there are how many unbind processed data
    processed_amount = 10

class LIMIT:
    min = 1.5
    max = 8

class DIR:
    preprocess_test = '../preprocessed_test/'
    preprocess_base = '../preprocessed_data/'
    bind_data = '../preprocessed_data/voxelise_%d/bind_data'
    unbind_data = '../preprocessed_data/voxelise_%d/unbind_data_%02d'
    voxelise_base = '../preprocessed_data/voxelise_%d/'
    # dir 'training' to save training data
    training_base = '../preprocessed_data/training/voxelise_%d/'
    training_bind_data = '../preprocessed_data/training/voxelise_%d/bind_data'
    training_unbind_data = '../preprocessed_data/training/voxelise_%d/unbind_data_%02d'
    unbind_filename = 'unbind_data_%02d'
    bind_filename = 'bind_data'

