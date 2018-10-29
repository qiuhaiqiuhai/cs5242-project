class VOXEL:
    size = 19
    step = 1.5
    ch_pro_p, ch_lig_p, ch_pro_h, ch_lig_h = 0, 1, 2, 3


class DATA:
    test_total = 8
    unbind_count = 2 # For one ligand atom, there are how many unbind processed data
    processed_amount = 10


class LIMIT:
    min = 1.5
    max = 8


class MODEL:
    name = 'box_size=19,step=1.5,epochs=10,unbind=2.0,model=test4,voxelise=1,repeat=1,' \
           'retrain=0_07-loss=0.0630-acc=0.9780'


class DIR:
    training_data = '../training_data/'
    testing_data = '../testing_data/'
    preprocess_test = '../preprocessed_test/size%d_step%.1f/'
    preprocess_data = '../preprocessed_data/size%d_step%.1f/'
    test_result = '../test_result/'

    unbind_filename = 'unbind_data_%02d'
    bind_filename = 'bind_data'

