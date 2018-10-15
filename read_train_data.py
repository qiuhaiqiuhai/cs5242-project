from read_pdb_file import read_pdb

def read_train_data():
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

