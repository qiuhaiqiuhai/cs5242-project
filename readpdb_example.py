import CONST


def read_pdb_test(index, type='lig'):

    filename = CONST.DIR.testing_data + gen_file_name(index, type)
    with open(filename, 'r') as file:
        strline_L = file.readlines()

    X_list = list()
    Y_list = list()
    Z_list = list()
    atomtype_list = list()
    for strline in strline_L:
        stripped_line = strline.strip()
        splitted_line = stripped_line.split('\t')

        X_list.append(float(splitted_line[0]))
        Y_list.append(float(splitted_line[1]))
        Z_list.append(float(splitted_line[2]))
        atomtype_list.append(str(splitted_line[3]))

    return {'x':X_list,
			'y':Y_list,
			'z':Z_list,
			'type':atomtype_list}


def read_pdb(index, type="lig"):
	"""
	read training pdb file
	:param index: index of protein or ligand
	:param type: 'lig' or 'pro'
	:return: json object with x, y, z and type list
	"""
	filename = CONST.DIR.training_data + gen_file_name(index, type)

	with open(filename, 'r') as file:
		strline_L = file.readlines()

	X_list = list()
	Y_list = list()
	Z_list = list()
	atomtype_list = list()
	for strline in strline_L:
		# removes all whitespace at the start and end, including spaces, tabs, newlines and carriage returns
		stripped_line = strline.strip()

		X_list.append(float(stripped_line[30:38].strip()))
		Y_list.append(float(stripped_line[38:46].strip()))
		Z_list.append(float(stripped_line[46:54].strip()))

		atomtype = stripped_line[76:78].strip()
		if atomtype == 'C':
			atomtype_list.append('h') # 'h' means hydrophobic
		else:
			atomtype_list.append('p') # 'p' means polar

	return {'x':X_list,
			'y':Y_list,
			'z':Z_list,
			'type':atomtype_list}


def gen_file_name(index, type="lig"):
	return "%04d_%s_cg.pdb"%(index, type)






