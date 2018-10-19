def read_pdb(index, type="lig"):
	filename = gen_file_name(index, type)

	with open(filename, 'r') as file:
		strline_L = file.readlines()
		# print(strline_L)

	X_list = list()
	Y_list = list()
	Z_list = list()
	atomtype_list = list()
	for strline in strline_L:
		# removes all whitespace at the start and end, including spaces, tabs, newlines and carriage returns
		stripped_line = strline.strip()

		line_length = len(stripped_line)
		# print("Line length:{}".format(line_length))
		# if line_length != 78:
		# 	print("index: %d, ERROR: line length is different. Expected=78, current=%d"%(index, line_length))
		# 	print(stripped_line)

		X_list.append(float(stripped_line[30:38].strip()))
		Y_list.append(float(stripped_line[38:46].strip()))
		Z_list.append(float(stripped_line[46:54].strip()))

		atomtype = stripped_line[76:78].strip()
		if atomtype == 'C':
			# atomtype_list.append('h') # 'h' means hydrophobic
			atomtype_list.append(1)
		else:
			# atomtype_list.append('p') # 'p' means polar
			atomtype_list.append(-1)

	return {'x':X_list,
			'y':Y_list,
			'z':Z_list,
			'type':atomtype_list}

folder_path = "../training_data/"

def gen_file_name(index, type="lig"):
	return folder_path+"%04d_%s_cg.pdb"%(index, type)

# X_list, Y_list, Z_list, atomtype_list=read_pdb(129, type="lig")
# print(X_list)
# print(Y_list)
# print(Z_list)
# print(atomtype_list)
#
# X_list, Y_list, Z_list, atomtype_list=read_pdb(129, type="pro")
# print(X_list)
# print(Y_list)
# print(Z_list)
# print(atomtype_list)