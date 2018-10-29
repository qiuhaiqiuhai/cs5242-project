import numpy as np
import csv
import sys

with open('test_ground_truth_example.txt', 'a') as f:
	writer = csv.writer(f, delimiter='\t')
	for i in range(4, 825):
		writer.writerow([i, i])


ground_truth_filename = 'test_ground_truth_example.txt'
predictions_filename = 'test_predictions.txt'

ground_truth_arr = np.loadtxt(ground_truth_filename, dtype=np.int, delimiter='\t', skiprows=1)
predictions_arr = np.loadtxt(predictions_filename, dtype=np.int, delimiter='\t', skiprows=1)

num_samples = ground_truth_arr.shape[0]
num_predictions = predictions_arr.shape[0]

if num_predictions > num_samples:
	print('WARNING: number of predictions are higher than number of samples!!!')
	num_predictions = num_samples

# prepare ground truth dictionary
ground_truth_dict = dict()
for i in range(num_samples):
	dict_key = ground_truth_arr[i,0]
	dict_value = ground_truth_arr[i,1]
	ground_truth_dict[dict_key] = dict_value


# count correct predictions
num_correct_pred = 0
for i in range(num_predictions):
	pro_id = predictions_arr[i,0]
	lig_list = list(predictions_arr[i,1:])

	truth_lig_id = ground_truth_dict[pro_id]
	if truth_lig_id in lig_list:
		num_correct_pred += 1

acc = num_correct_pred / num_samples

print('accuracy:{:.3f}'.format(acc))


