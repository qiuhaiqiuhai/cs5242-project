import random
import numpy as np
import math

test_ratio = 0.2
all_count = 3000
all_indexes = np.array(range(1, all_count+1))
random.shuffle(all_indexes)
split_point = math.floor(test_ratio*all_count)
testing_indexes = all_indexes[:split_point]
training_indexes = all_indexes[split_point:]
np.savetxt('testing_indexes.txt', testing_indexes, fmt='%d', )
np.savetxt('training_indexes.txt', training_indexes, fmt='%d')

