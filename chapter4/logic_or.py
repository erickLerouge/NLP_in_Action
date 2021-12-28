from random import random
import numpy as np

sample_data = [[0,0],[0,1],[1,0],[1,1]]
expected_results = [0,1,1,1]

activation_threshold = 0.5

weights = np.random.random(2)/1000
print(weights)

bias_weight = np.random.random()/1000
print(bias_weight)

for idx, sample in enumerate(sample_data):
	input_vector = np.array(sample) # should be a typo in the book? sample VS sample_data
	activation_level = np.dot(input_vector,weights)+(bias_weight * 1)
	
	if activation_level > activation_threshold:
		perceptron_output = 1
	else:
		perceptron_output = 0
	print('Predicted {} '.format(perceptron_output))
	print('Expected {} '.format(expected_results[idx]))
	print()

### time to make it learn:

for iteration_num in range(5):
	correct_answers = 0
	for idx, sample in enumerate(sample_data):
		input_vector = np.array(sample)
		weights = np.array(weights)
		activation_level = np.dot(input_vector,weights)+(bias_weight * 1)
		
		if activation_level > activation_threshold:
			perceptron_output = 1
		else:
			perceptron_output = 0
		if perceptron_output == expected_results[idx]:
			correct_answers +=1
		new_weights = []
		for i, x, in enumerate(sample):
			#print('1 {}'.format(weights[i]))
			#print('2 {}'.format(expected_results[idx]))
			#print('3 {}'.format(perceptron_output))
			#print('4 {}'.format(x))
			new_weights.append(weights[i] + (expected_results[idx] - perceptron_output) * x)
		bias_weight = bias_weight + ((expected_results[idx] - perceptron_output)*1)
		weights = np.array(new_weights)
	print('{} correct_answers out of 4 for iteration {}'.format(correct_answers,iteration_num))

