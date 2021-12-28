import numpy as np

input_x = [1, 0.2, 0.1, 0.05, 0.2]
input_w = [0.2, 0.12, 0.4, 0.6, 0.9]

input_vector = np.array(input_x)
weights = np.array(input_w)
bias_weight = 0.2

activation_level = np.dot(input_vector, weights)+(bias_weight*1)
print(activation_level)

threshold = 0.5

if activation_level >= threshold:
	perceptron_output = 1
else:
	perceptron_output = 0

print(perceptron_output)

## second part

expected_output = 0
example_weights = weights
new_weights = []

for i, x in enumerate(input_x):
	new_weights.append(weights[i]+ (expected_output -perceptron_output)*x)
weights = np.array(new_weights)

print(example_weights)
print(weights)

