import numpy as np
from itertools import permutations, combinations_with_replacement
import builtins
import random
from copy import deepcopy
import os
import json

# Method for generating distractors
def gen_distractor(all_prob):
	# Methods and transformations
	distractor_methods = ['other_element',
						  'other_element_transformed',
						  'correct_answer_transformed',
						  'previous_distractor_transformed',
						  'random_new_element']
	transformations = np.array([1,2,-1,-2])
	# Additional methods for multi-rule problems
	if all_prob.shape[-1] > 1:
		distractor_methods.append('correct_answer_permuted')
		distractor_methods.append('other_element_permuted')
		distractor_methods.append('previous_distractor_permuted')
		distractor_methods.append('combination_other_elements')
		distractor_methods.append('combination_previous_distractors')
	# Loop through all problems
	all_probtype_answer_choices = []
	all_probtype_correct_ind = []
	for t in range(all_prob.shape[0]):
		all_answer_choices = []
		all_correct_ind = []
		for p in range(all_prob.shape[1]):
			# Problem
			prob = all_prob[t][p]
			# Extract correct answer
			correct_answer = prob[2,2,:]
			# Other elements (besides correct answer) in problem
			prob_flat = prob.reshape(prob.shape[0]*prob.shape[1],prob.shape[2])
			other_prob_elements = prob_flat[:-1,:]
			other_prob_elements = other_prob_elements[np.logical_not(np.all(other_prob_elements == np.expand_dims(correct_answer,0),1))]
			# Generate each distractor
			answer_choices = []
			for d in range(7):
				valid_distractor = False
				while not valid_distractor:
					# Sample method
					method = distractor_methods[int(np.floor(np.random.rand() * len(distractor_methods)))]
					# Other element from problem
					if method == 'other_element':
						np.random.shuffle(other_prob_elements)
						distractor = deepcopy(other_prob_elements[0])
					# Other element transformed
					elif method == 'other_element_transformed':
						np.random.shuffle(other_prob_elements)
						distractor = deepcopy(other_prob_elements[0])
						transform_dim = int(np.floor(np.random.rand() * all_prob.shape[-1]))
						np.random.shuffle(transformations)
						distractor[transform_dim] += transformations[0]
					# Correct answer transformed
					elif method == 'correct_answer_transformed':
						transform_dim = int(np.floor(np.random.rand() * all_prob.shape[-1]))
						np.random.shuffle(transformations)
						distractor = deepcopy(correct_answer)
						distractor[transform_dim] += transformations[0]
					# Previous distractor transformed
					elif method == 'previous_distractor_transformed':
						random.shuffle(answer_choices)
						if len(answer_choices) > 0:
							distractor = deepcopy(answer_choices[0])
							transform_dim = int(np.floor(np.random.rand() * all_prob.shape[-1]))
							np.random.shuffle(transformations)
							distractor[transform_dim] += transformations[0]
						else:
							distractor = np.ones(all_prob.shape[-1]).astype(int) * -1
					# Random new element
					elif method == 'random_new_element':
						distractor = np.floor(np.random.rand(all_prob.shape[-1]) * 10).astype(int)
					# Correct answer permuted
					elif method == 'correct_answer_permuted':
						valid_perm = builtins.list(permutations(np.arange(all_prob.shape[-1]),all_prob.shape[-1]))[1:]
						random.shuffle(valid_perm)
						distractor = deepcopy(correct_answer)[builtins.list(valid_perm[0])]
					# Other element permuted
					elif method == 'other_element_permuted':
						valid_perm = builtins.list(permutations(np.arange(all_prob.shape[-1]),all_prob.shape[-1]))[1:]
						random.shuffle(valid_perm)
						np.random.shuffle(other_prob_elements)
						other_element = other_prob_elements[0]
						distractor = deepcopy(other_element)[builtins.list(valid_perm[0])]
					# Previous distractor permuted
					elif method == 'previous_distractor_permuted':
						random.shuffle(answer_choices)
						if len(answer_choices) > 0:
							valid_perm = builtins.list(permutations(np.arange(all_prob.shape[-1]),all_prob.shape[-1]))[1:]
							random.shuffle(valid_perm)
							previous_distractor = answer_choices[0]
							distractor = deepcopy(previous_distractor)[builtins.list(valid_perm[0])]
						else:
							distractor = np.ones(all_prob.shape[-1]).astype(int) * -1
					# Combination of other elements
					elif method == 'combination_other_elements':
						distractor = []
						for dim in range(all_prob.shape[-1]):
							other_prob_elements_copy = deepcopy(other_prob_elements)
							np.random.shuffle(other_prob_elements_copy)
							distractor.append(other_prob_elements_copy[0,dim])
						distractor = np.array(distractor)
					# Combination of previous distractors
					elif method == 'combination_previous_distractors':
						random.shuffle(answer_choices)
						if len(answer_choices) > 1:
							distractor = []
							for dim in range(all_prob.shape[-1]):
								answer_choices_copy = deepcopy(answer_choices)
								np.random.shuffle(answer_choices_copy)
								distractor.append(answer_choices_copy[0][dim])
							distractor = np.array(distractor)
						else:
							distractor = np.ones(all_prob.shape[-1]).astype(int) * -1
					# Check to make sure distractor isn't invalid or doesn't already exist
					if  np.all(distractor >= 0) and np.all(distractor <= 9) and not np.all(distractor == correct_answer):
						if len(answer_choices) == 0:
							answer_choices.append(distractor)
							valid_distractor = True
						else:
							if np.logical_not(np.any(np.all(np.array(answer_choices) == np.expand_dims(distractor,0),1))):
								answer_choices.append(distractor)
								valid_distractor = True
			# Add correct answer and shuffle
			answer_choices.append(correct_answer)
			answer_choices = np.array(answer_choices)
			shuffled_order = np.arange(8)
			np.random.shuffle(shuffled_order)
			correct_ind = np.where(shuffled_order == 7)[0][0]
			answer_choices = answer_choices[shuffled_order]
			# Add to list
			all_answer_choices.append(answer_choices)
			all_correct_ind.append(correct_ind)
		# Combine across problem types
		all_probtype_answer_choices.append(np.array(all_answer_choices))
		all_probtype_correct_ind.append(np.array(all_correct_ind))
	# Convert to arrays
	all_probtype_answer_choices = np.array(all_probtype_answer_choices)
	all_probtype_correct_ind = np.array(all_probtype_correct_ind)
	return all_probtype_answer_choices, all_probtype_correct_ind

# Save problems as json and numpy files
def save_prob(all_prob, all_answer_choices, all_correct_ind, prob_type_name, all_problems_np, all_problems_js, perm_invariant=False):
	# Add problems to numpy dict
	all_problems_np[prob_type_name] = {'prob': all_prob, 'answer_choices': all_answer_choices, 'correct_ind': all_correct_ind, 'perm_invariant': perm_invariant}
	# Convert to strings and save as json
	all_data = []
	for p in range(all_prob.shape[0]):
		# Convert problem to string
		prompt = ''
		for r in range(3):
			for c in range(3):
				prompt += '['
				if r == 2 and c == 2:
					for i in range(len(all_prob[p][1][c])):
						prompt += '&nbsp&nbsp'
						if i < (len(all_prob[p][1][c]) - 1):
							prompt += ' '
				else:
					for i in range(len(all_prob[p][r][c])):
						if all_prob[p][r][c][i] == -1:
							prompt += '&nbsp&nbsp'
						else:
							prompt += str(all_prob[p][r][c][i])

						if i < (len(all_prob[p][r][c]) - 1):
							prompt += ' '
				prompt += ']'
				if c < 2:
					prompt += ' &nbsp '
				if r < 2 and c == 2:
					prompt += '<br>'
		# Convert choices to strings
		options = []
		for a in range(8):
			option = '['
			for i in range(len(all_answer_choices[p][a])):
				option += str(all_answer_choices[p][a][i])
				if i < (len(all_answer_choices[p][a]) - 1):
					option += ' '
			if len(all_answer_choices[p][a]) == 0:
				option += '&nbsp&nbsp'
			option += ']'
			options.append(option)
		# Add to dataset
		all_data.append({'prompt': prompt, 'options': options, 'correct': int(all_correct_ind[p]), 'prob_ind': p})
	# Add to javascript data
	all_problems_js[prob_type_name] = all_data
	return all_problems_np, all_problems_js
			
#### 4-rule and 5-rule problems:
# - 1 for each of 15 4-rule combinations
# - 1 for each of 21 5-rule combinations

# Number of problems-per-category will either be N (below) or maximum number possible
N_probs = 100

# All 10choose3 permutations
all_10c3_perm = np.array(builtins.list(permutations(np.arange(10),3)))

# Constant
all_constant = []
all_row_constant = []
all_col_constant = []
for p in range(all_10c3_perm.shape[0]):
	row_prob = np.array([[all_10c3_perm[p][0], all_10c3_perm[p][0], all_10c3_perm[p][0]],
						 [all_10c3_perm[p][1], all_10c3_perm[p][1], all_10c3_perm[p][1]],
						 [all_10c3_perm[p][2], all_10c3_perm[p][2], all_10c3_perm[p][2]]])
	all_row_constant.append(row_prob)
	all_constant.append(row_prob)
	col_prob = np.array([[all_10c3_perm[p][0], all_10c3_perm[p][1], all_10c3_perm[p][2]],
						 [all_10c3_perm[p][0], all_10c3_perm[p][1], all_10c3_perm[p][2]],
						 [all_10c3_perm[p][0], all_10c3_perm[p][1], all_10c3_perm[p][2]]])
	all_constant.append(col_prob)
	all_col_constant.append(col_prob)
all_constant = np.array(all_constant)

# Distribution-of-3
all_dist3 = []
all_dist3_diag1 = []
all_dist3_diag2 = []
for p in range(all_10c3_perm.shape[0]):
	diag1_prob = np.array([[all_10c3_perm[p][0], all_10c3_perm[p][1], all_10c3_perm[p][2]],
						   [all_10c3_perm[p][1], all_10c3_perm[p][2], all_10c3_perm[p][0]],
						   [all_10c3_perm[p][2], all_10c3_perm[p][0], all_10c3_perm[p][1]]])
	all_dist3_diag1.append(diag1_prob)
	all_dist3.append(diag1_prob)
	diag2_prob = np.array([[all_10c3_perm[p][0], all_10c3_perm[p][1], all_10c3_perm[p][2]],
						   [all_10c3_perm[p][2], all_10c3_perm[p][0], all_10c3_perm[p][1]],
						   [all_10c3_perm[p][1], all_10c3_perm[p][2], all_10c3_perm[p][0]]])
	all_dist3_diag2.append(diag2_prob)
	all_dist3.append(diag2_prob)
all_dist3 = np.array(all_dist3)
all_dist3_diag1 = np.array(all_dist3_diag1)
all_dist3_diag2 = np.array(all_dist3_diag2)
# Select subset of distribution-of-3 problems
np.random.shuffle(all_dist3_diag1)

# Progression
prog_size1 = np.array([np.arange(0,5), np.arange(1,6), np.arange(2,7), np.arange(3,8), np.arange(4,9), np.arange(5,10)])
prog_size1_reversed = np.fliplr(prog_size1)
prog_size2 = np.array([np.arange(0,10,2), np.arange(1,11,2)])
prog_size2_reversed = np.fliplr(prog_size2)
all_prog_range = np.concatenate([prog_size1, prog_size1_reversed, prog_size2, prog_size2_reversed], 0)
size1_size2 = np.concatenate([np.zeros(prog_size1.shape[0] + prog_size1_reversed.shape[0]), np.ones(prog_size2.shape[0] + prog_size2_reversed.shape[0])])
all_prog = []
all_prog_size1 = []
all_prog_size2 = []
for p in range(all_prog_range.shape[0]):
	prog_prob = np.array([[all_prog_range[p][0], all_prog_range[p][1], all_prog_range[p][2]],
						  [all_prog_range[p][1], all_prog_range[p][2], all_prog_range[p][3]],
						  [all_prog_range[p][2], all_prog_range[p][3], all_prog_range[p][4]]])
	all_prog.append(prog_prob)
	if size1_size2[p] == 0:
		all_prog_size1.append(prog_prob)
	elif size1_size2[p] == 1:
		all_prog_size2.append(prog_prob)
all_prog = np.array(all_prog)

# All 4-rule and 5-rule sets (combinations with replacement) 
all_4rule_comb = builtins.list(combinations_with_replacement(np.arange(3), 4))
all_5rule_comb = builtins.list(combinations_with_replacement(np.arange(3), 5))
# All 4-rule and 5-rule permutations (with replacement)
# 4 rules
all_4rule_perm = []
for r1 in range(3):
	for r2 in range(3):
		for r3 in range(3):
			for r4 in range(3):
				all_4rule_perm.append([r1, r2, r3, r4])
all_4rule_perm = np.array(all_4rule_perm)
#5 rules
all_5rule_perm = []
for r1 in range(3):
	for r2 in range(3):
		for r3 in range(3):
			for r4 in range(3):
				for r5 in range(3):
					all_5rule_perm.append([r1, r2, r3, r4, r5])
all_5rule_perm = np.array(all_5rule_perm)
# Sort permutations by combination
# 4 rules
all_4rule_perm_sorted = []
for c in range(len(all_4rule_comb)):
	all_4rule_perm_sorted.append(all_4rule_perm[np.all(np.expand_dims(np.array(all_4rule_comb[c]),0) == np.sort(all_4rule_perm,1), 1)])
# 5 rules
all_5rule_perm_sorted = []
for c in range(len(all_5rule_comb)):
	all_5rule_perm_sorted.append(all_5rule_perm[np.all(np.expand_dims(np.array(all_5rule_comb[c]),0) == np.sort(all_5rule_perm,1), 1)])

# Combine problem types
prob_types = [all_constant, all_dist3, all_prog]

# Generate 4-rule problems
all_4rule_prob = []
for c in range(len(all_4rule_comb)):
	all_comb_prob = []
	for p in range(N_probs):
		duplicate_prob = True
		while duplicate_prob:
			# Randomly sample permutation
			all_perm = all_4rule_perm_sorted[c]
			np.random.shuffle(all_perm)
			perm = all_perm[0]
			# Sample rule instances
			# Rule 1
			r1_ind = np.floor(np.random.rand() * prob_types[perm[0]].shape[0]).astype(int)
			r1 = prob_types[perm[0]][r1_ind]
			# Rule 2
			duplicate_rule = True
			while duplicate_rule:
				r2_ind = np.floor(np.random.rand() * prob_types[perm[1]].shape[0]).astype(int)
				r2 = prob_types[perm[1]][r2_ind]
				duplicate_rule =  np.any(np.all(np.expand_dims(np.stack([r1.flatten(), r2.flatten()], 0), 0) == np.expand_dims(np.stack([r1.flatten(), r2.flatten()], 0), 1), 2).flatten()[np.logical_not(np.eye(2).astype(bool).flatten())])
			# Rule 3
			duplicate_rule = True
			while duplicate_rule:
				r3_ind = np.floor(np.random.rand() * prob_types[perm[2]].shape[0]).astype(int)
				r3 = prob_types[perm[2]][r3_ind]
				duplicate_rule =  np.any(np.all(np.expand_dims(np.stack([r1.flatten(), r2.flatten(), r3.flatten()], 0), 0) == np.expand_dims(np.stack([r1.flatten(), r2.flatten(), r3.flatten()], 0), 1), 2).flatten()[np.logical_not(np.eye(3).astype(bool).flatten())])
			# Rule 4
			duplicate_rule = True
			while duplicate_rule:
				r4_ind = np.floor(np.random.rand() * prob_types[perm[3]].shape[0]).astype(int)
				r4 = prob_types[perm[3]][r4_ind]
				duplicate_rule =  np.any(np.all(np.expand_dims(np.stack([r1.flatten(), r2.flatten(), r3.flatten(), r4.flatten()], 0), 0) == np.expand_dims(np.stack([r1.flatten(), r2.flatten(), r3.flatten(), r4.flatten()], 0), 1), 2).flatten()[np.logical_not(np.eye(4).astype(bool).flatten())])
			# Combine rules 1-3
			prob = np.stack([r1,r2,r3,r4],2)
			# Check if duplicate
			duplicate_detected = False
			for i in range(len(all_comb_prob)):
				if np.all(prob == all_comb_prob[i]):
					duplicate_detected = True
			if not duplicate_detected:
				duplicate_prob = False
		all_comb_prob.append(prob)	
	all_comb_prob = np.array(all_comb_prob)
	all_4rule_prob.append(all_comb_prob)
all_4rule_prob = np.array(all_4rule_prob)
# Generate distractors
all_4rule_answer_choices, all_4rule_correct_ind = gen_distractor(all_4rule_prob)

# Generate 5-rule problems
all_5rule_prob = []
for c in range(len(all_5rule_comb)):
	all_comb_prob = []
	for p in range(N_probs):
		duplicate_prob = True
		while duplicate_prob:
			# Randomly sample permutation
			all_perm = all_5rule_perm_sorted[c]
			np.random.shuffle(all_perm)
			perm = all_perm[0]
			# Sample rule instances
			# Rule 1
			r1_ind = np.floor(np.random.rand() * prob_types[perm[0]].shape[0]).astype(int)
			r1 = prob_types[perm[0]][r1_ind]
			# Rule 2
			duplicate_rule = True
			while duplicate_rule:
				r2_ind = np.floor(np.random.rand() * prob_types[perm[1]].shape[0]).astype(int)
				r2 = prob_types[perm[1]][r2_ind]
				duplicate_rule =  np.any(np.all(np.expand_dims(np.stack([r1.flatten(), r2.flatten()], 0), 0) == np.expand_dims(np.stack([r1.flatten(), r2.flatten()], 0), 1), 2).flatten()[np.logical_not(np.eye(2).astype(bool).flatten())])
			# Rule 3
			duplicate_rule = True
			while duplicate_rule:
				r3_ind = np.floor(np.random.rand() * prob_types[perm[2]].shape[0]).astype(int)
				r3 = prob_types[perm[2]][r3_ind]
				duplicate_rule =  np.any(np.all(np.expand_dims(np.stack([r1.flatten(), r2.flatten(), r3.flatten()], 0), 0) == np.expand_dims(np.stack([r1.flatten(), r2.flatten(), r3.flatten()], 0), 1), 2).flatten()[np.logical_not(np.eye(3).astype(bool).flatten())])
			# Rule 4
			duplicate_rule = True
			while duplicate_rule:
				r4_ind = np.floor(np.random.rand() * prob_types[perm[3]].shape[0]).astype(int)
				r4 = prob_types[perm[3]][r4_ind]
				duplicate_rule =  np.any(np.all(np.expand_dims(np.stack([r1.flatten(), r2.flatten(), r3.flatten(), r4.flatten()], 0), 0) == np.expand_dims(np.stack([r1.flatten(), r2.flatten(), r3.flatten(), r4.flatten()], 0), 1), 2).flatten()[np.logical_not(np.eye(4).astype(bool).flatten())])
			# Rule 5
			duplicate_rule = True
			while duplicate_rule:
				r5_ind = np.floor(np.random.rand() * prob_types[perm[4]].shape[0]).astype(int)
				r5 = prob_types[perm[4]][r5_ind]
				duplicate_rule =  np.any(np.all(np.expand_dims(np.stack([r1.flatten(), r2.flatten(), r3.flatten(), r4.flatten(), r5.flatten()], 0), 0) == np.expand_dims(np.stack([r1.flatten(), r2.flatten(), r3.flatten(), r4.flatten(), r5.flatten()], 0), 1), 2).flatten()[np.logical_not(np.eye(5).astype(bool).flatten())])
			# Combine rules 1-3
			prob = np.stack([r1,r2,r3,r4,r5],2)
			# Check if duplicate
			duplicate_detected = False
			for i in range(len(all_comb_prob)):
				if np.all(prob == all_comb_prob[i]):
					duplicate_detected = True
			if not duplicate_detected:
				duplicate_prob = False
		all_comb_prob.append(prob)	
	all_comb_prob = np.array(all_comb_prob)
	all_5rule_prob.append(all_comb_prob)
all_5rule_prob = np.array(all_5rule_prob)
# Generate distractors
all_5rule_answer_choices, all_5rule_correct_ind = gen_distractor(all_5rule_prob)

# Convert problems to strings and save as js script, also as numpy file
all_problems_np = {}
all_problems_js = {}
# 4-rule problems
for c in range(all_4rule_prob.shape[0]):
	all_problems_np, all_problems_js = save_prob(all_4rule_prob[c], all_4rule_answer_choices[c], all_4rule_correct_ind[c], 'four_rule_comb' + str(c), all_problems_np, all_problems_js)
# 5-rule problems
for c in range(all_5rule_prob.shape[0]):
	all_problems_np, all_problems_js = save_prob(all_5rule_prob[c], all_5rule_answer_choices[c], all_5rule_correct_ind[c], 'five_rule_comb' + str(c), all_problems_np, all_problems_js)
# Save numpy file
np_fname = './all_4_5_rule_problems.npz'
np.savez(np_fname, all_problems=all_problems_np)
# Convert to json string
json_string = json.dumps(all_problems_js)
# Write to js script
js_fname = './all_4_5_rule_problems.js'
js_fid = open(js_fname, 'w')
js_fid.write('var all_problems = ' + json_string)
js_fid.close()
