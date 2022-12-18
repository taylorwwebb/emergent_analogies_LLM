import numpy as np
from itertools import permutations, combinations, combinations_with_replacement
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

# Method for generating distractors in logic problems
def gen_distractor_logicprob(all_prob, prob_elements, permute_answers=False):
	# Loop through all problems
	all_answer_choices = []
	all_correct_ind = []
	for p in range(all_prob.shape[0]):
		# Problem
		prob = all_prob[p]
		# Extract correct answer
		correct_answer = prob[2,2]
		# Remove -1 from correct answer
		correct_answer = builtins.list(np.array(correct_answer)[np.array(correct_answer) != -1])
		# Get all subsets of unique elements
		all_subsets = []
		for i in range(prob_elements.shape[1]):
			subsets = builtins.list(combinations(prob_elements[p], i+1))
			for s in range(len(subsets)):
				all_subsets.append(builtins.list(subsets[s]))
		# Add empty set
		all_subsets.append([])
		all_subsets = np.array(all_subsets, dtype=object)
		# Eliminate correct answer from list of subsets
		all_include = []
		for s in range(all_subsets.shape[0]):
			subset = all_subsets[s]
			random.shuffle(subset)
			if len(subset) == len(correct_answer):
				if np.all(np.sort(subset) == np.sort(correct_answer)):
					include = False
				else:
					include = True
			else:
				include = True
			all_include.append(include)
		all_subsets = all_subsets[all_include]
		# Select subset and shuffle
		np.random.shuffle(all_subsets)
		answer_choices = []
		for a in range(7):
			answer_choices.append(all_subsets[a])
		answer_choices.append(correct_answer)
		answer_choices = np.array(answer_choices, dtype=object)
		shuffled_order = np.arange(8)
		np.random.shuffle(shuffled_order)
		correct_ind = np.where(shuffled_order == 7)[0][0]
		answer_choices = answer_choices[shuffled_order]
		# Permute elements within each answer
		if permute_answers:
			shuffled_answer_choices = []
			for a in range(8):
				choice = deepcopy(answer_choices[a])
				random.shuffle(choice)
				shuffled_answer_choices.append(choice)
			answer_choices = np.array(shuffled_answer_choices, dtype=object)
		# Sort elements based on order of appearance in problem
		else:
			sorted_answer_choices = []
			for a in range(8):
				choice = answer_choices[a]
				sorted_choice = []
				for i in range(len(prob_elements[p])):
					if np.any(np.array(choice) == prob_elements[p][i]):
						sorted_choice.append(prob_elements[p][i])
				sorted_answer_choices.append(sorted_choice)
			answer_choices = np.array(sorted_answer_choices, dtype=object)
		# Add to list
		all_answer_choices.append(answer_choices)
		all_correct_ind.append(correct_ind)
	# Convert to arrays
	all_answer_choices = np.array(all_answer_choices)
	all_correct_ind = np.array(all_correct_ind)
	return all_answer_choices, all_correct_ind

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
			
#### 1-rule problems (1 per condition below, 6 total):
# - row constant
# - column constant
# - distribution-of-3 (bottom-left to top-right diagonal)
# - distribution-of-3 (top-left to bottom-right diagonal)
# - size-1 progression
# - size-2 progression
#### 2-rule and 3-rule problems:
# - 1 for each of 6 2-rule combinations
# - 1 for each of 10 3-rule combinations
#### Logic problems (2 per condition below, 1 shuffled and 1 with reliable spatial locations, 10 total):
# - set union, c3 = c1 U c2
# - set union, c2 = c1 U c3
# - set union, c1 = c2 U c3
# - AND
# - XOR

# Number of problems-per-category will either be N (below) or maximum number possible
N_probs = 100

# All 10choose3 permutations
all_10c3_perm = np.array(builtins.list(permutations(np.arange(10),3)))

# Constant
all_constant = []
all_row_constant = []
all_col_constant = []
row_col = []
for p in range(all_10c3_perm.shape[0]):
	row_prob = np.array([[all_10c3_perm[p][0], all_10c3_perm[p][0], all_10c3_perm[p][0]],
						 [all_10c3_perm[p][1], all_10c3_perm[p][1], all_10c3_perm[p][1]],
						 [all_10c3_perm[p][2], all_10c3_perm[p][2], all_10c3_perm[p][2]]])
	all_row_constant.append(row_prob)
	all_constant.append(row_prob)
	row_col.append(0)
	col_prob = np.array([[all_10c3_perm[p][0], all_10c3_perm[p][1], all_10c3_perm[p][2]],
						 [all_10c3_perm[p][0], all_10c3_perm[p][1], all_10c3_perm[p][2]],
						 [all_10c3_perm[p][0], all_10c3_perm[p][1], all_10c3_perm[p][2]]])
	all_constant.append(col_prob)
	all_col_constant.append(col_prob)
	row_col.append(1)
all_constant = np.array(all_constant)
all_row_constant = np.array(all_row_constant)
all_col_constant = np.array(all_col_constant)
row_col = np.array(row_col)
# Select subset of row-constant and column-constant problems
np.random.shuffle(all_row_constant)
row_constant_prob = np.expand_dims(np.expand_dims(all_row_constant[:N_probs],0),4)
np.random.shuffle(all_col_constant)
col_constant_prob = np.expand_dims(np.expand_dims(all_col_constant[:N_probs],0),4)
# Generate distractors
row_constant_answer_choices, row_constant_correct_ind = gen_distractor(row_constant_prob)
col_constant_answer_choices, col_constant_correct_ind = gen_distractor(col_constant_prob)

# Distribution-of-3
all_dist3 = []
all_dist3_diag1 = []
all_dist3_diag2 = []
diag1_diag2 = []
for p in range(all_10c3_perm.shape[0]):
	diag1_prob = np.array([[all_10c3_perm[p][0], all_10c3_perm[p][1], all_10c3_perm[p][2]],
						   [all_10c3_perm[p][1], all_10c3_perm[p][2], all_10c3_perm[p][0]],
						   [all_10c3_perm[p][2], all_10c3_perm[p][0], all_10c3_perm[p][1]]])
	all_dist3_diag1.append(diag1_prob)
	all_dist3.append(diag1_prob)
	diag1_diag2.append(0)
	diag2_prob = np.array([[all_10c3_perm[p][0], all_10c3_perm[p][1], all_10c3_perm[p][2]],
						   [all_10c3_perm[p][2], all_10c3_perm[p][0], all_10c3_perm[p][1]],
						   [all_10c3_perm[p][1], all_10c3_perm[p][2], all_10c3_perm[p][0]]])
	all_dist3_diag2.append(diag2_prob)
	all_dist3.append(diag2_prob)
	diag1_diag2.append(1)
all_dist3 = np.array(all_dist3)
all_dist3_diag1 = np.array(all_dist3_diag1)
all_dist3_diag2 = np.array(all_dist3_diag2)
diag1_diag2 = np.array(diag1_diag2)
# Select subset of distribution-of-3 problems
np.random.shuffle(all_dist3_diag1)
dist3_diag1_prob = np.expand_dims(np.expand_dims(all_dist3_diag1[:N_probs],0),4)
np.random.shuffle(all_dist3_diag2)
dist3_diag2_prob = np.expand_dims(np.expand_dims(all_dist3_diag2[:N_probs],0),4)
# Generate distractors
dist3_diag1_answer_choices, dist3_diag1_correct_ind = gen_distractor(dist3_diag1_prob)
dist3_diag2_answer_choices, dist3_diag2_correct_ind = gen_distractor(dist3_diag2_prob)

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
prog_size1_prob = np.expand_dims(np.expand_dims(np.array(all_prog_size1),0),4)
prog_size2_prob = np.expand_dims(np.expand_dims(np.array(all_prog_size2),0),4)
# Generate distractors
prog_size1_answer_choices, prog_size1_correct_ind = gen_distractor(prog_size1_prob)
prog_size2_answer_choices, prog_size2_correct_ind = gen_distractor(prog_size2_prob)

# All 2-rule and 3-rule sets (combinations with replacement) 
all_2rule_comb = builtins.list(combinations_with_replacement(np.arange(3), 2))
all_3rule_comb = builtins.list(combinations_with_replacement(np.arange(3), 3))
# All 2-rule and 3-rule permutations (with replacement)
# 2 rules
all_2rule_perm = []
for r1 in range(3):
	for r2 in range(3):
		all_2rule_perm.append([r1, r2])
all_2rule_perm = np.array(all_2rule_perm)
# 3 rules
all_3rule_perm = []
for r1 in range(3):
	for r2 in range(3):
		for r3 in range(3):
			all_3rule_perm.append([r1, r2, r3])
all_3rule_perm = np.array(all_3rule_perm)
# Sort permutations by combination
# 2 rules
all_2rule_perm_sorted = []
for c in range(len(all_2rule_comb)):
	all_2rule_perm_sorted.append(all_2rule_perm[np.all(np.expand_dims(np.array(all_2rule_comb[c]),0) == np.sort(all_2rule_perm,1), 1)])
# 3 rules
all_3rule_perm_sorted = []
for c in range(len(all_3rule_comb)):
	all_3rule_perm_sorted.append(all_3rule_perm[np.all(np.expand_dims(np.array(all_3rule_comb[c]),0) == np.sort(all_3rule_perm,1), 1)])

# Combine problem types
prob_types = [all_constant, all_dist3, all_prog]

# Generate 2-rule problems
all_2rule_prob = []
for c in range(len(all_2rule_comb)):
	all_comb_prob = []
	for p in range(N_probs):
		duplicate_prob = True
		while duplicate_prob:
			# Randomly sample permutation
			all_perm = all_2rule_perm_sorted[c]
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
				if not np.all(r1 == r2):
					duplicate_rule = False
			# Combine rules 1 and 2
			prob = np.stack([r1,r2],2)
			# Check if duplicate
			duplicate_detected = False
			for i in range(len(all_comb_prob)):
				if np.all(prob == all_comb_prob[i]):
					duplicate_detected = True
			if not duplicate_detected:
				duplicate_prob = False
		all_comb_prob.append(prob)	
	all_comb_prob = np.array(all_comb_prob)
	all_2rule_prob.append(all_comb_prob)
all_2rule_prob = np.array(all_2rule_prob)
# Generate distractors
all_2rule_answer_choices, all_2rule_correct_ind = gen_distractor(all_2rule_prob)

# Number of unique rules per problem
N_unique_rules_2rule_prob = []
for p in range(len(all_2rule_comb)):
	N_unique_rules_2rule_prob.append(np.unique(all_2rule_comb[p]).shape[0])
np.savez('./N_unique_rules_2rule_prob.npz', N_unique_rules=N_unique_rules_2rule_prob)
N_unique_rules_3rule_prob = []
for p in range(len(all_3rule_comb)):
	N_unique_rules_3rule_prob.append(np.unique(all_3rule_comb[p]).shape[0])
np.savez('./N_unique_rules_3rule_prob.npz', N_unique_rules=N_unique_rules_3rule_prob)

# Generate 3-rule problems
all_3rule_prob = []
for c in range(len(all_3rule_comb)):
	all_comb_prob = []
	for p in range(N_probs):
		duplicate_prob = True
		while duplicate_prob:
			# Randomly sample permutation
			all_perm = all_3rule_perm_sorted[c]
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
				if not np.all(r1 == r2):
					duplicate_rule = False
			# Rule 3
			duplicate_rule = True
			while duplicate_rule:
				r3_ind = np.floor(np.random.rand() * prob_types[perm[2]].shape[0]).astype(int)
				r3 = prob_types[perm[2]][r3_ind]
				if not np.all(r1 == r3) and not np.all(r2 == r3):
					duplicate_rule = False
			# Combine rules 1-3
			prob = np.stack([r1,r2,r3],2)
			# Check if duplicate
			duplicate_detected = False
			for i in range(len(all_comb_prob)):
				if np.all(prob == all_comb_prob[i]):
					duplicate_detected = True
			if not duplicate_detected:
				duplicate_prob = False
		all_comb_prob.append(prob)	
	all_comb_prob = np.array(all_comb_prob)
	all_3rule_prob.append(all_comb_prob)
all_3rule_prob = np.array(all_3rule_prob)
# Generate distractors
all_3rule_answer_choices, all_3rule_correct_ind = gen_distractor(all_3rule_prob)

### Logic problems

# Set union problems
# 3 different rules:
# 	c3 = c1 U c2
# 	c2 = c1 U c3
# 	c1 = c2 U c3
# These rules follow a specific format in which the rule is applied both along rows and columns
# Example of c3 = c1 U c2
# [1  ] [4  ] [1 4]
# [  7] [  2] [7 2]
# [1 7] [4 2] [1 4 7 2]

# All 10choose4 combinations
all_10c4_comb = np.array(builtins.list(combinations(np.arange(10),4)))

## Generate all 'c3 = c1 U c2' problems
np.random.shuffle(all_10c4_comb)
all_c3_set_union_prob = []
for p in range(N_probs):
	# Get set of elements for this problem
	prob_comb = deepcopy(all_10c4_comb[p])
	np.random.shuffle(prob_comb)
	all_10c4_comb[p] = prob_comb
	# Generate problem
	prob = np.array([[[prob_comb[0], -1], 			[prob_comb[2], -1], 			[prob_comb[0], prob_comb[2]]],
					 [[-1, prob_comb[1]], 			[-1, prob_comb[3]], 			[prob_comb[1], prob_comb[3]]],
					 [[prob_comb[0], prob_comb[1]], [prob_comb[2], prob_comb[3]], 	[prob_comb[0], prob_comb[1], prob_comb[2], prob_comb[3]]]], dtype=object)
	all_c3_set_union_prob.append(prob)
all_c3_set_union_prob = np.array(all_c3_set_union_prob)
# Generate distractors
all_c3_set_union_answer_choices, all_c3_set_union_correct_ind = gen_distractor_logicprob(all_c3_set_union_prob, all_10c4_comb)

# Same as above, but with elements permuted within each cell
all_c3_set_union_permuted_prob = []
for p in range(N_probs):
	# Get set of elements for this problem
	prob_comb = deepcopy(all_10c4_comb[p + N_probs])
	np.random.shuffle(prob_comb)
	all_10c4_comb[p + N_probs] = prob_comb
	# Generate problem
	prob = np.array([[[prob_comb[0], -1], 			[prob_comb[1], -1], 			[prob_comb[0], prob_comb[1]]],
					 [[-1, prob_comb[2]], 			[-1, prob_comb[3]], 			[prob_comb[2], prob_comb[3]]],
					 [[prob_comb[0], prob_comb[2]], [prob_comb[1], prob_comb[3]], 	[prob_comb[0], prob_comb[1], prob_comb[2], prob_comb[3]]]], dtype=object)
	# Shuffle position of elements within each cell
	for r in range(prob.shape[0]):
		for c in range(prob.shape[1]):
			random.shuffle(prob[r][c])
	all_c3_set_union_permuted_prob.append(prob)
all_c3_set_union_permuted_prob = np.array(all_c3_set_union_permuted_prob)
# Generate distractors
all_c3_set_union_permuted_answer_choices, all_c3_set_union_permuted_correct_ind = gen_distractor_logicprob(all_c3_set_union_permuted_prob, all_10c4_comb[N_probs:], permute_answers=True)

## Generate all 'c2 = c1 U c3' problems
np.random.shuffle(all_10c4_comb)
all_c2_set_union_prob = []
for p in range(N_probs):
	# Get set of elements for this problem
	prob_comb = deepcopy(all_10c4_comb[p])
	np.random.shuffle(prob_comb)
	all_10c4_comb[p] = prob_comb
	# Generate problem
	prob = np.array([[[-1, prob_comb[1]], 			[-1, prob_comb[1], prob_comb[2], -1], 						[prob_comb[2], -1]],
					 [[prob_comb[0], prob_comb[1]], [prob_comb[0], prob_comb[1], prob_comb[2], prob_comb[3]], 	[prob_comb[2], prob_comb[3]]],
					 [[prob_comb[0], -1], 			[prob_comb[0], -1, -1, prob_comb[3]], 						[-1, prob_comb[3]]]], dtype=object)
	all_c2_set_union_prob.append(prob)
all_c2_set_union_prob = np.array(all_c2_set_union_prob)
# Generate distractors
all_c2_set_union_answer_choices, all_c2_set_union_correct_ind = gen_distractor_logicprob(all_c2_set_union_prob, all_10c4_comb)

# Same as above, but with elements permuted within each cell
all_c2_set_union_permuted_prob = []
for p in range(N_probs):
	# Get set of elements for this problem
	prob_comb = deepcopy(all_10c4_comb[p + N_probs])
	np.random.shuffle(prob_comb)
	all_10c4_comb[p + N_probs] = prob_comb
	# Generate problem
	prob = np.array([[[-1, prob_comb[1]], 			[-1, prob_comb[1], prob_comb[2], -1], 						[prob_comb[2], -1]],
					 [[prob_comb[0], prob_comb[1]], [prob_comb[0], prob_comb[1], prob_comb[2], prob_comb[3]], 	[prob_comb[2], prob_comb[3]]],
					 [[prob_comb[0], -1], 			[prob_comb[0], -1, -1, prob_comb[3]], 						[-1, prob_comb[3]]]], dtype=object)
	# Shuffle position of elements within each cell
	for r in range(prob.shape[0]):
		for c in range(prob.shape[1]):
			random.shuffle(prob[r][c])
	all_c2_set_union_permuted_prob.append(prob)
all_c2_set_union_permuted_prob = np.array(all_c2_set_union_permuted_prob)
# Generate distractors
all_c2_set_union_permuted_answer_choices, all_c2_set_union_permuted_correct_ind = gen_distractor_logicprob(all_c2_set_union_permuted_prob, all_10c4_comb[N_probs:], permute_answers=True)

## Generate all 'c1 = c2 U c3' problems
np.random.shuffle(all_10c4_comb)
all_c1_set_union_prob = []
for p in range(N_probs):
	# Get set of elements for this problem
	prob_comb = deepcopy(all_10c4_comb[p])
	np.random.shuffle(prob_comb)
	all_10c4_comb[p] = prob_comb
	# Generate problem
	prob = np.array([[[prob_comb[0], prob_comb[1], prob_comb[2], prob_comb[3]], [prob_comb[0], prob_comb[1]], 	[prob_comb[2], prob_comb[3]]],
					 [[-1, prob_comb[1], -1, prob_comb[3]], 					[-1, prob_comb[1]], 			[-1, prob_comb[3]]],
					 [[prob_comb[0], -1, prob_comb[2], -1], 					[prob_comb[0], -1], 			[prob_comb[2], -1]]], dtype=object)
	all_c1_set_union_prob.append(prob)
all_c1_set_union_prob = np.array(all_c1_set_union_prob)
# Generate distractors
all_c1_set_union_answer_choices, all_c1_set_union_correct_ind = gen_distractor_logicprob(all_c1_set_union_prob, all_10c4_comb)

# Same as above, but with elements permuted within each cell
all_c1_set_union_permuted_prob = []
for p in range(N_probs):
	# Get set of elements for this problem
	prob_comb = deepcopy(all_10c4_comb[p + N_probs])
	np.random.shuffle(prob_comb)
	all_10c4_comb[p + N_probs] = prob_comb
	# Generate problem
	prob = np.array([[[prob_comb[0], prob_comb[1], prob_comb[2], prob_comb[3]], [prob_comb[0], prob_comb[1]], 	[prob_comb[2], prob_comb[3]]],
					 [[-1, prob_comb[1], -1, prob_comb[3]], 					[-1, prob_comb[1]], 			[-1, prob_comb[3]]],
					 [[prob_comb[0], -1, prob_comb[2], -1], 					[prob_comb[0], -1], 			[prob_comb[2], -1]]], dtype=object)
	# Shuffle position of elements within each cell
	for r in range(prob.shape[0]):
		for c in range(prob.shape[1]):
			random.shuffle(prob[r][c])
	all_c1_set_union_permuted_prob.append(prob)
all_c1_set_union_permuted_prob = np.array(all_c1_set_union_permuted_prob)
# Generate distractors
all_c1_set_union_permuted_answer_choices, all_c1_set_union_permuted_correct_ind = gen_distractor_logicprob(all_c1_set_union_permuted_prob, all_10c4_comb[N_probs:], permute_answers=True)

# AND problems
# The rules also have a very specific format, here is an example:
# [1 3 9] [1 3 5] [1 3]
# [2 3 9] [2 3 5] [2 3]
# [  3 9] [  3 5] [  3]

# All 10choose4 combinations
all_10c5_comb = np.array(builtins.list(combinations(np.arange(10),5)))
np.random.shuffle(all_10c5_comb)

# Generate AND problems
all_AND_prob = []
for p in range(N_probs):
	# Get set of elements for this problem
	prob_comb = deepcopy(all_10c5_comb[p])
	np.random.shuffle(prob_comb)
	all_10c5_comb[p] = prob_comb
	# Generate problem
	prob = np.array([[[prob_comb[0], prob_comb[2], prob_comb[4]], 	[prob_comb[1], prob_comb[2], prob_comb[4]],		[prob_comb[2], prob_comb[4]]],
					 [[prob_comb[0], prob_comb[2], prob_comb[3]], 	[prob_comb[1], prob_comb[2], prob_comb[3]], 	[prob_comb[2], prob_comb[3]]],
					 [[prob_comb[0], prob_comb[2], -1], 			[prob_comb[1], prob_comb[2], -1], 				[prob_comb[2]]]], dtype=object)
	all_AND_prob.append(prob)
all_AND_prob = np.array(all_AND_prob)
# Generate distractors
all_AND_answer_choices, all_AND_correct_ind = gen_distractor_logicprob(all_AND_prob, all_10c5_comb)

# Same as above, but with elements permuted within each cell, and also correct answer is blank cell
all_AND_permuted_prob = []
for p in range(N_probs):
	# Get set of elements for this problem
	prob_comb = deepcopy(all_10c5_comb[p + N_probs])
	np.random.shuffle(prob_comb)
	all_10c5_comb[p + N_probs] = prob_comb
	# Generate problem
	prob = np.array([[[prob_comb[0], prob_comb[4]], [prob_comb[1], prob_comb[4]],	[prob_comb[4]]],
					 [[prob_comb[0], prob_comb[3]], [prob_comb[1], prob_comb[3]], 	[prob_comb[3]]],
					 [[prob_comb[0], -1], 			[prob_comb[1], -1], 			[]]], dtype=object)
	# Shuffle position of elements within each cell
	for r in range(prob.shape[0]):
		for c in range(prob.shape[1]):
			random.shuffle(prob[r][c])
	all_AND_permuted_prob.append(prob)
all_AND_permuted_prob = np.array(all_AND_permuted_prob)
# Generate distractors
all_AND_permuted_answer_choices, all_AND_permuted_correct_ind = gen_distractor_logicprob(all_AND_permuted_prob, all_10c5_comb[N_probs:], permute_answers=True)

# XOR problems
# For each row, there will always be:
#	- 1 element in each of the four corners
# 	- 1 element that forms a square in the upper left
#	- 1 element that forms a square in the lower right
#	- 1 element that appears twice in each row
# For example:
# [0 1] [0 2] [1 2]
# [0 2] [0 3] [2 3]
# [1 2] [2 3] [1 3]

# Generate XOR problems
np.random.shuffle(all_10c4_comb)
all_XOR_prob = []
for p in range(N_probs):
	# Get set of elements for this problem
	prob_comb = deepcopy(all_10c4_comb[p])
	np.random.shuffle(prob_comb)
	all_10c4_comb[p] = prob_comb
	# Generate problem
	prob = np.array([[[prob_comb[0], prob_comb[1]], 	[prob_comb[0], prob_comb[2]],	[prob_comb[1], prob_comb[2]]],
					 [[prob_comb[0], prob_comb[2]], 	[prob_comb[3], prob_comb[0]], 	[prob_comb[2], prob_comb[3]]],
					 [[prob_comb[1], prob_comb[2]], 	[prob_comb[2], prob_comb[3]], 	[prob_comb[1], prob_comb[3]]]])
	all_XOR_prob.append(prob)
all_XOR_prob = np.array(all_XOR_prob)
# Generate distractors
all_XOR_answer_choices, all_XOR_correct_ind = gen_distractor_logicprob(all_XOR_prob, all_10c4_comb)

# Same as above, but with elements permuted within each cell
all_XOR_permuted_prob = []
for p in range(N_probs):
	# Get set of elements for this problem
	prob_comb = deepcopy(all_10c4_comb[p + N_probs])
	np.random.shuffle(prob_comb)
	all_10c4_comb[p + N_probs] = prob_comb
	# Generate problem
	prob = np.array([[[prob_comb[0], prob_comb[1]], 	[prob_comb[0], prob_comb[2]],	[prob_comb[1], prob_comb[2]]],
					 [[prob_comb[0], prob_comb[2]], 	[prob_comb[3], prob_comb[0]], 	[prob_comb[2], prob_comb[3]]],
					 [[prob_comb[1], prob_comb[2]], 	[prob_comb[2], prob_comb[3]], 	[prob_comb[1], prob_comb[3]]]])
	# Shuffle position of elements within each cell
	for r in range(prob.shape[0]):
		for c in range(prob.shape[1]):
			random.shuffle(prob[r][c])
	all_XOR_permuted_prob.append(prob)
all_XOR_permuted_prob = np.array(all_XOR_permuted_prob)
# Generate distractors
all_XOR_permuted_answer_choices, all_XOR_permuted_correct_ind = gen_distractor_logicprob(all_XOR_permuted_prob, all_10c4_comb[N_probs:], permute_answers=True)

# Convert problems to strings and save as js script, also as numpy file
all_problems_np = {}
all_problems_js = {}
# Constant problems
all_problems_np, all_problems_js = save_prob(row_constant_prob[0], row_constant_answer_choices[0], row_constant_correct_ind[0], 'row_constant', all_problems_np, all_problems_js)
all_problems_np, all_problems_js = save_prob(col_constant_prob[0], col_constant_answer_choices[0], col_constant_correct_ind[0], 'col_constant', all_problems_np, all_problems_js)
# Distribution-of-3 problems
all_problems_np, all_problems_js = save_prob(dist3_diag1_prob[0], dist3_diag1_answer_choices[0], dist3_diag1_correct_ind[0], 'dist3_diag1', all_problems_np, all_problems_js)
all_problems_np, all_problems_js = save_prob(dist3_diag2_prob[0], dist3_diag2_answer_choices[0], dist3_diag2_correct_ind[0], 'dist3_diag2', all_problems_np, all_problems_js)
# Progression problems
all_problems_np, all_problems_js = save_prob(prog_size1_prob[0], prog_size1_answer_choices[0], prog_size1_correct_ind[0], 'prog_size1', all_problems_np, all_problems_js)
all_problems_np, all_problems_js = save_prob(prog_size2_prob[0], prog_size2_answer_choices[0], prog_size2_correct_ind[0], 'prog_size2', all_problems_np, all_problems_js)
# 2-rule problems
for c in range(all_2rule_prob.shape[0]):
	all_problems_np, all_problems_js = save_prob(all_2rule_prob[c], all_2rule_answer_choices[c], all_2rule_correct_ind[c], 'two_rule_comb' + str(c), all_problems_np, all_problems_js)
# 3-rule problems
for c in range(all_3rule_prob.shape[0]):
	all_problems_np, all_problems_js = save_prob(all_3rule_prob[c], all_3rule_answer_choices[c], all_3rule_correct_ind[c], 'three_rule_comb' + str(c), all_problems_np, all_problems_js)
# Logic problems
all_problems_np, all_problems_js = save_prob(all_c3_set_union_prob, all_c3_set_union_answer_choices, all_c3_set_union_correct_ind, 'c3_set_union', all_problems_np, all_problems_js, perm_invariant=True)
all_problems_np, all_problems_js = save_prob(all_c1_set_union_prob, all_c1_set_union_answer_choices, all_c1_set_union_correct_ind, 'c1_set_union', all_problems_np, all_problems_js, perm_invariant=True)
all_problems_np, all_problems_js = save_prob(all_c2_set_union_prob, all_c2_set_union_answer_choices, all_c2_set_union_correct_ind, 'c2_set_union', all_problems_np, all_problems_js, perm_invariant=True)
all_problems_np, all_problems_js = save_prob(all_AND_prob, all_AND_answer_choices, all_AND_correct_ind, 'AND', all_problems_np, all_problems_js, perm_invariant=True)
all_problems_np, all_problems_js = save_prob(all_XOR_prob, all_XOR_answer_choices, all_XOR_correct_ind, 'XOR', all_problems_np, all_problems_js, perm_invariant=True)
# Permuted logic problems
all_problems_np, all_problems_js = save_prob(all_c3_set_union_permuted_prob, all_c3_set_union_permuted_answer_choices, all_c3_set_union_permuted_correct_ind, 'c3_set_union_permuted', all_problems_np, all_problems_js, perm_invariant=True)
all_problems_np, all_problems_js = save_prob(all_c2_set_union_permuted_prob, all_c2_set_union_permuted_answer_choices, all_c2_set_union_permuted_correct_ind, 'c2_set_union_permuted', all_problems_np, all_problems_js, perm_invariant=True)
all_problems_np, all_problems_js = save_prob(all_c1_set_union_permuted_prob, all_c1_set_union_permuted_answer_choices, all_c1_set_union_permuted_correct_ind, 'c1_set_union_permuted', all_problems_np, all_problems_js, perm_invariant=True)
all_problems_np, all_problems_js = save_prob(all_XOR_permuted_prob, all_XOR_permuted_answer_choices, all_XOR_permuted_correct_ind, 'XOR_permuted', all_problems_np, all_problems_js, perm_invariant=True)
all_problems_np, all_problems_js = save_prob(all_AND_permuted_prob, all_AND_permuted_answer_choices, all_AND_permuted_correct_ind, 'AND_permuted', all_problems_np, all_problems_js, perm_invariant=True)
# Save numpy file
np_fname = './all_problems.npz'
np.savez(np_fname, all_problems=all_problems_np)
# Convert to json string
json_string = json.dumps(all_problems_js)
# Write to js script
js_fname = './all_problems.js'
js_fid = open(js_fname, 'w')
js_fid.write('var all_problems = ' + json_string)
js_fid.close()
