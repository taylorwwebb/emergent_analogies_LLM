import numpy as np
from copy import deepcopy
import random
import json

## Problems generated using one of the following 6 transformations:
#
# Successorship
# [a b c d] [a b c e]
# [i j k l] [i j k m]
#
# Predecessorship
# [b c d e] [a c d e]
# [j k l m] [i k l m]
#
# Add letter to sequence
# [a b c d] [a b c d e]
# [i j k l] [i j k l m]
#
# Remove redundant character
# [a b b c d e] [a b c d e]
# [i j k l l m] [i j k l m]
#
# Fix alphabetic sequence
# [a w c d e] [a b c d e]
# [i j k p m] [i j k l m]
#
# Sort characters
# [a b e d c] [a b c d e]
# [i k j l m] [i j k l m]
#
#
## and between 0 and 3 generalizations, out of the following 6:
#
# Larger interval
# [a b c d] [a b c e]
# [i k m o] [i k m q]
#
# Longer target
# [a b c d] [a b c e]
# [i j k l m n o p] [i j k l m n o q]
#
# Grouping
# [a b c d] [a b c e]
# [i i j j k k l l] [i i j j k k m m]
#
# Interleaved X's
# [a b c d] [a b c e]
# [i x j x k x l x] [i x j x k x m x] 
#
# Letter-to-number
# [a b c d] [a b c e]
# [1 2 3 4] [1 2 3 5]
#
# Reversal
# [a b c d] [a b c e]
# [m l k j] [m l k i]
#
##

# Alphabet
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
N_letters = len(letters)
# Numbers
numbers = np.arange(N_letters) + 1
# Linearly ordered real-world concepts
realworld_linear = [['cold', 'cool', 'warm', 'hot'],
					['love', 'like', 'dislike', 'hate'],
					['jack', 'queen', 'king', 'ace'],
					['penny', 'nickel', 'dime', 'quarter'],
					['second', 'minute', 'hour', 'day']]

# Successor transformation
def apply_succ(prob_letters):
	return [prob_letters[:-1], prob_letters[:-2] + [prob_letters[-1]]]

# Predecessor transformation
def apply_pred(prob_letters):
	return [prob_letters[1:], [prob_letters[0]] + prob_letters[2:]]

# Add letter to sequence
def apply_add_letter(prob_letters):
	return [prob_letters[:-1], prob_letters]

# Remove redundant letter
def apply_remove_redundant(prob_letters):
	redundant_loc = np.arange(len(prob_letters))
	np.random.shuffle(redundant_loc)
	redundant_loc = redundant_loc[0]
	prob_redundant = deepcopy(prob_letters)
	prob_redundant.insert(redundant_loc, prob_letters[redundant_loc])
	return [prob_redundant, prob_letters]

# Remove out-of-place character
def apply_fix_alphabet(prob_letters):
	remaining_letters = np.array(deepcopy(letters))
	remaining_letters = remaining_letters[np.all(np.expand_dims(np.array(remaining_letters),1) != np.expand_dims(np.array(prob_letters),0), 1)]
	np.random.shuffle(remaining_letters)
	insert_letter = remaining_letters[0]
	insert_loc = np.arange(len(prob_letters))
	np.random.shuffle(insert_loc)
	insert_loc = insert_loc[0]
	prob_letters_insert = deepcopy(prob_letters)
	prob_letters_insert[insert_loc] = insert_letter
	return [prob_letters_insert, prob_letters]

# Sort letters
def apply_sort(prob_letters):
	swap_loc = np.arange(len(prob_letters))
	np.random.shuffle(swap_loc)
	i_loc = swap_loc[0]
	j_loc = swap_loc[1]
	i_letter = prob_letters[i_loc]
	j_letter = prob_letters[j_loc]
	prob_swapped = deepcopy(prob_letters)
	prob_swapped[i_loc] = j_letter
	prob_swapped[j_loc] = i_letter
	return [prob_swapped, prob_letters]

# Method for generating subset of problems
def gen_prob_subset(N_generalize=0, N_prob=100, standard_len=5, longer_targ_len=9, larger_int_size=2,
					trans_allowed=['succ', 'pred', 'add_letter', 'remove_redundant', 'fix_alphabet', 'sort'],
					gen_allowed=['larger_int', 'longer_targ', 'group', 'interleaved', 'letter2num', 'reverse', 'realworld']):
	# Initialize storage for problems
	all_prob = []
	all_trans = []
	all_gen = []
	all_src_letters = []
	all_tgt_letters = []
	while len(all_prob) < N_prob:
		# Sample source letters
		src_start = np.floor(np.random.rand() * (len(letters)-(standard_len-1))).astype(int)
		src_letters = letters[src_start:src_start+standard_len]
		# Sample generalizations
		random.shuffle(gen_allowed)
		generalize = gen_allowed[:N_generalize]
		# Sample target letters
		if 'realworld' in generalize:
			random.shuffle(realworld_linear)
			tgt_letters = realworld_linear[0]
		else:
			if 'longer_targ' in generalize and 'larger_int' in generalize:
				tgt_span = (longer_targ_len * larger_int_size) - 1
				src_duplicate = True
				while src_duplicate:
					tgt_start = np.floor(np.random.rand() * (len(letters)-(tgt_span-1))).astype(int)
					if src_start != tgt_start:
						src_duplicate = False
				tgt_letters = letters[tgt_start:tgt_start+tgt_span][::2]
			elif 'longer_targ' in generalize and 'larger_int' not in generalize:
				src_duplicate = True
				while src_duplicate:
					tgt_start = np.floor(np.random.rand() * (len(letters)-(longer_targ_len-1))).astype(int)
					if src_start != tgt_start:
						src_duplicate = False
				tgt_letters = letters[tgt_start:tgt_start+longer_targ_len]
			elif 'longer_targ' not in generalize and 'larger_int' in generalize:
				tgt_span = (standard_len * larger_int_size) - 1
				src_duplicate = True
				while src_duplicate:
					tgt_start = np.floor(np.random.rand() * (len(letters)-(tgt_span-1))).astype(int)
					if src_start != tgt_start:
						src_duplicate = False
				tgt_letters = letters[tgt_start:tgt_start+tgt_span][::2]
			elif 'longer_targ' not in generalize and 'larger_int' not in generalize:
				src_duplicate = True
				while src_duplicate:
					tgt_start = np.floor(np.random.rand() * (len(letters)-(standard_len-1))).astype(int)
					if src_start != tgt_start:
						src_duplicate = False
				tgt_letters = letters[tgt_start:tgt_start+standard_len]
		# Reverse target letters
		if 'reverse' in generalize:
			tgt_letters.reverse()
		# Sample transformation
		random.shuffle(trans_allowed)
		trans = trans_allowed[0]
		# Apply transformation
		if trans == 'succ':
			src = apply_succ(src_letters)
			tgt = apply_succ(tgt_letters)
		elif trans == 'pred':
			src = apply_pred(src_letters)
			tgt = apply_pred(tgt_letters)
		elif trans == 'add_letter':
			src = apply_add_letter(src_letters)
			tgt = apply_add_letter(tgt_letters)
		elif trans == 'remove_redundant':
			src = apply_remove_redundant(src_letters)
			tgt = apply_remove_redundant(tgt_letters)
		elif trans == 'fix_alphabet':
			src = apply_fix_alphabet(src_letters)
			tgt = apply_fix_alphabet(tgt_letters)
		elif trans == 'sort':
			src = apply_sort(src_letters)
			tgt = apply_sort(tgt_letters)
		# Generalization from letters to numbers
		if 'letter2num' in generalize:
			new_tgt = []
			for i in range(len(tgt)):
				new_tgt_i = []
				for j in range(len(tgt[i])):
					new_tgt_i.append(numbers[np.where(np.array(letters) == tgt[i][j])[0][0]])
				new_tgt.append(new_tgt_i)
			tgt = new_tgt
		# Interleaved X's (or 0's, if target composed of numbers)
		if 'interleaved' in generalize:
			new_tgt = []
			for i in range(len(tgt)):
				new_tgt_i = []
				for j in range(len(tgt[i])):
					new_tgt_i.append(tgt[i][j])
					if 'letter2num' in generalize:
						new_tgt_i.append('0')
					else:
						new_tgt_i.append('x')
				new_tgt.append(new_tgt_i)
			tgt = new_tgt
		# Grouping
		if 'group' in generalize:
			new_tgt = []
			for i in range(len(tgt)):
				new_tgt_i = []
				for j in range(len(tgt[i])):
					new_tgt_i.append(tgt[i][j])
					new_tgt_i.append(tgt[i][j])
				new_tgt.append(new_tgt_i)
			tgt = new_tgt
		# Check that problem doesn't already exist
		prob = [src, tgt]
		duplicate = False
		for p_prev in range(len(all_prob)):
			if np.array(prob).shape == np.array(all_prob[p_prev]).shape:
				if np.all(np.array(prob) == np.array(all_prob[p_prev])):
					duplicate = True
		# Add to problem subset
		if not duplicate:
			all_prob.append(prob)
			all_trans.append(trans)
			all_gen.append(generalize)
			all_src_letters.append(src_letters)
			all_tgt_letters.append(tgt_letters)
	return {'prob': all_prob, 'trans': all_trans, 'gen': all_gen, 'src_letters': all_src_letters, 'tgt_letters': all_tgt_letters}

# Add problems to json and numpy file
def save_prob(all_prob, prob_type_name, all_prob_js):
	# Convert to strings and save as json
	all_data = []
	for p in range(len(all_prob['prob'])):
		# A
		prompt = '['
		for i in range(len(all_prob['prob'][p][0][0])):
			prompt += str(all_prob['prob'][p][0][0][i])
			if i < len(all_prob['prob'][p][0][0]) - 1:
				prompt += ' '
		prompt += '] &nbsp ['
		# B
		for i in range(len(all_prob['prob'][p][0][1])):
			prompt += str(all_prob['prob'][p][0][1][i])
			if i < len(all_prob['prob'][p][0][1]) - 1:
				prompt += ' '
		prompt += ']<br>['
		# C
		for i in range(len(all_prob['prob'][p][1][0])):
			prompt += str(all_prob['prob'][p][1][0][i])
			if i < len(all_prob['prob'][p][1][0]) - 1:
				prompt += ' '
		prompt += '] &nbsp [&nbsp ? &nbsp]'
		# Add to dataset
		all_data.append({'prompt': prompt, 'prob_ind': p})
	# Add to javascript data
	all_prob_js[prob_type_name] = all_data
	return all_prob_js

# Split subset
def split_subset(all_prob, N_split):
	all_prob_split = []
	N_subset = int(len(all_prob['prob']) / N_split)
	for s in range(N_split):
		subset = {}
		for key in all_prob.keys():
			subset[key] = []
		for p in range(N_subset*s,N_subset*(s+1)):
			for key in all_prob.keys():
				subset[key].append(all_prob[key][p])
		all_prob_split.append(subset)
	return all_prob_split

# Generate all basic analogies (zero generalizations)
all_succ = gen_prob_subset(trans_allowed=['succ'])
all_pred = gen_prob_subset(trans_allowed=['pred'])
all_add_letter = gen_prob_subset(trans_allowed=['add_letter'])
all_remove_redundant = gen_prob_subset(trans_allowed=['remove_redundant'])
all_fix_alphabet = gen_prob_subset(trans_allowed=['fix_alphabet'])
all_sort = gen_prob_subset(trans_allowed=['sort'])

# Generate all problems with one generalization (randomly sample transformations)
all_larger_int = gen_prob_subset(N_generalize=1, gen_allowed=['larger_int'])
all_longer_targ = gen_prob_subset(N_generalize=1, gen_allowed=['longer_targ'])
all_group = gen_prob_subset(N_generalize=1, gen_allowed=['group'])
all_interleaved = gen_prob_subset(N_generalize=1, gen_allowed=['interleaved'])
all_letter2num = gen_prob_subset(N_generalize=1, gen_allowed=['letter2num'])
all_reverse = gen_prob_subset(N_generalize=1, gen_allowed=['reverse'])

# Generate all problems with 2 and 3 generalizations
all_2gen = gen_prob_subset(N_generalize=2, N_prob=600, gen_allowed=['larger_int', 'longer_targ', 'group', 'interleaved', 'letter2num', 'reverse'])
all_2gen_split = split_subset(all_2gen, 6)
all_3gen = gen_prob_subset(N_generalize=3, N_prob=600, gen_allowed=['larger_int', 'longer_targ', 'group', 'interleaved', 'letter2num', 'reverse'])
all_3gen_split = split_subset(all_3gen, 6)

# Generate problems involving generalization to real-world concepts
all_realworld_succ = gen_prob_subset(standard_len=4, N_generalize=1, trans_allowed=['succ'], gen_allowed=['realworld'])
all_realworld_pred = gen_prob_subset(standard_len=4, N_generalize=1, trans_allowed=['pred'], gen_allowed=['realworld'])
all_realworld_add_letter = gen_prob_subset(standard_len=4, N_generalize=1, trans_allowed=['add_letter'], gen_allowed=['realworld'])
all_realworld_sort = gen_prob_subset(standard_len=4, N_generalize=1, trans_allowed=['sort'], gen_allowed=['realworld'])

# Combine problems
all_prob_types = [all_succ, all_pred, all_add_letter, all_remove_redundant, all_fix_alphabet, all_sort,
				  all_larger_int, all_longer_targ, all_group, all_interleaved, all_letter2num, all_reverse,
				  all_2gen_split[0], all_2gen_split[1], all_2gen_split[2], all_2gen_split[3], all_2gen_split[4], all_2gen_split[5],
				  all_3gen_split[0], all_3gen_split[1], all_3gen_split[2], all_3gen_split[3], all_3gen_split[4], all_3gen_split[5],
				  all_realworld_succ, all_realworld_pred, all_realworld_add_letter, all_realworld_sort]
all_prob_type_names = ['succ', 'pred', 'add_letter', 'remove_redundant', 'fix_alphabet', 'sort',
					   'larger_int', 'longer_targ', 'group', 'interleaved', 'letter2num', 'reverse',
					   '2gen_split1', '2gen_split2', '2gen_split3', '2gen_split4', '2gen_split5', '2gen_split6',
					   '3gen_split1', '3gen_split2', '3gen_split3', '3gen_split4', '3gen_split5', '3gen_split6',
					   'realworld_succ', 'realworld_pred', 'realworld_add_letter', 'realworld_sort']

# Create js variable for all_problems
all_prob_js = {}
all_prob_np = {}
for p in range(len(all_prob_types)):
	all_prob_js = save_prob(all_prob_types[p], all_prob_type_names[p], all_prob_js)
	all_prob_np[all_prob_type_names[p]] = all_prob_types[p]
# Write numpy file
np.savez('./all_prob.npz', all_prob=all_prob_np)
# Convert to json strings
all_prob_json_string = json.dumps(all_prob_js)
# Write to js script
js_fname = './all_prob.js'
js_fid = open(js_fname, 'w')
js_fid.write('var all_problems = ' + all_prob_json_string)
js_fid.close()







