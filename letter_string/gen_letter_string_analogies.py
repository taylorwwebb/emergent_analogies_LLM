import numpy as np
from itertools import permutations
import builtins

# Alphabet
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
N_letters = len(letters)

# Problem 1: basic problem
# [a b c] [a b d]
# [i j k] [i j l]
prob1_name = 'basic'
all_prob1 = []
all_prob1_completion = []
src_letters = letters[:4]
for tgt in range(N_letters-4):
	tgt_letters = letters[tgt:tgt+4]
	# Ensure no overlap between source and target sets
	if (np.expand_dims(np.array(src_letters),1) == np.expand_dims(np.array(tgt_letters),0)).sum() == 0:
		# Construct problem
		prob = '[' + src_letters[0] + ' ' + src_letters[1] + ' ' + src_letters[2] + '] '
		prob += '[' + src_letters[0] + ' ' + src_letters[1] + ' ' + src_letters[3] + ']\n'
		prob += '[' + tgt_letters[0] + ' ' + tgt_letters[1] + ' ' + tgt_letters[2] + '] ['
		all_prob1.append(prob)
		completion = tgt_letters[0] + ' ' + tgt_letters[1] + ' ' + tgt_letters[3] + ']'
		all_prob1_completion.append([completion])
# Shuffle
all_prob1 = np.array(all_prob1)
all_prob1_completion = np.array(all_prob1_completion)
random_ind = np.arange(all_prob1.shape[0])
np.random.shuffle(random_ind)
all_prob1 = all_prob1[random_ind]
all_prob1_completion = all_prob1_completion[random_ind]

# Problem 2: cleaning up a string 
# [a b b c d e] [a b c d e]
# [p q r r s t] [p q r s t]
prob2_name = 'clean_string'
all_prob2 = []
all_prob2_completion = []
src_letters = letters[:5]
for tgt in range(N_letters-5):
	tgt_letters = letters[tgt:tgt+5]
	# Ensure no overlap between source and target sets
	if (np.expand_dims(np.array(src_letters),1) == np.expand_dims(np.array(tgt_letters),0)).sum() == 0:
		# Randomly repeat one letter in source and one letter in target (not the same letter)
		repeat_ind = np.arange(1,4)
		np.random.shuffle(repeat_ind)
		src_repeat = repeat_ind[0]
		tgt_repeat = repeat_ind[1]
		# Construct problem
		prob = '['
		for i in range(5):
			if i == src_repeat:
				prob += src_letters[i] + ' ' + src_letters[i]
			else:
				prob += src_letters[i]
			if i < 4:
				prob += ' '
			else:
				prob += '] ['
		for i in range(5):
			prob += src_letters[i]
			if i < 4:
				prob += ' '
			else:
				prob += ']\n['
		for i in range(5):
			if i == tgt_repeat:
				prob += tgt_letters[i] + ' ' + tgt_letters[i]
			else:
				prob += tgt_letters[i]
			if i < 4:
				prob += ' '
			else:
				prob += '] ['
		all_prob2.append(prob)
		completion = ''
		for i in range(5):
			completion += tgt_letters[i]
			if i < 4:
				completion += ' '
			else:
				completion += ']'
		all_prob2_completion.append([completion])
# Shuffle
all_prob2 = np.array(all_prob2)
all_prob2_completion = np.array(all_prob2_completion)
random_ind = np.arange(all_prob2.shape[0])
np.random.shuffle(random_ind)
all_prob2 = all_prob2[random_ind]
all_prob2_completion = all_prob2_completion[random_ind]

# Problem 3: abstract succesorship, version 1
# [a b c] [a b d]
# [1 2 3] [1 2 4]
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
N_numbers = len(numbers)
prob3_name = 'letter_to_num'
all_prob3 = []
all_prob3_completion = []
src_letters = letters[:4]
for tgt in range(N_numbers-4):
	tgt_numbers = numbers[tgt:tgt+4]
	# Construct problem
	prob = '[' + src_letters[0] + ' ' + src_letters[1] + ' ' + src_letters[2] + '] '
	prob += '[' + src_letters[0] + ' ' + src_letters[1] + ' ' + src_letters[3] + ']\n'
	prob += '[' + tgt_numbers[0] + ' ' + tgt_numbers[1] + ' ' + tgt_numbers[2] + '] ['
	all_prob3.append(prob)
	completion = tgt_numbers[0] + ' ' + tgt_numbers[1] + ' ' + tgt_numbers[3] + ']'
	all_prob3_completion.append([completion])
# Shuffle
all_prob3 = np.array(all_prob3)
all_prob3_completion = np.array(all_prob3_completion)
random_ind = np.arange(all_prob3.shape[0])
np.random.shuffle(random_ind)
all_prob3 = all_prob3[random_ind]
all_prob3_completion = all_prob3_completion[random_ind]

# Problem 4: generalizing to groupings 
# [a b c] [a b d]
# [i i j j k k] [i i j j l l]
prob4_name = 'groupings'
all_prob4 = []
all_prob4_completion = []
src_letters = letters[:4]
for tgt in range(N_letters-4):
	tgt_letters = letters[tgt:tgt+4]
	# Ensure no overlap between source and target sets
	if (np.expand_dims(np.array(src_letters),1) == np.expand_dims(np.array(tgt_letters),0)).sum() == 0:
		# Construct problem
		prob = '[' + src_letters[0] + ' ' + src_letters[1] + ' ' + src_letters[2] + '] '
		prob += '[' + src_letters[0] + ' ' + src_letters[1] + ' ' + src_letters[3] + ']\n'
		prob += '[' + tgt_letters[0] + ' ' + tgt_letters[0] + ' ' + tgt_letters[1] + ' ' + tgt_letters[1] + ' ' + tgt_letters[2] + ' ' + tgt_letters[2] + '] ['
		all_prob4.append(prob)
		completion = tgt_letters[0] + ' ' + tgt_letters[0] + ' ' + tgt_letters[1] + ' ' + tgt_letters[1] + ' ' + tgt_letters[3] + ' ' + tgt_letters[3] + ']'
		all_prob4_completion.append([completion])
# Shuffle
all_prob4 = np.array(all_prob4)
all_prob4_completion = np.array(all_prob4_completion)
random_ind = np.arange(all_prob4.shape[0])
np.random.shuffle(random_ind)
all_prob4 = all_prob4[random_ind]
all_prob4_completion = all_prob4_completion[random_ind]

# Problem 5: generalizing to strings of longer length
# [a b c] [a b d]
# [i j k l m] [i j l k l n]
prob5_name = 'longer_targ'
all_prob5 = []
all_prob5_completion = []
src_letters = letters[:4]
for tgt in range(N_letters-6):
	tgt_letters = letters[tgt:tgt+6]
	# Ensure no overlap between source and target sets
	if (np.expand_dims(np.array(src_letters),1) == np.expand_dims(np.array(tgt_letters),0)).sum() == 0:
		# Construct problem
		prob = '[' + src_letters[0] + ' ' + src_letters[1] + ' ' + src_letters[2] + '] '
		prob += '[' + src_letters[0] + ' ' + src_letters[1] + ' ' + src_letters[3] + ']\n'
		prob += '[' + tgt_letters[0] + ' ' + tgt_letters[1] + ' ' + tgt_letters[2] + ' ' + tgt_letters[3] + ' ' + tgt_letters[4] + '] ['
		all_prob5.append(prob)
		completion = tgt_letters[0] + ' ' + tgt_letters[1] + ' ' + tgt_letters[2] + ' ' + tgt_letters[3] + ' ' + tgt_letters[5] + ']'
		all_prob5_completion.append([completion])
# Shuffle
all_prob5 = np.array(all_prob5)
all_prob5_completion = np.array(all_prob5_completion)
random_ind = np.arange(all_prob5.shape[0])
np.random.shuffle(random_ind)
all_prob5 = all_prob5[random_ind]
all_prob5_completion = all_prob5_completion[random_ind]

# Problem 6: successor to predecessor
# [a b c] [a b d]
# [k j i] [k j h]
prob6_name = 'succ_to_pred'
all_prob6 = []
all_prob6_completion = []
src_letters = letters[:4]
for tgt in range(N_letters-5):
	tgt_letters = letters[tgt:tgt+5]
	# Ensure no overlap between source and target sets
	if (np.expand_dims(np.array(src_letters),1) == np.expand_dims(np.array(tgt_letters),0)).sum() == 0:
		# Construct problem
		prob = '[' + src_letters[0] + ' ' + src_letters[1] + ' ' + src_letters[2] + '] '
		prob += '[' + src_letters[0] + ' ' + src_letters[1] + ' ' + src_letters[3] + ']\n'
		prob += '[' + tgt_letters[3] + ' ' + tgt_letters[2] + ' ' + tgt_letters[1] + '] ['
		all_prob6.append(prob)
		completion1 = tgt_letters[3] + ' ' + tgt_letters[2] + ' ' + tgt_letters[0] + ']'
		completion2 = tgt_letters[3] + ' ' + tgt_letters[2] + ' ' + tgt_letters[2] + ']'
		all_prob6_completion.append([completion1, completion2])
# Shuffle
all_prob6 = np.array(all_prob6)
all_prob6_completion = np.array(all_prob6_completion)
random_ind = np.arange(all_prob6.shape[0])
np.random.shuffle(random_ind)
all_prob6 = all_prob6[random_ind]
all_prob6_completion = all_prob6_completion[random_ind]

# Problem 7: context problem for relational priming
# [a b c] [a b d]
# [m r r j j j] [m r r k k k]
# Generate all permutations of 3 letters (excluding any involving adjacent letters)
all_perm = np.array(builtins.list(permutations(range(N_letters),3)))
all_perm_include = []
for p in range(all_perm.shape[0]):
	perm = all_perm[p]
	if np.all(np.array([np.abs(perm[0] - perm[1]), np.abs(perm[0] - perm[2]), np.abs(perm[1] - perm[2])]) > 3):
		all_perm_include.append(perm)
all_perm_include = np.array(all_perm_include)
# Generate problems
prob7_name = 'letter_to_count'
all_prob7 = []
all_prob7_completion = []
src_letters = letters[:4]
for tgt in range(all_perm_include.shape[0]):
	tgt_letters = [letters[all_perm_include[tgt,0]], letters[all_perm_include[tgt,1]], letters[all_perm_include[tgt,2]]]
	# Ensure no overlap between source and target sets
	non_overlapping = (np.expand_dims(np.array(src_letters),1) == np.expand_dims(np.array(tgt_letters),0)).sum() == 0
	if non_overlapping and tgt_letters[-1] != 'z':
		# Construct problem
		prob = '[' + src_letters[0] + ' ' + src_letters[1] + ' ' + src_letters[2] + '] '
		prob += '[' + src_letters[0] + ' ' + src_letters[1] + ' ' + src_letters[3] + ']\n' 
		prob += '[' + tgt_letters[0] + ' ' + tgt_letters[1] + ' ' + tgt_letters[1] + ' ' + tgt_letters[2] + ' ' + tgt_letters[2] + ' ' + tgt_letters[2] + '] ['
		all_prob7.append(prob)
		tgt_successor = letters[np.where(np.array(letters) == tgt_letters[-1])[0][0] + 1]
		completion1 = tgt_letters[0] + ' ' + tgt_letters[1] + ' ' + tgt_letters[1] + ' ' + tgt_letters[2] + ' ' + tgt_letters[2] + ' ' + tgt_successor + ']'
		completion2 = tgt_letters[0] + ' ' + tgt_letters[1] + ' ' + tgt_letters[1] + ' ' + tgt_successor + ' ' + tgt_successor + ' ' + tgt_successor + ']'
		all_prob7_completion.append([completion1, completion2])
# Shuffle
all_prob7 = np.array(all_prob7)
all_prob7_completion = np.array(all_prob7_completion)
random_ind = np.arange(all_prob7.shape[0])
np.random.shuffle(random_ind)
all_prob7 = all_prob7[random_ind]
all_prob7_completion = all_prob7_completion[random_ind]

# All problems
all_prob = np.array([all_prob1, all_prob2, all_prob3, all_prob4, all_prob5, all_prob6, all_prob7], dtype=object)
all_prob_completion = np.array([all_prob1_completion, all_prob2_completion, all_prob3_completion, all_prob4_completion, all_prob5_completion, all_prob6_completion, all_prob7_completion], dtype=object)
all_prob_names = np.array([prob1_name, prob2_name, prob3_name, prob4_name, prob5_name, prob6_name, prob7_name])

# Save all problems
np.savez('./letter_string_analogies.npz', all_prob=all_prob, all_prob_completion=all_prob_completion, all_prob_names=all_prob_names)






