import numpy as np
import json

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

# Number of problems per category
max_N_prob_per_cat = 10
N_instances_per_prob = 100
# All categories
all_cat = ['one_rule', 'two_rule', 'three_rule', 'four_rule', 'five_rule']

# Load 1-3-rule problems
one_3_rule_prob_fname = './all_problems.npz'
one_3_rule_prob = np.load(one_3_rule_prob_fname, allow_pickle=True)['all_problems']

# Load 4-5-rule problems
four_5_rule_prob_fname = './all_4_5_rule_problems.npz'
four_5_rule_prob = np.load(four_5_rule_prob_fname, allow_pickle=True)['all_problems']

# Subsample problems and save as js script, also as numpy file
all_problems_np = {}
all_problems_js = {}
# 1-rule problems
one_rule_prob_names = ['row_constant', 'col_constant', 'dist3_diag1', 'dist3_diag2', 'prog_size1', 'prog_size2']
for prob_name in one_rule_prob_names:
	all_problems_np, all_problems_js = save_prob(one_3_rule_prob.item()[prob_name]['prob'], one_3_rule_prob.item()[prob_name]['answer_choices'], one_3_rule_prob.item()[prob_name]['correct_ind'], prob_name, all_problems_np, all_problems_js)
# 2-rule problems
for prob_name in one_3_rule_prob.item().keys():
	if 'two_rule' in prob_name:
		all_problems_np, all_problems_js = save_prob(one_3_rule_prob.item()[prob_name]['prob'], one_3_rule_prob.item()[prob_name]['answer_choices'], one_3_rule_prob.item()[prob_name]['correct_ind'], prob_name, all_problems_np, all_problems_js)
# 3-rule problems
for prob_name in one_3_rule_prob.item().keys():
	if 'three_rule' in prob_name:
		all_problems_np, all_problems_js = save_prob(one_3_rule_prob.item()[prob_name]['prob'], one_3_rule_prob.item()[prob_name]['answer_choices'], one_3_rule_prob.item()[prob_name]['correct_ind'], prob_name, all_problems_np, all_problems_js)
# 4-rule problems
all_four_rule_prob_names = []
for prob_name in four_5_rule_prob.item().keys():
	if 'four_rule' in prob_name:
		all_four_rule_prob_names.append(prob_name)
all_sampled_probs = []
for prob in range(max_N_prob_per_cat):
	# Sample problem instances
	prob_instances = []
	answer_choices = []
	correct_ind = []
	for i in range(N_instances_per_prob):
		duplicate = True
		while duplicate:
			prob_type = all_four_rule_prob_names[np.floor(np.random.rand() * len(all_four_rule_prob_names)).astype(int)]
			prob_ind = np.floor(np.random.rand() * N_instances_per_prob).astype(int)
			combined_name = prob_type + '_' + str(prob_ind)
			if np.logical_not(np.any(combined_name == np.array(all_sampled_probs))):
				duplicate = False
				all_sampled_probs.append(combined_name)
				prob_instances.append(four_5_rule_prob.item()[prob_type]['prob'][prob_ind])
				answer_choices.append(four_5_rule_prob.item()[prob_type]['answer_choices'][prob_ind])
				correct_ind.append(four_5_rule_prob.item()[prob_type]['correct_ind'][prob_ind])
	prob_instances = np.array(prob_instances)
	answer_choices = np.array(answer_choices)
	correct_ind = np.array(correct_ind)
	all_problems_np, all_problems_js = save_prob(prob_instances, answer_choices, correct_ind, 'four_rule_prob' + str(prob), all_problems_np, all_problems_js)
# 5-rule problems
all_five_rule_prob_names = []
for prob_name in four_5_rule_prob.item().keys():
	if 'five_rule' in prob_name:
		all_five_rule_prob_names.append(prob_name)
all_sampled_probs = []
for prob in range(max_N_prob_per_cat):
	# Sample problem instances
	prob_instances = []
	answer_choices = []
	correct_ind = []
	for i in range(N_instances_per_prob):
		duplicate = True
		while duplicate:
			prob_type = all_five_rule_prob_names[np.floor(np.random.rand() * len(all_five_rule_prob_names)).astype(int)]
			prob_ind = np.floor(np.random.rand() * N_instances_per_prob).astype(int)
			combined_name = prob_type + '_' + str(prob_ind)
			if np.logical_not(np.any(combined_name == np.array(all_sampled_probs))):
				duplicate = False
				all_sampled_probs.append(combined_name)
				prob_instances.append(four_5_rule_prob.item()[prob_type]['prob'][prob_ind])
				answer_choices.append(four_5_rule_prob.item()[prob_type]['answer_choices'][prob_ind])
				correct_ind.append(four_5_rule_prob.item()[prob_type]['correct_ind'][prob_ind])
	prob_instances = np.array(prob_instances)
	answer_choices = np.array(answer_choices)
	correct_ind = np.array(correct_ind)
	all_problems_np, all_problems_js = save_prob(prob_instances, answer_choices, correct_ind, 'five_rule_prob' + str(prob), all_problems_np, all_problems_js)
# Save numpy file
np_fname = './all_problems_1thru5.npz'
np.savez(np_fname, all_problems=all_problems_np)
# Convert to json string
json_string = json.dumps(all_problems_js)
# Write to js script
js_fname = './all_problems_1thru5.js'
js_fid = open(js_fname, 'w')
js_fid.write('var all_problems = ' + json_string)
js_fid.close()

