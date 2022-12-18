import numpy as np
import csv
import builtins

# Load human behavioral data
human_data = np.load('./exp1_behavioral_data/ind_subj_results.npz')
all_human_gen_correct_pred = human_data['all_subj_gen_correct_pred']
all_human_MC_correct_pred = human_data['all_subj_MC_correct_pred']
N_subj = all_human_gen_correct_pred.shape[0]
N_prob = all_human_gen_correct_pred.shape[1]
# Load data about # of unique relations in each problem
N_unique_rules_2rule_prob = np.load('./N_unique_rules_2rule_prob.npz')['N_unique_rules']
N_unique_rules_3rule_prob = np.load('./N_unique_rules_3rule_prob.npz')['N_unique_rules']
# Create dataset
subjID = []
gen_correct_pred = []
MC_correct_pred = []
human_vs_gpt = []
N_unique_rules = []
prob_type = []
onerule_rule_type = []
tworule_prog_noprog = []
aligned_permuted = []
for s in range(N_subj):
	for p in range(N_prob):
		# Subject ID
		subjID.append(s)
		# Correct prediction
		gen_correct_pred.append(all_human_gen_correct_pred[s,p])
		MC_correct_pred.append(all_human_MC_correct_pred[s,p])
		# Human vs. GPT-3
		human_vs_gpt.append(0)
		# Problem-type specific variables
		# One-rule problems
		if p < 6:
			# Problem type
			prob_type.append(0)
			# Number of unique rules
			N_unique_rules.append(0)
			# Rule type
			# Constant rule
			if p == 0 or p == 1:
				onerule_rule_type.append(0)
			# Distribution-of-3 rule
			elif p == 2 or p == 3:
				onerule_rule_type.append(1)
			# Progression rule
			elif p == 4 or p == 5:
				onerule_rule_type.append(2)
			# Dummy code for aligned vs. permuted (logic rules only)
			aligned_permuted.append(-1)
			# Dummy code for two-rule progression vs. no progression
			tworule_prog_noprog.append(-1)
		# Two-rule problems
		elif p >= 6 and p < 12:
			# Problem type
			prob_type.append(1)
			# Dummy code for one-rule rule type
			onerule_rule_type.append(-1)
			# Dummy code for aligned vs. permuted (logic rules only)
			aligned_permuted.append(-1)
			# Number of unique rules
			N_unique_rules.append(N_unique_rules_2rule_prob[p-6] - 1)
			# Progression rule present
			if p == 6 or p == 7 or p == 9:
				tworule_prog_noprog.append(0)
			elif p == 8 or p == 10 or p == 11:
				tworule_prog_noprog.append(1)
		# Three-rule problems
		elif p >= 12 and p < 22:
			# Problem type
			prob_type.append(2)
			# Dummy code for one-rule rule type
			onerule_rule_type.append(-1)
			# Dummy code for aligned vs. permuted (logic rules only)
			aligned_permuted.append(-1)
			# Number of unique rules
			N_unique_rules.append(N_unique_rules_3rule_prob[p-12] - 1)
			# Dummy code for two-rule progression vs. no progression
			tworule_prog_noprog.append(-1)
		# Logic problems
		elif p >= 22:
			# Problem type
			prob_type.append(3)
			# Dummy code for one-rule rule type
			onerule_rule_type.append(-1)
			# Dummy code for two-rule progression vs. no progression
			tworule_prog_noprog.append(-1)
			# Number of unique rules
			N_unique_rules.append(0)
			# Spatially aligned elements
			if p < 27:
				aligned_permuted.append(0)
			else:
				aligned_permuted.append(1)
# Load GPT-3 data
gpt3_data = np.load('./gpt_matprob_results.npz', allow_pickle=True)
prob_type_names = builtins.list(gpt3_data['all_gen_correct_pred'].item().keys())
# Add to dataset
for p in range(N_prob):
	# Problem type name
	prob_type_name = prob_type_names[p]
	# Loop through all trials for this problem type
	N_trials = len(gpt3_data['all_gen_correct_pred'].item()[prob_type_name])
	for t in range(N_trials):
		# Subject ID
		subjID.append(N_subj)
		# Correct prediction
		gen_correct_pred.append(int(gpt3_data['all_gen_correct_pred'].item()[prob_type_name][t]))
		MC_correct_pred.append(int(gpt3_data['all_MC_correct_pred'].item()[prob_type_name][t]))
		# Human vs. GPT-3
		human_vs_gpt.append(1)
		# Problem-type specific variables
		# One-rule problems
		if p < 6:
			# Problem type
			prob_type.append(0)
			# Number of unique rules
			N_unique_rules.append(0)
			# Rule type
			# Constant rule
			if p == 0 or p == 1:
				onerule_rule_type.append(0)
			# Distribution-of-3 rule
			elif p == 2 or p == 3:
				onerule_rule_type.append(1)
			# Progression rule
			elif p == 4 or p == 5:
				onerule_rule_type.append(2)
			# Dummy code for aligned vs. permuted (logic rules only)
			aligned_permuted.append(-1)
			# Dummy code for two-rule progression vs. no progression
			tworule_prog_noprog.append(-1)
		# Two-rule problems
		elif p >= 6 and p < 12:
			# Problem type
			prob_type.append(1)
			# Dummy code for one-rule rule type
			onerule_rule_type.append(-1)
			# Number of unique rules
			N_unique_rules.append(N_unique_rules_2rule_prob[p-6] - 1)
			# Dummy code for aligned vs. permuted (logic rules only)
			aligned_permuted.append(-1)
			# Progression rule present
			if p == 6 or p == 7 or p == 9:
				tworule_prog_noprog.append(0)
			elif p == 8 or p == 10 or p == 11:
				tworule_prog_noprog.append(1)
		# Three-rule problems
		elif p >= 12 and p < 22:
			# Problem type
			prob_type.append(2)
			# Dummy code for one-rule rule type
			onerule_rule_type.append(-1)
			# Number of unique rules
			N_unique_rules.append(N_unique_rules_3rule_prob[p-12] - 1)
			# Dummy code for aligned vs. permuted (logic rules only)
			aligned_permuted.append(-1)
			# Dummy code for two-rule progression vs. no progression
			tworule_prog_noprog.append(-1)
		# Logic problems
		elif p >= 22:
			# Problem type
			prob_type.append(3)
			# Dummy code for one-rule rule type
			onerule_rule_type.append(-1)
			# Dummy code for two-rule progression vs. no progression
			tworule_prog_noprog.append(-1)
			# Number of unique rules
			N_unique_rules.append(0)
			# Spatially aligned elements
			if p < 27:
				aligned_permuted.append(0)
			else:
				aligned_permuted.append(1)

# Write csv file
# Create file
f = open('./exp1_all_data.csv', 'w')
writer = csv.writer(f)
# Header
header = ['subjID', 'gen_correct_pred', 'MC_correct_pred', 'human_vs_gpt', 'N_unique_rules', 'prob_type', 'onerule_rule_type', 'aligned_permuted', 'tworule_prog_noprog']
writer.writerow(header)
# Write data
for i in range(len(subjID)):
	data_row = [subjID[i], gen_correct_pred[i], MC_correct_pred[i], human_vs_gpt[i], N_unique_rules[i], prob_type[i], onerule_rule_type[i], aligned_permuted[i], tworule_prog_noprog[i]]
	writer.writerow(data_row)
# Close file
f.close()







