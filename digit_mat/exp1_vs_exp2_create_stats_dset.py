import numpy as np
import csv
import builtins

# Load human behavioral data for experiment 1
human_data = np.load('./exp1_behavioral_data/ind_subj_results.npz')
all_human_gen_correct_pred = human_data['all_subj_gen_correct_pred']
all_human_MC_correct_pred = human_data['all_subj_MC_correct_pred']
N_subj_exp1 = all_human_gen_correct_pred.shape[0]
N_prob = all_human_gen_correct_pred.shape[1]
# Create dataset
subjID = []
gen_correct_pred = []
MC_correct_pred = []
human_vs_gpt = []
prob_type = []
exp1_vs_exp2 = []
onerule_subtype = []
for s in range(N_subj_exp1):
	for p in range(N_prob):
		if p < 22:
			# Subject ID
			subjID.append(s)
			# Correct prediction
			gen_correct_pred.append(all_human_gen_correct_pred[s,p])
			MC_correct_pred.append(all_human_MC_correct_pred[s,p])
			# Human vs. GPT-3
			human_vs_gpt.append(0)
			# Experiment 1 vs. experiment 4
			exp1_vs_exp2.append(0)
		# Problem-type specific variables
		# One-rule problems
		if p < 6:
			# Problem type
			prob_type.append(0)
			# One-rule subtypes
			if p < 2:
				onerule_subtype.append(0)
			elif p == 2 or p == 3:
				onerule_subtype.append(1)
			elif p == 4 or p == 5:
				onerule_subtype.append(2)
		# Two-rule problems
		elif p >= 6 and p < 12:
			# Problem type
			prob_type.append(1)
			# Dummy code one-rule subtypes
			onerule_subtype.append(-1)
		# Three-rule problems
		elif p >= 12 and p < 22:
			# Problem type
			prob_type.append(2)
			# Dummy code one-rule subtypes
			onerule_subtype.append(-1)
# Load GPT-3 data for experiment 1
gpt3_data = np.load('./gpt_matprob_results.npz', allow_pickle=True)
prob_type_names = builtins.list(gpt3_data['all_gen_correct_pred'].item().keys())
# Add to dataset
for p in range(N_prob):
	# Problem type name
	prob_type_name = prob_type_names[p]
	# Loop through all trials for this problem type
	N_trials = len(gpt3_data['all_gen_correct_pred'].item()[prob_type_name])
	for t in range(N_trials):
		if p < 22:
			# Subject ID
			subjID.append(N_subj_exp1)
			# Correct prediction
			gen_correct_pred.append(int(gpt3_data['all_gen_correct_pred'].item()[prob_type_name][t]))
			MC_correct_pred.append(int(gpt3_data['all_MC_correct_pred'].item()[prob_type_name][t]))
			# Human vs. GPT-3
			human_vs_gpt.append(1)
			# Experiment 1 vs. experiment 4
			exp1_vs_exp2.append(0)
		# Problem-type specific variables
		# One-rule problems
		if p < 6:
			# Problem type
			prob_type.append(0)
			# One-rule subtypes
			if p < 2:
				onerule_subtype.append(0)
			elif p == 2 or p == 3:
				onerule_subtype.append(1)
			elif p == 4 or p == 5:
				onerule_subtype.append(2)
		# Two-rule problems
		elif p >= 6 and p < 12:
			# Problem type
			prob_type.append(1)
			# Dummy code one-rule subtypes
			onerule_subtype.append(-1)
		# Three-rule problems
		elif p >= 12 and p < 22:
			# Problem type
			prob_type.append(2)
			# Dummy code one-rule subtypes
			onerule_subtype.append(-1)

# Load human behavioral data for experiment 2
human_data = np.load('./exp2_behavioral_data/ind_subj_results.npz')
all_human_gen_correct_pred = human_data['all_subj_gen_correct_pred']
all_human_MC_correct_pred = human_data['all_subj_MC_correct_pred']
N_subj_exp2 = all_human_gen_correct_pred.shape[0]
N_prob = all_human_gen_correct_pred.shape[1]
for s in range(N_subj_exp2):
	for p in range(N_prob):
		if p < 22:
			# Subject ID
			subjID.append(N_subj_exp1 + 1 + s)
			# Correct prediction
			gen_correct_pred.append(all_human_gen_correct_pred[s,p])
			MC_correct_pred.append(all_human_MC_correct_pred[s,p])
			# Human vs. GPT-3
			human_vs_gpt.append(0)
			# Experiment 1 vs. experiment 4
			exp1_vs_exp2.append(1)
		# Problem-type specific variables
		# One-rule problems
		if p < 6:
			# Problem type
			prob_type.append(0)
			# One-rule subtypes
			if p < 2:
				onerule_subtype.append(0)
			elif p == 2 or p == 3:
				onerule_subtype.append(1)
			elif p == 4 or p == 5:
				onerule_subtype.append(2)
		# Two-rule problems
		elif p >= 6 and p < 12:
			# Problem type
			prob_type.append(1)
			# Dummy code one-rule subtypes
			onerule_subtype.append(-1)
		# Three-rule problems
		elif p >= 12 and p < 22:
			# Problem type
			prob_type.append(2)
			# Dummy code one-rule subtypes
			onerule_subtype.append(-1)

# Load GPT-3 data
gpt3_data = np.load('./gpt_matprob_results_1thru5.npz', allow_pickle=True)
prob_type_names = builtins.list(gpt3_data['all_gen_correct_pred'].item().keys())
# Add to dataset
for p in range(N_prob):
	# Problem type name
	prob_type_name = prob_type_names[p]
	# Loop through all trials for this problem type
	N_trials = len(gpt3_data['all_gen_correct_pred'].item()[prob_type_name])
	for t in range(N_trials):
		if p < 22:
			# Subject ID
			subjID.append(N_subj_exp1)
			# Correct prediction
			gen_correct_pred.append(int(gpt3_data['all_gen_correct_pred'].item()[prob_type_name][t]))
			MC_correct_pred.append(int(gpt3_data['all_MC_correct_pred'].item()[prob_type_name][t]))
			# Human vs. GPT-3
			human_vs_gpt.append(1)
			# Experiment 1 vs. experiment 2
			exp1_vs_exp2.append(1)
		# Problem-type specific variables
		# One-rule problems
		if p < 6:
			# Problem type
			prob_type.append(0)
			# One-rule subtypes
			if p < 2:
				onerule_subtype.append(0)
			elif p == 2 or p == 3:
				onerule_subtype.append(1)
			elif p == 4 or p == 5:
				onerule_subtype.append(2)
		# Two-rule problems
		elif p >= 6 and p < 12:
			# Problem type
			prob_type.append(1)
			# Dummy code one-rule subtypes
			onerule_subtype.append(-1)
		# Three-rule problems
		elif p >= 12 and p < 22:
			# Problem type
			prob_type.append(2)
			# Dummy code one-rule subtypes
			onerule_subtype.append(-1)

# Write csv file
# Create file
f = open('./exp1_vs_exp2_all_data.csv', 'w')
writer = csv.writer(f)
# Header
header = ['subjID', 'gen_correct_pred', 'MC_correct_pred', 'human_vs_gpt', 'exp1_vs_exp2', 'prob_type', 'onerule_subtype']
writer.writerow(header)
# Write data
for i in range(len(subjID)):
	data_row = [subjID[i], gen_correct_pred[i], MC_correct_pred[i], human_vs_gpt[i], exp1_vs_exp2[i], prob_type[i], onerule_subtype[i]]
	writer.writerow(data_row)
# Close file
f.close()






