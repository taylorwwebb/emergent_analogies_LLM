import numpy as np
import csv
import builtins

# Load human behavioral data
human_data = np.load('./behavioral_results/ind_subj_results.npz')
human_correct_pred = human_data['all_subj_correct_pred']
human_prob_subtype = human_data['all_subj_prob_subtype']
human_N_gen = human_data['all_subj_N_gen']
human_realworld = human_data['all_subj_realworld']

# Load GPT-3 data
gpt3_data =  np.load('./GPT3_results/ind_trial_results.npz')
gpt3_correct_pred = gpt3_data['all_prob_type_correct_pred']
gpt3_prob_subtype = gpt3_data['all_prob_type_subtype']
gpt3_N_gen = gpt3_data['all_prob_N_gen']
gpt3_realworld = gpt3_data['all_prob_realworld']

# Create dataset
subjID = []
correct_pred = []
human_vs_gpt = []
prob_subtype = []
N_gen = []
realworld = []

# Add human data
N_subj = human_correct_pred.shape[0]
N_prob = human_correct_pred.shape[1]
for s in range(N_subj):
	for p in range(N_prob):
		subjID.append(s)
		correct_pred.append(int(human_correct_pred[s,p]))
		human_vs_gpt.append(0)
		N_gen.append(human_N_gen[s,p])
		realworld.append(human_realworld[s,p])
		if p < human_prob_subtype.shape[1]:
			prob_subtype.append(human_prob_subtype[s,p])
		else:
			prob_subtype.append(-1)

# Add GPT-3 data
N_trials_per_prob = gpt3_correct_pred.shape[1]
for t in range(N_trials_per_prob):
	for p in range(N_prob):
		subjID.append(s+1)
		correct_pred.append(int(gpt3_correct_pred[p,t]))
		human_vs_gpt.append(1)
		N_gen.append(gpt3_N_gen[p,t])
		realworld.append(gpt3_realworld[p,t])
		if p < gpt3_prob_subtype.shape[0]:
			prob_subtype.append(gpt3_prob_subtype[p,t])
		else:
			prob_subtype.append(-1)

# Write csv files
# Create file
f = open('./letterstring_data.csv', 'w')
writer = csv.writer(f)
# Header
header = ['subjID', 'correct_pred', 'human_vs_gpt', 'prob_subtype', 'N_gen']
writer.writerow(header)
# Write data
for i in range(len(subjID)):
	if realworld[i] == 0:
		data_row = [subjID[i], correct_pred[i], human_vs_gpt[i], prob_subtype[i], N_gen[i]]
		writer.writerow(data_row)
# Close file
f.close()
