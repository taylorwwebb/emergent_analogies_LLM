import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint
from itertools import combinations
import builtins
import argparse
import os

def check_path(path):
	if not os.path.exists(path):
		os.mkdir(path)

def hide_top_right(ax):
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--sentence', action='store_true', help="Present problem in sentence format.")
parser.add_argument('--noprompt', action='store_true', help="Present problem without prompt.")
args = parser.parse_args()

# Load data
if args.sentence:
	all_responses = np.load('./gpt3_letterstring_results_sentence.npz')['all_prob_type_responses']
elif args.noprompt:
	all_responses = np.load('./gpt3_letterstring_results_noprompt.npz')['all_prob_type_responses']
else:
	all_responses = np.load('./gpt3_letterstring_results.npz')['all_prob_type_responses']
N_prob_types = all_responses.shape[0]
N_trials_per_prob_type = all_responses.shape[1]
# Load problems
all_prob = np.load('./all_prob.npz', allow_pickle=True)['all_prob']
prob_types = builtins.list(all_prob.item().keys())

# All possible combinations of transformations and generalizations
trans = ['succ', 'pred', 'add_letter', 'remove_redundant', 'fix_alphabet', 'sort']
gen = ['larger_int', 'longer_targ', 'group', 'interleaved', 'letter2num', 'reverse']
all_trans = []
all_gen = []
for t in range(len(trans)):
	all_trans.append(trans[t])
	all_gen.append([])
for t in range(len(trans)):
	for g in range(len(gen)):
		all_trans.append(trans[t])
		all_gen.append([gen[g]])
all_2comb = builtins.list(combinations(np.arange(len(gen)),2))
for t in range(len(trans)):
	for c in range(len(all_2comb)):
		all_trans.append(trans[t])
		all_gen.append([gen[all_2comb[c][0]], gen[all_2comb[c][1]]])
all_3comb = builtins.list(combinations(np.arange(len(gen)),3))
for t in range(len(trans)):
	for c in range(len(all_3comb)):
		all_trans.append(trans[t])
		all_gen.append([gen[all_3comb[c][0]], gen[all_3comb[c][1]], gen[all_3comb[c][2]]])
N_prob_subtype = len(all_trans)
subtype_counts = np.zeros(N_prob_subtype)

# Calculate performance
all_prob_type_correct_pred = []
all_prob_type_subtype = []
all_prob_N_gen = []
all_prob_realworld = []
for p in range(N_prob_types):
	all_correct_pred = []
	all_subtype = []
	all_N_gen = []
	all_realworld = []
	for t in range(N_trials_per_prob_type):
		response = all_responses[p][t]
		if args.sentence:
			if response[-1] == '.':
				response = response[:-1]
			if response[0] == ' ':
				response = response[1:]
			linebreak = True
			while linebreak:
				if response[:1] == '\n':
					response = response[1:]
				else:
					linebreak = False
			response_parsed = response.split(' ')
			correct_answer = all_prob.item()[prob_types[p]]['prob'][t][1][1]
			if np.array(correct_answer).astype(str).shape[0] == np.array(response_parsed).astype(str).shape[0]:
				correct_pred = np.all(np.array(correct_answer).astype(str) == np.array(response_parsed))
			else:
				correct_pred = False
			all_correct_pred.append(correct_pred)
		else:
			if response[-1] == ']':
				response = response[:-1]
			response_parsed = response.split(' ')
			correct_answer = all_prob.item()[prob_types[p]]['prob'][t][1][1]
			if np.array(correct_answer).astype(str).shape[0] == np.array(response_parsed).astype(str).shape[0]:
				correct_pred = np.all(np.array(correct_answer).astype(str) == np.array(response_parsed))
			else:
				correct_pred = False
			all_correct_pred.append(correct_pred)
		# Classify problem subtype
		for sp in range(N_prob_subtype):
			if all_prob.item()[prob_types[p]]['trans'][t] == all_trans[sp]:
				if len(all_prob.item()[prob_types[p]]['gen'][t]) == len(all_gen[sp]):
					if np.all(np.sort(all_prob.item()[prob_types[p]]['gen'][t]) == np.sort(np.array(all_gen[sp]))):
						all_subtype.append(sp)
						subtype_counts[sp] += 1
		# Number of generalizations
		N_gen = len(all_prob.item()[prob_types[p]]['gen'][t])
		all_N_gen.append(N_gen)
		# Real-world problems
		if 'realworld' in all_prob.item()[prob_types[p]]['gen'][t]:
			all_realworld.append(1)
		else:
			all_realworld.append(0)
	all_prob_type_correct_pred.append(all_correct_pred)
	all_prob_N_gen.append(all_N_gen)
	all_prob_realworld.append(all_realworld)
	if len(all_subtype) > 0:
		all_prob_type_subtype.append(all_subtype)
# Convert to arrays
all_prob_type_correct_pred = np.array(all_prob_type_correct_pred)
all_prob_type_subtype = np.array(all_prob_type_subtype)
all_prob_N_gen = np.array(all_prob_N_gen)
all_prob_realworld = np.array(all_prob_realworld)

# Create directory for results
if args.sentence:
	results_dir = './GPT3_results_sentence/'
elif args.noprompt:
	results_dir = './GPT3_results_noprompt/'
else:
	results_dir = './GPT3_results/'
check_path(results_dir)

# Save individual trial results
np.savez(results_dir + 'ind_trial_results.npz', all_prob_type_correct_pred=all_prob_type_correct_pred, all_prob_type_subtype=all_prob_type_subtype, all_prob_N_gen=all_prob_N_gen, all_prob_realworld=all_prob_realworld)

# Correlation analysis
all_subtype_acc = []
for sp in range(N_prob_subtype):
	all_subtype_acc.append(all_prob_type_correct_pred[:all_prob_type_subtype.shape[0],:][all_prob_type_subtype == sp].mean())
np.savez(results_dir + 'prob_subtype_acc.npz', subtype_acc=all_subtype_acc, subtype_counts=subtype_counts)

# Plot settings
gpt3_color = 'darkslateblue'
plot_fontsize = 10
title_fontsize = 12
axis_label_fontsize = 12
bar_width = 0.8

# Calculate accuracy for all zero-generalization problems
all_zerogen_prob_types = ['succ', 'pred', 'add_letter', 'remove_redundant', 'fix_alphabet', 'sort']
all_zerogen_acc = []
all_zerogen_ci_lower = []
all_zerogen_ci_upper = []
for p in range(len(all_zerogen_prob_types)):
	correct_pred = np.array(all_prob_type_correct_pred[np.where(np.array(prob_types)==all_zerogen_prob_types[p])[0][0]]).astype(float)
	all_zerogen_acc.append(correct_pred.mean())
	ci_lower, ci_upper = proportion_confint(correct_pred.sum(), correct_pred.shape[0])
	all_zerogen_ci_lower.append(ci_lower)
	all_zerogen_ci_upper.append(ci_upper)
all_zerogen_acc = np.array(all_zerogen_acc)
all_zerogen_ci_lower = np.array(all_zerogen_ci_lower)
all_zerogen_ci_upper = np.array(all_zerogen_ci_upper)
all_zerogen_lower_err = all_zerogen_acc - all_zerogen_ci_lower
all_zerogen_upper_err =  all_zerogen_ci_upper - all_zerogen_acc
all_zerogen_err = np.array([all_zerogen_lower_err, all_zerogen_upper_err])
# Sort based on accuracy
rank_order = np.flip(np.argsort(all_zerogen_acc))
# Plot
all_zerogen_prob_type_names = ['Successor', 'Predecessor', 'Extend\nsequence', 'Remove\nredundant\nletter', 'Fix\nalphabetic\nsequence', 'Sort']
x_points = np.arange(len(all_zerogen_prob_types))
ax = plt.subplot(111)
plt.bar(x_points, all_zerogen_acc[rank_order], yerr=all_zerogen_err[:,rank_order], color=gpt3_color, edgecolor='black', width=bar_width)
plt.ylim([0,1])
plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','0.2','0.4','0.6','0.8','1'], fontsize=plot_fontsize)
plt.ylabel('Generative accuracy', fontsize=axis_label_fontsize)
plt.xticks(x_points, np.array(all_zerogen_prob_type_names)[rank_order], fontsize=plot_fontsize)
plt.xlabel('Transformation type', fontsize=axis_label_fontsize)
plt.title('Zero-generalization problems')
plt.legend(['GPT-3'],fontsize=plot_fontsize,frameon=False)
hide_top_right(ax)
plt.tight_layout()
plt.savefig(results_dir + 'zerogen_acc.png', dpi=300, bbox_inches="tight")
plt.close()
# Save results
np.savez(results_dir + 'zerogen_acc.npz', all_acc=all_zerogen_acc, all_err=all_zerogen_err)

# Calculate all accuracy for one-generalization problems
all_onegen_prob_types = ['larger_int', 'longer_targ', 'group', 'interleaved', 'letter2num', 'reverse']
all_onegen_acc = []
all_onegen_ci_lower = []
all_onegen_ci_upper = []
for p in range(len(all_onegen_prob_types)):
	correct_pred = np.array(all_prob_type_correct_pred[np.where(np.array(prob_types)==all_onegen_prob_types[p])[0][0]]).astype(float)
	all_onegen_acc.append(correct_pred.mean())
	ci_lower, ci_upper = proportion_confint(correct_pred.sum(), correct_pred.shape[0])
	all_onegen_ci_lower.append(ci_lower)
	all_onegen_ci_upper.append(ci_upper)
all_onegen_acc = np.array(all_onegen_acc)
all_onegen_ci_lower = np.array(all_onegen_ci_lower)
all_onegen_ci_upper = np.array(all_onegen_ci_upper)
all_onegen_lower_err = all_onegen_acc - all_onegen_ci_lower
all_onegen_upper_err =  all_onegen_ci_upper - all_onegen_acc
all_onegen_err = np.array([all_onegen_lower_err, all_onegen_upper_err])
# Sort based on accuracy
rank_order = np.flip(np.argsort(all_onegen_acc))
# Plot
all_onegen_prob_type_names = ['Larger\ninterval', 'Longer\ntarget', 'Grouping', 'Interleaved\ndistractor', 'Letter-to-\nnumber', 'Reverse\norder']
x_points = np.arange(len(all_onegen_prob_types))
ax = plt.subplot(111)
plt.bar(x_points, all_onegen_acc[rank_order], yerr=all_onegen_err[:,rank_order], color=gpt3_color, edgecolor='black', width=bar_width)
plt.ylim([0,1])
plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','0.2','0.4','0.6','0.8','1'], fontsize=plot_fontsize)
plt.ylabel('Generative accuracy', fontsize=axis_label_fontsize)
plt.xticks(x_points, np.array(all_onegen_prob_type_names)[rank_order], fontsize=plot_fontsize)
plt.xlabel('Generalization type', fontsize=axis_label_fontsize)
plt.title('One-generalization problems')
plt.legend(['GPT-3'],fontsize=plot_fontsize,frameon=False)
hide_top_right(ax)
plt.tight_layout()
plt.savefig(results_dir + 'onegen_acc.png', dpi=300, bbox_inches="tight")
plt.close()
# Save results
np.savez(results_dir + 'onegen_acc.npz', all_acc=all_onegen_acc, all_err=all_onegen_err)

# Calculate accuracy by number of generalizations
gen_ind = [[0,6], [6,12], [12,18], [18,24]]
all_gen_acc = []
all_gen_ci_lower = []
all_gen_ci_upper = []
for p in range(len(gen_ind)):
	correct_pred = np.array(all_prob_type_correct_pred[gen_ind[p][0]:gen_ind[p][1]]).flatten().astype(float)
	acc = correct_pred.mean()
	all_gen_acc.append(acc)
	ci_lower, ci_upper = proportion_confint(correct_pred.sum(), correct_pred.shape[0])
	all_gen_ci_lower.append(ci_lower)
	all_gen_ci_upper.append(ci_upper)
all_gen_ac = np.array(all_gen_acc)
all_gen_ci_lower = np.array(all_gen_ci_lower)
all_gen_ci_upper = np.array(all_gen_ci_upper)
all_gen_lower_err = all_gen_acc - all_gen_ci_lower
all_gen_upper_err =  all_gen_ci_upper - all_gen_acc
all_gen_err = np.array([all_gen_lower_err, all_gen_upper_err])
# Plot
all_gen_prob_type_names = np.arange(len(gen_ind)).astype(str)
x_points = np.arange(len(gen_ind))
ax = plt.subplot(111)
plt.bar(x_points, all_gen_acc, yerr=all_gen_err, color=gpt3_color, edgecolor='black', width=bar_width)
plt.ylim([0,1])
plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','0.2','0.4','0.6','0.8','1'], fontsize=plot_fontsize)
plt.ylabel('Generative accuracy', fontsize=axis_label_fontsize)
plt.xticks(x_points, all_gen_prob_type_names, fontsize=plot_fontsize)
plt.xlabel('Number of generalizations', fontsize=axis_label_fontsize)
plt.legend(['GPT-3'],fontsize=plot_fontsize,frameon=False)
hide_top_right(ax)
plt.tight_layout()
plt.savefig(results_dir + 'all_gen_acc.png', dpi=300, bbox_inches="tight")
plt.close()
# Save results
np.savez(results_dir + 'all_gen_acc.npz', all_acc=all_gen_acc, all_err=all_gen_err)

# Calculate accuracy for all real-world concept problems
all_realworld_prob_types = ['realworld_succ', 'realworld_pred', 'realworld_add_letter', 'realworld_sort']
all_realworld_acc = []
all_realworld_ci_lower = []
all_realworld_ci_upper = []
for p in range(len(all_realworld_prob_types)):
	correct_pred = np.array(all_prob_type_correct_pred[np.where(np.array(prob_types)==all_realworld_prob_types[p])[0][0]]).astype(float)
	all_realworld_acc.append(correct_pred.mean())
	ci_lower, ci_upper = proportion_confint(correct_pred.sum(), correct_pred.shape[0])
	all_realworld_ci_lower.append(ci_lower)
	all_realworld_ci_upper.append(ci_upper)
all_realworld_acc = np.array(all_realworld_acc)
all_realworld_ci_lower = np.array(all_realworld_ci_lower)
all_realworld_ci_upper = np.array(all_realworld_ci_upper)
all_realworld_lower_err = all_realworld_acc - all_realworld_ci_lower
all_realworld_upper_err =  all_realworld_ci_upper - all_realworld_acc
all_realworld_err = np.array([all_realworld_lower_err, all_realworld_upper_err])
# Sort based on accuracy
rank_order = np.flip(np.argsort(all_realworld_acc))
# Plot
all_realworld_prob_type_names = ['Successor', 'Predecessor', 'Extend\nsequence', 'Sort']
x_points = np.arange(len(all_realworld_prob_types))
ax = plt.subplot(111)
plt.bar(x_points, all_realworld_acc[rank_order], yerr=all_realworld_err[:,rank_order], color=gpt3_color, edgecolor='black', width=bar_width)
plt.ylim([0,1])
plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','0.2','0.4','0.6','0.8','1'], fontsize=plot_fontsize)
plt.ylabel('Generative accuracy', fontsize=axis_label_fontsize)
plt.xticks(x_points, np.array(all_realworld_prob_type_names)[rank_order], fontsize=plot_fontsize)
plt.xlabel('Transformation type', fontsize=axis_label_fontsize)
plt.title('Real-world concept problems')
plt.legend(['GPT-3'],fontsize=plot_fontsize,frameon=False)
hide_top_right(ax)
plt.tight_layout()
plt.savefig(results_dir + 'realworld_acc.png', dpi=300, bbox_inches="tight")
plt.close()
# Save results
np.savez(results_dir + 'realworld_acc.npz', all_acc=all_realworld_acc, all_err=all_realworld_err)
