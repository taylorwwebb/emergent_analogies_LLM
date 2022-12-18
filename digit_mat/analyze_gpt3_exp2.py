import numpy as np
import matplotlib.pyplot as plt
import builtins
from statsmodels.stats.proportion import proportion_confint
import os

def check_path(path):
	if not os.path.exists(path):
		os.mkdir(path)

def hide_top_right(ax):
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')

# Load data
all_data = np.load('./gpt_matprob_results_1thru5.npz', allow_pickle=True)
MC_correct_pred = all_data['all_MC_correct_pred']
gen_correct_pred = all_data['all_gen_correct_pred']
all_prob_types = builtins.list(MC_correct_pred.item().keys())

## Analyze by major problem type
correct_pred = {'combined_gen': [],
				'combined_MC': [],
				'one_rule_gen': [],
				'one_rule_MC': [],
				'two_rule_gen': [],
				'two_rule_MC': [],
				'three_rule_gen': [],
				'three_rule_MC': [],
				'four_rule_gen': [],
				'four_rule_MC': [],
				'five_rule_gen': [],
				'five_rule_MC': []}
for prob_type in all_prob_types:
	correct_pred['combined_gen'].append(gen_correct_pred.item()[prob_type])
	correct_pred['combined_MC'].append(MC_correct_pred.item()[prob_type])
	if 'constant' in prob_type or 'dist3' in prob_type or 'prog' in prob_type:
		correct_pred['one_rule_gen'].append(gen_correct_pred.item()[prob_type])
		correct_pred['one_rule_MC'].append(MC_correct_pred.item()[prob_type])
	elif 'two_rule' in prob_type:
		correct_pred['two_rule_gen'].append(gen_correct_pred.item()[prob_type])
		correct_pred['two_rule_MC'].append(MC_correct_pred.item()[prob_type])
	elif 'three_rule' in prob_type:
		correct_pred['three_rule_gen'].append(gen_correct_pred.item()[prob_type])
		correct_pred['three_rule_MC'].append(MC_correct_pred.item()[prob_type])
	elif 'four_rule' in prob_type:
		correct_pred['four_rule_gen'].append(gen_correct_pred.item()[prob_type])
		correct_pred['four_rule_MC'].append(MC_correct_pred.item()[prob_type])
	elif 'five_rule' in prob_type:
		correct_pred['five_rule_gen'].append(gen_correct_pred.item()[prob_type])
		correct_pred['five_rule_MC'].append(MC_correct_pred.item()[prob_type])
# Convert to arrays
for key in correct_pred.keys():
	correct_pred[key] = np.concatenate(correct_pred[key])
# Calculate accuracy and confidence intervals
all_acc = {}
all_ci_lower = {}
all_ci_upper = {}
for key in correct_pred.keys():
	all_acc[key] = correct_pred[key].mean()
	all_ci_lower[key], all_ci_upper[key] = proportion_confint(correct_pred[key].sum(), correct_pred[key].shape[0])

# Directory for saving results
results_dir = './exp2_GPT3_data/'
check_path(results_dir)

# Save results
# Generative 
all_gen_acc = np.array([all_acc['one_rule_gen'], all_acc['two_rule_gen'], all_acc['three_rule_gen'], all_acc['four_rule_gen'], all_acc['five_rule_gen']])
all_gen_lower_ci = np.array([all_ci_lower['one_rule_gen'], all_ci_lower['two_rule_gen'], all_ci_lower['three_rule_gen'], all_ci_lower['four_rule_gen'], all_ci_lower['five_rule_gen']])
all_gen_upper_ci = np.array([all_ci_upper['one_rule_gen'], all_ci_upper['two_rule_gen'], all_ci_upper['three_rule_gen'], all_ci_upper['four_rule_gen'], all_ci_upper['five_rule_gen']])
all_gen_lower_err = all_gen_acc - all_gen_lower_ci
all_gen_upper_err =  all_gen_upper_ci - all_gen_acc
all_gen_err = np.array([all_gen_lower_err, all_gen_upper_err])
np.savez(results_dir + 'all_probcat_gen_acc.npz', acc=all_gen_acc, err=all_gen_err)
# Multiple-choice 
all_MC_acc = np.array([all_acc['one_rule_MC'], all_acc['two_rule_MC'], all_acc['three_rule_MC'], all_acc['four_rule_MC'], all_acc['five_rule_MC']])
all_MC_lower_ci = np.array([all_ci_lower['one_rule_MC'], all_ci_lower['two_rule_MC'], all_ci_lower['three_rule_MC'], all_ci_lower['four_rule_MC'], all_ci_lower['five_rule_MC']])
all_MC_upper_ci = np.array([all_ci_upper['one_rule_MC'], all_ci_upper['two_rule_MC'], all_ci_upper['three_rule_MC'], all_ci_upper['four_rule_MC'], all_ci_upper['five_rule_MC']])
all_MC_lower_err = all_MC_acc - all_MC_lower_ci
all_MC_upper_err =  all_MC_upper_ci - all_MC_acc
all_MC_err = np.array([all_MC_lower_err, all_MC_upper_err])
np.savez(results_dir + 'all_probcat_MC_acc.npz', acc=all_MC_acc, err=all_MC_err)

## Three major one-rule problem types
correct_pred = {'constant_gen': [],
				'constant_MC': [],
				'dist3_gen': [],
				'dist3_MC': [],
				'prog_gen': [],
				'prog_MC': []}
for prob_type in all_prob_types:
	if 'constant' in prob_type:
		correct_pred['constant_gen'].append(gen_correct_pred.item()[prob_type])
		correct_pred['constant_MC'].append(MC_correct_pred.item()[prob_type])
	elif 'dist3' in prob_type:
		correct_pred['dist3_gen'].append(gen_correct_pred.item()[prob_type])
		correct_pred['dist3_MC'].append(MC_correct_pred.item()[prob_type])
	elif 'prog' in prob_type:
		correct_pred['prog_gen'].append(gen_correct_pred.item()[prob_type])
		correct_pred['prog_MC'].append(MC_correct_pred.item()[prob_type])
# Convert to arrays
for key in correct_pred.keys():
	correct_pred[key] = np.concatenate(correct_pred[key])
# Calculate accuracy and confidence intervals
all_acc = {}
all_ci_lower = {}
all_ci_upper = {}
for key in correct_pred.keys():
	all_acc[key] = correct_pred[key].mean()
	all_ci_lower[key], all_ci_upper[key] = proportion_confint(correct_pred[key].sum(), correct_pred[key].shape[0])

# Save results
# Generative
all_gen_acc = np.array([all_acc['constant_gen'], all_acc['dist3_gen'], all_acc['prog_gen']])
all_gen_lower_ci = np.array([all_ci_lower['constant_gen'], all_ci_lower['dist3_gen'], all_ci_lower['prog_gen']])
all_gen_upper_ci = np.array([all_ci_upper['constant_gen'], all_ci_upper['dist3_gen'], all_ci_upper['prog_gen']])
all_gen_lower_err = all_gen_acc - all_gen_lower_ci
all_gen_upper_err =  all_gen_upper_ci - all_gen_acc
all_gen_err = np.array([all_gen_lower_err, all_gen_upper_err])
np.savez(results_dir + 'all_onerule_gen_acc.npz', acc=all_gen_acc, err=all_gen_err)
# Multiple-choice
all_MC_acc = np.array([all_acc['constant_MC'], all_acc['dist3_MC'], all_acc['prog_MC']])
all_MC_lower_ci = np.array([all_ci_lower['constant_MC'], all_ci_lower['dist3_MC'], all_ci_lower['prog_MC']])
all_MC_upper_ci = np.array([all_ci_upper['constant_MC'], all_ci_upper['dist3_MC'], all_ci_upper['prog_MC']])
all_MC_lower_err = all_MC_acc - all_MC_lower_ci
all_MC_upper_err =  all_MC_upper_ci - all_MC_acc
all_MC_err = np.array([all_MC_lower_err, all_MC_upper_err])
np.savez(results_dir + 'all_onerule_MC_acc.npz', acc=all_MC_acc, err=all_MC_err)

