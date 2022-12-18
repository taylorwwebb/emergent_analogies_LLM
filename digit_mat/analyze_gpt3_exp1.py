import numpy as np
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
all_data = np.load('./gpt_matprob_results.npz', allow_pickle=True)
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
				'logic_rule_gen': [],
				'logic_rule_MC': []}
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
	elif 'union' in prob_type or 'AND' in prob_type or 'XOR' in prob_type:
		correct_pred['logic_rule_gen'].append(gen_correct_pred.item()[prob_type])
		correct_pred['logic_rule_MC'].append(MC_correct_pred.item()[prob_type])
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
results_dir = './exp1_GPT3_data/'
check_path(results_dir)

# All problems
np.savez(results_dir + 'all_prob.npz', all_gen=correct_pred['combined_gen'], all_MC=correct_pred['combined_MC'])

# Major problem types
# Generative 
all_gen_acc = np.array([all_acc['one_rule_gen'], all_acc['two_rule_gen'], all_acc['three_rule_gen'], all_acc['logic_rule_gen']])
all_gen_lower_ci = np.array([all_ci_lower['one_rule_gen'], all_ci_lower['two_rule_gen'], all_ci_lower['three_rule_gen'], all_ci_lower['logic_rule_gen']])
all_gen_upper_ci = np.array([all_ci_upper['one_rule_gen'], all_ci_upper['two_rule_gen'], all_ci_upper['three_rule_gen'], all_ci_upper['logic_rule_gen']])
all_gen_lower_err = all_gen_acc - all_gen_lower_ci
all_gen_upper_err =  all_gen_upper_ci - all_gen_acc
all_gen_err = np.array([all_gen_lower_err, all_gen_upper_err])
np.savez(results_dir + 'all_probcat_gen_acc.npz', acc=all_gen_acc, err=all_gen_err)
# Multiple-choice 
all_MC_acc = np.array([all_acc['one_rule_MC'], all_acc['two_rule_MC'], all_acc['three_rule_MC'], all_acc['logic_rule_MC']])
all_MC_lower_ci = np.array([all_ci_lower['one_rule_MC'], all_ci_lower['two_rule_MC'], all_ci_lower['three_rule_MC'], all_ci_lower['logic_rule_MC']])
all_MC_upper_ci = np.array([all_ci_upper['one_rule_MC'], all_ci_upper['two_rule_MC'], all_ci_upper['three_rule_MC'], all_ci_upper['logic_rule_MC']])
all_MC_lower_err = all_MC_acc - all_MC_lower_ci
all_MC_upper_err =  all_MC_upper_ci - all_MC_acc
all_MC_err = np.array([all_MC_lower_err, all_MC_upper_err])
np.savez(results_dir + 'all_probcat_MC_acc.npz', acc=all_MC_acc, err=all_MC_err)

## Relational complexity analysis (controlling for number of unique rules)
N_unique_rules_2rule_prob = np.load('./N_unique_rules_2rule_prob.npz')['N_unique_rules']
N_unique_rules_3rule_prob = np.load('./N_unique_rules_3rule_prob.npz')['N_unique_rules']
correct_pred = {'gen_2rule_1unique': [],
				'MC_2rule_1unique': [],
				'gen_2rule_2unique': [],
				'MC_2rule_2unique': [],
				'gen_3rule_1unique': [],
				'MC_3rule_1unique': [],
				'gen_3rule_2unique': [],
				'MC_3rule_2unique': [],
				'gen_3rule_3unique': [],
				'MC_3rule_3unique': []}
for prob_type in all_prob_types:
	if 'two_rule' in prob_type:
		if N_unique_rules_2rule_prob[int(prob_type[-1])] == 1:
			correct_pred['gen_2rule_1unique'].append(gen_correct_pred.item()[prob_type])
			correct_pred['MC_2rule_1unique'].append(MC_correct_pred.item()[prob_type])
		elif N_unique_rules_2rule_prob[int(prob_type[-1])] == 2:
			correct_pred['gen_2rule_2unique'].append(gen_correct_pred.item()[prob_type])
			correct_pred['MC_2rule_2unique'].append(MC_correct_pred.item()[prob_type])
	elif 'three_rule' in prob_type:
		if N_unique_rules_3rule_prob[int(prob_type[-1])] == 1:
			correct_pred['gen_3rule_1unique'].append(gen_correct_pred.item()[prob_type])
			correct_pred['MC_3rule_1unique'].append(MC_correct_pred.item()[prob_type])
		elif N_unique_rules_3rule_prob[int(prob_type[-1])] == 2:
			correct_pred['gen_3rule_2unique'].append(gen_correct_pred.item()[prob_type])
			correct_pred['MC_3rule_2unique'].append(MC_correct_pred.item()[prob_type])
		elif N_unique_rules_3rule_prob[int(prob_type[-1])] == 3:
			correct_pred['gen_3rule_3unique'].append(gen_correct_pred.item()[prob_type])
			correct_pred['MC_3rule_3unique'].append(MC_correct_pred.item()[prob_type])
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
# Two-rule problems
# Generative
all_gen_acc = np.array([all_acc['gen_2rule_1unique'], all_acc['gen_2rule_2unique']])
all_gen_ci_lower = np.array([all_ci_lower['gen_2rule_1unique'], all_ci_lower['gen_2rule_2unique']])
all_gen_ci_upper = np.array([all_ci_upper['gen_2rule_1unique'], all_ci_upper['gen_2rule_2unique']])
all_gen_lower_err = all_gen_acc - all_gen_ci_lower
all_gen_upper_err = all_gen_ci_upper - all_gen_acc
all_gen_err = np.array([all_gen_lower_err, all_gen_upper_err])
np.savez(results_dir + 'tworule_prob_N_unique_rules_gen.npz', acc=all_gen_acc, err=all_gen_err)
# Multiple-choice
all_MC_acc = np.array([all_acc['MC_2rule_1unique'], all_acc['MC_2rule_2unique']])
all_MC_ci_lower = np.array([all_ci_lower['MC_2rule_1unique'], all_ci_lower['MC_2rule_2unique']])
all_MC_ci_upper = np.array([all_ci_upper['MC_2rule_1unique'], all_ci_upper['MC_2rule_2unique']])
all_MC_lower_err = all_MC_acc - all_MC_ci_lower
all_MC_upper_err = all_MC_ci_upper - all_MC_acc
all_MC_err = np.array([all_MC_lower_err, all_MC_upper_err])
np.savez(results_dir + 'tworule_prob_N_unique_rules_MC.npz', acc=all_MC_acc, err=all_MC_err)
# Three-rule problems
# Generative
all_gen_acc = np.array([all_acc['gen_3rule_1unique'], all_acc['gen_3rule_2unique'], all_acc['gen_3rule_3unique']])
all_gen_ci_lower = np.array([all_ci_lower['gen_3rule_1unique'], all_ci_lower['gen_3rule_2unique'], all_ci_lower['gen_3rule_3unique']])
all_gen_ci_upper = np.array([all_ci_upper['gen_3rule_1unique'], all_ci_upper['gen_3rule_2unique'], all_ci_upper['gen_3rule_3unique']])
all_gen_lower_err = all_gen_acc - all_gen_ci_lower
all_gen_upper_err = all_gen_ci_upper - all_gen_acc
all_gen_err = np.array([all_gen_lower_err, all_gen_upper_err])
np.savez(results_dir + 'threerule_prob_N_unique_rules_gen.npz', acc=all_gen_acc, err=all_gen_err)
# Multiple-choice
all_MC_acc = np.array([all_acc['MC_3rule_1unique'], all_acc['MC_3rule_2unique'], all_acc['MC_3rule_3unique']])
all_MC_ci_lower = np.array([all_ci_lower['MC_3rule_1unique'], all_ci_lower['MC_3rule_2unique'], all_ci_lower['MC_3rule_3unique']])
all_MC_ci_upper = np.array([all_ci_upper['MC_3rule_1unique'], all_ci_upper['MC_3rule_2unique'], all_ci_upper['MC_3rule_3unique']])
all_MC_lower_err = all_MC_acc - all_MC_ci_lower
all_MC_upper_err = all_MC_ci_upper - all_MC_acc
all_MC_err = np.array([all_MC_lower_err, all_MC_upper_err])
np.savez(results_dir + 'threerule_prob_N_unique_rules_MC.npz', acc=all_MC_acc, err=all_MC_err)

## Compare problems with vs. without progression rule (two-rule problems)
correct_pred = {'tworule_prog_gen': [],
				'tworule_noprog_gen': []}
for prob_type in all_prob_types:
	if prob_type == 'two_rule_comb2' or prob_type == 'two_rule_comb4' or prob_type == 'two_rule_comb5':
		correct_pred['tworule_prog_gen'].append(gen_correct_pred.item()[prob_type])
	elif prob_type == 'two_rule_comb0' or prob_type == 'two_rule_comb1' or prob_type == 'two_rule_comb3':
		correct_pred['tworule_noprog_gen'].append(gen_correct_pred.item()[prob_type])
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
all_gen_acc = np.array([all_acc['tworule_noprog_gen'], all_acc['tworule_prog_gen']])
all_gen_lower_ci = np.array([all_ci_lower['tworule_noprog_gen'], all_ci_lower['tworule_prog_gen']])
all_gen_upper_ci = np.array([all_ci_upper['tworule_noprog_gen'], all_ci_upper['tworule_prog_gen']])
all_gen_lower_err = all_gen_acc - all_gen_lower_ci
all_gen_upper_err =  all_gen_upper_ci - all_gen_acc
all_gen_err = np.array([all_gen_lower_err, all_gen_upper_err])
np.savez(results_dir + 'tworule_prog_vs_noprog_gen_acc.npz', acc=all_gen_acc, err=all_gen_err)

# Three major one-rule problem types
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

# Permuted vs. non-permuted logic problems
correct_pred = {'aligned_gen': [],
				'aligned_MC': [],
				'permuted_gen': [],
				'permuted_MC': []}
for prob_type in all_prob_types:
	if 'union' in prob_type or 'AND' in prob_type or 'XOR' in prob_type:
		if 'permuted' in prob_type:
			correct_pred['permuted_gen'].append(gen_correct_pred.item()[prob_type])
			correct_pred['permuted_MC'].append(MC_correct_pred.item()[prob_type])
		else:
			correct_pred['aligned_gen'].append(gen_correct_pred.item()[prob_type])
			correct_pred['aligned_MC'].append(MC_correct_pred.item()[prob_type])
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
all_gen_acc = np.array([all_acc['aligned_gen'], all_acc['permuted_gen']])
all_gen_lower_ci = np.array([all_ci_lower['aligned_gen'], all_ci_lower['permuted_gen']])
all_gen_upper_ci = np.array([all_ci_upper['aligned_gen'], all_ci_upper['permuted_gen']])
all_gen_lower_err = all_gen_acc - all_gen_lower_ci
all_gen_upper_err =  all_gen_upper_ci - all_gen_acc
all_gen_err = np.array([all_gen_lower_err, all_gen_upper_err])
np.savez(results_dir + 'aligned_vs_permuted_gen_acc.npz', acc=all_gen_acc, err=all_gen_err)
# Multiple-choice
all_MC_acc = np.array([all_acc['aligned_MC'], all_acc['permuted_MC']])
all_MC_lower_ci = np.array([all_ci_lower['aligned_MC'], all_ci_lower['permuted_MC']])
all_MC_upper_ci = np.array([all_ci_upper['aligned_MC'], all_ci_upper['permuted_MC']])
all_MC_lower_err = all_MC_acc - all_MC_lower_ci
all_MC_upper_err =  all_MC_upper_ci - all_MC_acc
all_MC_err = np.array([all_MC_lower_err, all_MC_upper_err])
np.savez(results_dir + 'aligned_vs_permuted_MC_acc.npz', acc=all_MC_acc, err=all_MC_err)



