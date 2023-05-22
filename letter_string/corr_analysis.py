import numpy as np
import matplotlib.pyplot as plt
import argparse
import scipy.stats

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--sentence', action='store_true', help="Present problem in sentence format.")
parser.add_argument('--noprompt', action='store_true', help="Present problem without prompt.")
args = parser.parse_args()

# Load data
human_data = np.load('./behavioral_results/prob_subtype_acc.npz')
human_subtype_acc = human_data['subtype_acc']
human_subtype_counts = human_data['subtype_counts']
if args.sentence:
	gpt3_data = np.load('./GPT3_results_sentence/prob_subtype_acc.npz')
elif args.noprompt:
	gpt3_data = np.load('./GPT3_results_noprompt/prob_subtype_acc.npz')
else:
	gpt3_data = np.load('./GPT3_results/prob_subtype_acc.npz')
gpt3_subtype_acc = gpt3_data['subtype_acc']
gpt3_subtype_counts = gpt3_data['subtype_counts']

# Minimum number of trials per subtype
min_trials = 5
include = np.all(np.stack([(human_subtype_counts > min_trials),(gpt3_subtype_counts > min_trials)]),0)
print('minimum number of trials per subtype = ' + str(min_trials))
print(str(include.sum()) + ' out of ' + str(len(include)) +  ' subtypes included')

# Correlation analysis
corr_results = scipy.stats.pearsonr(gpt3_subtype_acc[include], human_subtype_acc[include])
print('correlation analysis:')
print('r = ' + str(np.around(corr_results[0],4)))
print('p = ' + str(np.around(corr_results[1],10)))