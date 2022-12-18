import numpy as np
import scipy.stats

# Load human data
human_prob_acc = np.load('./exp1_behavioral_data/ind_subj_results.npz')['all_subj_gen_correct_pred'].mean(0)
gpt3_prob_acc = np.load('./exp1_GPT3_data/all_prob.npz')['all_gen'].reshape((-1,32)).astype(float).mean(0)

# Correlation analysis
corr_results = scipy.stats.pearsonr(gpt3_prob_acc, human_prob_acc)
print('correlation analysis:')
print('r = ' + str(np.around(corr_results[0],4)))
print('p = ' + str(np.around(corr_results[1],4)))