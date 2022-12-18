import numpy as np
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt

def hide_top_right(ax):
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')

# Load all problems
all_prob = np.load('./letter_string_analogies.npz', allow_pickle=True)['all_prob']
all_completions = np.load('./letter_string_analogies.npz', allow_pickle=True)['all_prob_completion']
N_prob_type = all_prob.shape[0]

# Load GPT-3 responses
all_gpt_responses = np.load('./letter_string_results.npz', allow_pickle=True)['all_responses']

# Calculate accuracy
N_prob_type = 5
all_acc = []
all_err = []
for p in range(N_prob_type):
	correct_pred = np.array(all_gpt_responses[p]) == all_completions[p][:len(all_gpt_responses[p])].flatten()
	acc = correct_pred.astype(float).mean()
	all_acc.append(acc)
	ci_lower, ci_upper = proportion_confint(correct_pred.sum(), correct_pred.shape[0])
	lower_err = acc - ci_lower
	upper_err =  ci_upper - acc
	err = np.array([lower_err, upper_err])
	all_err.append(err)

# Plot settings
total_bar_width = 0.8
x_points = np.arange(N_prob_type)
gpt3_color = 'darkslateblue'
human_color = 'powderblue'
plot_fontsize = 14
title_fontsize = 16

# Plot
all_err = np.stack(all_err,1)
ax = plt.subplot(111)
plt.bar(x_points, all_acc, yerr=all_err, color=gpt3_color, edgecolor='black', width=total_bar_width)
plt.ylim([0,1])
plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','0.2','0.4','0.6','0.8','1'], fontsize=plot_fontsize)
plt.ylabel('Generative accuracy', fontsize=plot_fontsize)
plt.xticks(x_points, ['Basic\nsuccessor\nrelation', 'Removing\nredundant\ncharacters', 'Letter to\nnumber', 'Groupings', 'Longer\ntarget'], fontsize=10)
plt.xlabel('Problem type', fontsize=plot_fontsize)
hide_top_right(ax)
results_fname = './letter_string_acc.png'
plt.title('Zero-shot letter string analogies', fontsize=title_fontsize)
plt.tight_layout()
plt.savefig(results_fname, dpi=300)
plt.close()

# Mapping successor to predecessor
# Human data from Burns '96
Burns96_freq = np.array([24/66, 14/66, 28/66])
Burns96_ci_lower, Burns96_ci_upper = proportion_confint(np.array([24, 14, 28]), np.array([66, 66, 66]))
Burns96_lower_err = Burns96_freq - Burns96_ci_lower
Burns96_upper_err =  Burns96_ci_upper - Burns96_freq
Burns96_err = np.array([Burns96_lower_err, Burns96_upper_err])
# GPT-3's responses
gpt_N_completion1 = (np.array(all_gpt_responses[5]) == all_completions[5][:,0][:len(all_gpt_responses[5])].flatten()).sum()
gpt_N_completion2 = (np.array(all_gpt_responses[5]) == all_completions[5][:,1][:len(all_gpt_responses[5])].flatten()).sum()
gpt_N_runs = len(all_gpt_responses[5])
gpt_N_other = gpt_N_runs - gpt_N_completion1 - gpt_N_completion2
gpt_freq = np.array([gpt_N_completion1 / gpt_N_runs, gpt_N_completion2 / gpt_N_runs, gpt_N_other / gpt_N_runs])
gpt_ci_lower, gpt_ci_upper = proportion_confint(np.array([gpt_N_completion1, gpt_N_completion2, gpt_N_other]), np.array([gpt_N_runs, gpt_N_runs, gpt_N_runs]))
gpt_lower_err = gpt_freq - gpt_ci_lower
gpt_upper_err =  gpt_ci_upper - gpt_freq
gpt_err = np.array([gpt_lower_err, gpt_upper_err])
# Plot
ind_bar_width = total_bar_width/2
x_points = np.arange(3)
ax = plt.subplot(111)
plt.bar(x_points - 0.2, gpt_freq, yerr=gpt_err, color=gpt3_color, edgecolor='black', width=ind_bar_width)
plt.bar(x_points + 0.2, Burns96_freq, yerr=Burns96_err, color=human_color, edgecolor='black', width=ind_bar_width)
plt.ylim([0,1])
plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','0.2','0.4','0.6','0.8','1'], fontsize=plot_fontsize)
plt.xlim([-0.5,2.5])
plt.ylabel('Response frequency', fontsize=plot_fontsize)
plt.xticks(x_points, ['[k j h]\nPredecessor', '[k j j]\nSuccessor', 'Other'], fontsize=plot_fontsize)
plt.xlabel('Response', fontsize=plot_fontsize)
plt.legend(['GPT-3','Human'],fontsize=plot_fontsize,frameon=False)
plt.title('Zero-shot successor-to-predecessor', fontsize=title_fontsize)
hide_top_right(ax)
results_fname = './zeroshot_succ_to_pred.png'
ax.set_aspect(2.5)
plt.tight_layout()
plt.savefig(results_fname, dpi=300)
plt.close()

