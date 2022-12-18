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
context_completions = all_completions[6]
target_completions = all_completions[5]

# Load GPT-3 responses
all_gpt_context_responses = np.load('./relational_prime_results.npz', allow_pickle=True)['all_context_responses']
all_gpt_target_responses = np.load('./relational_prime_results.npz', allow_pickle=True)['all_target_responses']

# Plot settings
N_conds = 3
total_bar_width = 0.8
ind_bar_width = total_bar_width / 2
x_points = np.arange(N_conds)
gpt3_color = 'darkslateblue'
human_color = 'powderblue'
plot_fontsize = 14
title_fontsize = 16

# Context results
# Human data from Burns '96
Burns96_freq = np.array([11/74, 34/74, 29/74])
Burns96_ci_lower, Burns96_ci_upper = proportion_confint(np.array([11, 34, 29]), np.array([74, 74, 74]))
Burns96_lower_err = Burns96_freq - Burns96_ci_lower
Burns96_upper_err =  Burns96_ci_upper - Burns96_freq
Burns96_err = np.array([Burns96_lower_err, Burns96_upper_err])
# GPT-3's responses
gpt_N_completion1 = (np.array(all_gpt_context_responses) == context_completions[:,0][:len(all_gpt_context_responses)].flatten()).sum()
gpt_N_completion2 = (np.array(all_gpt_context_responses) == context_completions[:,1][:len(all_gpt_context_responses)].flatten()).sum()
gpt_N_runs = len(all_gpt_context_responses)
gpt_N_other = gpt_N_runs - gpt_N_completion1 - gpt_N_completion2
gpt_freq = np.array([gpt_N_completion1 / gpt_N_runs, gpt_N_completion2 / gpt_N_runs, gpt_N_other / gpt_N_runs])
gpt_ci_lower, gpt_ci_upper = proportion_confint(np.array([gpt_N_completion1, gpt_N_completion2, gpt_N_other]), np.array([gpt_N_runs, gpt_N_runs, gpt_N_runs]))
gpt_lower_err = gpt_freq - gpt_ci_lower
gpt_upper_err =  gpt_ci_upper - gpt_freq
gpt_err = np.array([gpt_lower_err, gpt_upper_err])
# Plot
ax = plt.subplot(111)
plt.bar(x_points - ind_bar_width / 2, gpt_freq, yerr=gpt_err, color=gpt3_color, edgecolor='black', width=ind_bar_width)
plt.bar(x_points + ind_bar_width / 2, Burns96_freq, yerr=Burns96_err, color=human_color, edgecolor='black', width=ind_bar_width)
plt.ylim([0,1])
plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','0.2','0.4','0.6','0.8','1'], fontsize=plot_fontsize)
plt.ylabel('Response frequency', fontsize=plot_fontsize)
plt.xticks(x_points, ['[m r r j j k]', '[m r r k k k]', 'Other'], fontsize=plot_fontsize)
plt.xlim([-0.5,2.5])
plt.xlabel('Response', fontsize=plot_fontsize)
plt.legend(['GPT-3','Human'],fontsize=plot_fontsize,frameon=False)
plt.title('[a b c] [a b d]\n[m r r j j j] [ ? ]', fontsize=title_fontsize)
hide_top_right(ax)
results_fname = './relational_prime_context_results.png'
ax.set_aspect(2.5)
plt.tight_layout()
plt.savefig(results_fname, dpi=300)
plt.close()

# Target results
# Human data from Burns '96'
Burns96_freq = np.array([9/74, 43/74, 22/74])
Burns96_ci_lower, Burns96_ci_upper = proportion_confint(np.array([9, 43, 22]), np.array([74, 74, 74]))
Burns96_lower_err = Burns96_freq - Burns96_ci_lower
Burns96_upper_err =  Burns96_ci_upper - Burns96_freq
Burns96_err = np.array([Burns96_lower_err, Burns96_upper_err])
# GPT-3's responses
gpt_N_completion1 = (np.array(all_gpt_target_responses) == target_completions[:,0][:len(all_gpt_target_responses)].flatten()).sum()
gpt_N_completion2 = (np.array(all_gpt_target_responses) == target_completions[:,1][:len(all_gpt_target_responses)].flatten()).sum()
gpt_N_runs = len(all_gpt_target_responses)
gpt_N_other = gpt_N_runs - gpt_N_completion1 - gpt_N_completion2
gpt_freq = np.array([gpt_N_completion1 / gpt_N_runs, gpt_N_completion2 / gpt_N_runs, gpt_N_other / gpt_N_runs])
gpt_ci_lower, gpt_ci_upper = proportion_confint(np.array([gpt_N_completion1, gpt_N_completion2, gpt_N_other]), np.array([gpt_N_runs, gpt_N_runs, gpt_N_runs]))
gpt_lower_err = gpt_freq - gpt_ci_lower
gpt_upper_err =  gpt_ci_upper - gpt_freq
gpt_err = np.array([gpt_lower_err, gpt_upper_err])
# Plot
ax = plt.subplot(111)
plt.bar(x_points - ind_bar_width / 2, gpt_freq, yerr=gpt_err, color=gpt3_color, edgecolor='black', width=ind_bar_width)
plt.bar(x_points + ind_bar_width / 2, Burns96_freq, yerr=Burns96_err, color=human_color, edgecolor='black', width=ind_bar_width)
plt.ylim([0,1])
plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','0.2','0.4','0.6','0.8','1'], fontsize=plot_fontsize)
plt.ylabel('Response frequency', fontsize=plot_fontsize)
plt.xticks(x_points, ['[k j h]\nPredecessor', '[k j j]\nSuccessor', 'Other'], fontsize=plot_fontsize)
plt.xlim([-0.5,2.5])
plt.xlabel('Response', fontsize=plot_fontsize)
plt.legend(['GPT-3','Human'],fontsize=plot_fontsize,frameon=False)
plt.title('Effect of priming successor relation', fontsize=title_fontsize)
hide_top_right(ax)
results_fname = './relational_prime_target_results.png'
ax.set_aspect(2.5)
plt.tight_layout()
plt.savefig(results_fname, dpi=300)
plt.close()
