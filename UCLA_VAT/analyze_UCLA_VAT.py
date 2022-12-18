import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint

def hide_top_right(ax):
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')

# Load data
all_data = np.load('./UCLA_VAT_results.npz', allow_pickle=True)

# Calculate accuracy and confidence intervals
conditions = ['category', 'function', 'opposite', 'synonym']
N_conditions = len(conditions)
all_acc = []
all_ci_lower = []
all_ci_upper = []
for c in range(N_conditions):
	correct_pred = all_data[conditions[c]]
	acc = correct_pred.mean()
	all_acc.append(acc)
	ci_lower, ci_upper = proportion_confint(correct_pred.sum(), correct_pred.shape[0])
	all_ci_lower.append(ci_lower)
	all_ci_upper.append(ci_upper)
# Combine conditions
all_acc = np.array(all_acc)
all_ci_lower = np.array(all_ci_lower)
all_ci_upper = np.array(all_ci_upper)
all_lower_err = all_acc - all_ci_lower
all_upper_err =  all_ci_upper - all_acc
all_err = np.array([all_lower_err, all_upper_err])

# Human data
human_acc = [0.861403509, 0.814035088, 0.857894737, 0.842105263]
human_err = [0.028264813, 0.022449972, 0.020709134, 0.026953428]

# Plot
bar_width = 0.8
ind_bar_width = bar_width / 2
x_points = np.arange(N_conditions)
gpt3_color = 'darkslateblue'
human_color = 'powderblue'
plot_fontsize = 14
title_fontsize = 16
ax = plt.subplot(111)
plt.bar(x_points - (ind_bar_width/2), all_acc, yerr=all_err, color=gpt3_color, edgecolor='black', width=ind_bar_width)
plt.bar(x_points + (ind_bar_width/2), human_acc, yerr=human_err, color=human_color, edgecolor='black', width=ind_bar_width)
plt.ylim([0.5,1])
plt.yticks([0.5,0.6,0.7,0.8,0.9,1],['0.5','0.6','0.7','0.8','0.9','1'], fontsize=plot_fontsize)
plt.ylabel('Accuracy', fontsize=plot_fontsize)
plt.xticks(x_points, ['Categorical', 'Function', 'Antonym', 'Synonym'], fontsize=plot_fontsize)
hide_top_right(ax)
plt.legend(['GPT-3','Human'],fontsize=plot_fontsize,frameon=False, bbox_to_anchor=(1.2,1))
plt.tight_layout()
plt.savefig('./UCLA_VAT_results.png', dpi=300, bbox_inches="tight")
plt.close()
