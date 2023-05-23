import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import sem
import pandas as pd
import builtins

def hide_top_right(ax):
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	
def plot_ind_data(ax, x_points, data, ind_bar_width):
	max_count = 30
	point_unit = ind_bar_width / max_count
	# Plot
	for i in range(len(x_points)):
		unique_vals = np.unique(data[i])
		for v in unique_vals:
			count = (data[i]==v).sum()
			span = count * point_unit
			x_min = x_points[i] - (span/2)
			x_max = x_points[i] + (span/2)
			x_vals = np.linspace(x_min,x_max,count)
			if count == 1:
				x_vals = np.mean([x_min,x_max])
			if v == 0:
				y_vals = np.ones(count) * 0.005
			else:
				y_vals = np.ones(count) * v
			plt.scatter(x_vals, y_vals, color='black', s=0.4, clip_on=False, marker='_')
	return ax

# Load data
all_data = np.load('./UCLA_VAT_results.npz', allow_pickle=True)

# Calculate accuracy and confidence intervals
conditions = ['category', 'function', 'opposite', 'synonym']
N_conditions = len(conditions)
all_acc = []
all_correct_pred = []
all_ci_lower = []
all_ci_upper = []
for c in range(N_conditions):
	correct_pred = all_data[conditions[c]]
	all_correct_pred.append(correct_pred)
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
# Overall accuracy
all_correct_pred = np.concatenate(all_correct_pred)
overall_gpt3_acc = all_correct_pred.astype(float).mean()
overall_gpt3_ci_lower, overall_gpt3_ci_upper = proportion_confint(all_correct_pred.sum(), all_correct_pred.shape[0])
overall_gpt3_err = [overall_gpt3_acc - overall_gpt3_ci_lower, overall_gpt3_ci_upper - overall_gpt3_acc]

# Human data
df = pd.read_excel (r'./UCLA_VAT_ind_subj_data.xlsx', sheet_name='ind_subj')
category_ind_subj_acc = np.array(builtins.list(df['category'])[1:]) / 100.
function_ind_subj_acc = np.array(builtins.list(df['function'])[1:]) / 100.
opposite_ind_subj_acc = np.array(builtins.list(df['opposite'])[1:]) / 100.
synonym_ind_subj_acc = np.array(builtins.list(df['synonym'])[1:]) / 100.
human_ind_subj = np.array([category_ind_subj_acc, function_ind_subj_acc, opposite_ind_subj_acc, synonym_ind_subj_acc])
human_acc = human_ind_subj.mean(1)
human_err = sem(human_ind_subj,1)

# Plot
bar_width = 0.8
ind_bar_width = bar_width / 2
x_points = np.arange(N_conditions)
gpt3_color = 'darkslateblue'
human_color = 'powderblue'
plot_fontsize = 14
title_fontsize = 16
ax = plt.subplot(111)
plt.bar(x_points - (ind_bar_width/2), all_acc, yerr=all_err, color=gpt3_color, edgecolor='black', width=ind_bar_width, ecolor='gray')
plt.bar(x_points + (ind_bar_width/2), human_acc, yerr=human_err, color=human_color, edgecolor='black', width=ind_bar_width)
plt.ylim([0,1])
plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','0.2','0.4','0.6','0.8','1'], fontsize=plot_fontsize)
plt.ylabel('Accuracy', fontsize=plot_fontsize)
plt.xticks(x_points, ['Categorical', 'Function', 'Antonym', 'Synonym'], fontsize=12)
hide_top_right(ax)
plt.legend(['GPT-3','Human'],fontsize=plot_fontsize,frameon=False, bbox_to_anchor=(0.9,1))
plot_ind_data(ax, x_points + (ind_bar_width/2), human_ind_subj, ind_bar_width)
current_xlim = ax.get_xlim()
plt.plot([current_xlim[0], current_xlim[1]],[0.5,0.5],color='black',alpha=0.4)
plt.xlim([current_xlim[0], current_xlim[1]])
plt.title('UCLA VAT', fontsize=title_fontsize, pad=20)
ax.set_aspect(4)
plt.tight_layout()
plt.savefig('./UCLA_VAT_results.pdf', dpi=300, bbox_inches="tight")
plt.close()
