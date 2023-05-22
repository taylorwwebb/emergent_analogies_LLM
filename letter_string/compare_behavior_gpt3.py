import numpy as np
import matplotlib.pyplot as plt
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

def plot_ind_data(ax, x_points, data, ind_bar_width):
	max_count = 24
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
			if v == 0:
				y_vals = np.ones(count) * 0.005
			else:
				y_vals = np.ones(count) * v
			plt.scatter(x_vals, y_vals, color='black', s=0.4, clip_on=False, marker='_')
	return ax

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--sentence', action='store_true', help="Present problem in sentence format.")
parser.add_argument('--noprompt', action='store_true', help="Present problem without prompt.")
args = parser.parse_args()

# Results directories
if args.sentence:
	results_dir = './human_vs_GPT3_sentence/'
	GPT3_results_dir = './GPT3_results_sentence/'
elif args.noprompt:
	results_dir = './human_vs_GPT3_noprompt/'
	GPT3_results_dir = './GPT3_results_noprompt/'
else:
	results_dir = './human_vs_GPT3/'
	GPT3_results_dir = './GPT3_results/'
check_path(results_dir)

# Plot settings
gpt3_color = 'darkslateblue'
human_color = 'powderblue'
plot_fontsize = 10
title_fontsize = 12
axis_label_fontsize = 12
bar_width = 0.8
ind_bar_width = bar_width / 2

## Zero-generalization problems, grouped by transformation type
# Load results
human_zerogen_results = np.load('./behavioral_results/zerogen_acc.npz')
human_zerogen_acc = human_zerogen_results['all_acc']
human_zerogen_err = human_zerogen_results['all_err']
GPT3_zerogen_results = np.load(GPT3_results_dir + 'zerogen_acc.npz')
GPT3_zerogen_acc = GPT3_zerogen_results['all_acc']
GPT3_zerogen_err = GPT3_zerogen_results['all_err']
# Sort based on accuracy
rank_order = np.flip(np.argsort(human_zerogen_acc))
# Plot
all_zerogen_prob_type_names = ['Successor', 'Predecessor', 'Extend\nsequence', 'Remove\nredundant\nletter', 'Fix\nalphabetic\nsequence', 'Sort']
x_points = np.arange(len(all_zerogen_prob_type_names))
ax = plt.subplot(111)
plt.bar(x_points - (ind_bar_width/2), GPT3_zerogen_acc[rank_order], yerr=GPT3_zerogen_err[:,rank_order], color=gpt3_color, edgecolor='black', width=ind_bar_width, ecolor='gray')
plt.bar(x_points + (ind_bar_width/2), human_zerogen_acc[rank_order], yerr=human_zerogen_err[:,rank_order], color=human_color, edgecolor='black', width=ind_bar_width, ecolor='gray')
plt.ylim([0,1])
plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','0.2','0.4','0.6','0.8','1'], fontsize=plot_fontsize)
plt.ylabel('Generative accuracy', fontsize=axis_label_fontsize)
plt.xticks(x_points, np.array(all_zerogen_prob_type_names)[rank_order], fontsize=plot_fontsize)
plt.xlabel('Transformation type', fontsize=axis_label_fontsize)
plt.title('Zero-generalization problems')
plt.legend(['GPT-3', 'Human'],fontsize=12,frameon=False)
hide_top_right(ax)
plt.tight_layout()
plt.savefig(results_dir + 'zerogen_acc.pdf', dpi=300, bbox_inches="tight")
plt.close()

## One-generalization problems, grouped by generalization type
# Load results
human_onegen_results = np.load('./behavioral_results/onegen_acc.npz')
human_onegen_acc = human_onegen_results['all_acc']
human_onegen_err = human_onegen_results['all_err']
GPT3_onegen_results = np.load(GPT3_results_dir + 'onegen_acc.npz')
GPT3_onegen_acc = GPT3_onegen_results['all_acc']
GPT3_onegen_err = GPT3_onegen_results['all_err']
# Sort based on accuracy
rank_order = np.flip(np.argsort(human_onegen_acc))
# Plot
all_onegen_prob_type_names = ['Larger\ninterval', 'Longer\ntarget', 'Grouping', 'Interleaved\ndistractor', 'Letter-to-\nnumber', 'Reverse\norder']
x_points = np.arange(len(all_onegen_prob_type_names))
ax = plt.subplot(111)
plt.bar(x_points - (ind_bar_width/2), GPT3_onegen_acc[rank_order], yerr=GPT3_onegen_err[:,rank_order], color=gpt3_color, edgecolor='black', width=ind_bar_width, ecolor='gray')
plt.bar(x_points + (ind_bar_width/2), human_onegen_acc[rank_order], yerr=human_onegen_err[:,rank_order], color=human_color, edgecolor='black', width=ind_bar_width, ecolor='gray')
plt.ylim([0,1])
plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','0.2','0.4','0.6','0.8','1'], fontsize=plot_fontsize)
plt.ylabel('Generative accuracy', fontsize=axis_label_fontsize)
plt.xticks(x_points, np.array(all_onegen_prob_type_names)[rank_order], fontsize=plot_fontsize)
plt.xlabel('Generalization type', fontsize=axis_label_fontsize)
plt.title('One-generalization problems')
plt.legend(['GPT-3','Human'],fontsize=plot_fontsize,frameon=False)
hide_top_right(ax)
plt.tight_layout()
plt.savefig(results_dir + 'onegen_acc.pdf', dpi=300, bbox_inches="tight")
plt.close()

## All problems, grouped by number of generalizations
# Load results
human_all_gen_results = np.load('./behavioral_results/all_gen_acc.npz')
human_all_gen_acc_ind_results = human_all_gen_results['all_ind_results']
human_all_gen_acc = human_all_gen_results['all_acc']
human_all_gen_err = human_all_gen_results['all_err']
human_all_gen_std = human_all_gen_results['all_std']
GPT3_all_gen_results = np.load(GPT3_results_dir + 'all_gen_acc.npz')
GPT3_all_gen_acc = GPT3_all_gen_results['all_acc']
GPT3_all_gen_err = GPT3_all_gen_results['all_err']
# Sort based on accuracy
rank_order = np.flip(np.argsort(human_all_gen_acc))
# Plot
all_gen_prob_type_names = np.arange(len(human_all_gen_acc)).astype(str)
x_points = np.arange(len(all_gen_prob_type_names))
ax = plt.subplot(111)
plt.bar(x_points - (ind_bar_width/2), GPT3_all_gen_acc[rank_order], yerr=GPT3_all_gen_err[:,rank_order], color=gpt3_color, edgecolor='black', width=ind_bar_width, ecolor='gray')
plt.bar(x_points + (ind_bar_width/2), human_all_gen_acc[rank_order], yerr=human_all_gen_err[rank_order], color=human_color, edgecolor='black', width=ind_bar_width)
plt.ylim([0,1])
plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','0.2','0.4','0.6','0.8','1'], fontsize=plot_fontsize)
plt.ylabel('Generative accuracy', fontsize=axis_label_fontsize)
plt.xticks(x_points, all_gen_prob_type_names, fontsize=plot_fontsize)
plt.xlabel('Number of generalizations', fontsize=axis_label_fontsize)
plt.title('  ')
plot_ind_data(ax, x_points + (ind_bar_width/2), human_all_gen_acc_ind_results, ind_bar_width)
hide_top_right(ax)
ax.set_aspect(4)
plt.tight_layout()
plt.savefig(results_dir + 'all_gen_acc.pdf', dpi=300, bbox_inches="tight")
plt.close()

## Generalization to real-world concepts
# Load results
human_realworld_results = np.load('./behavioral_results/realworld_acc.npz')
human_realworld_acc = human_realworld_results['all_acc']
human_realworld_err = human_realworld_results['all_err']
GPT3_realworld_results = np.load(GPT3_results_dir + 'realworld_acc.npz')
GPT3_realworld_acc = GPT3_realworld_results['all_acc']
GPT3_realworld_err = GPT3_realworld_results['all_err']
# Sort based on accuracy
rank_order = np.flip(np.argsort(GPT3_realworld_acc))
# Plot
all_realworld_prob_type_names = ['Successor', 'Predecessor', 'Extend\nsequence', 'Sort']
x_points = np.arange(len(all_realworld_prob_type_names))
ax = plt.subplot(111)
plt.bar(x_points - (ind_bar_width/2), GPT3_realworld_acc[rank_order], yerr=GPT3_realworld_err[:,rank_order], color=gpt3_color, edgecolor='black', width=ind_bar_width, ecolor='gray')
plt.bar(x_points + (ind_bar_width/2), human_realworld_acc[rank_order], yerr=human_realworld_err[:,rank_order], color=human_color, edgecolor='black', width=ind_bar_width, ecolor='gray')
plt.ylim([0,1])
plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','0.2','0.4','0.6','0.8','1'], fontsize=plot_fontsize)
plt.ylabel('Generative accuracy', fontsize=axis_label_fontsize)
plt.xticks(x_points, np.array(all_realworld_prob_type_names)[rank_order], fontsize=plot_fontsize)
plt.xlabel('Transformation type', fontsize=axis_label_fontsize)
plt.title('Real-world concept problems')
plt.legend(['GPT-3','Human'],fontsize=plot_fontsize,frameon=False)
hide_top_right(ax)
ax.set_aspect(4)
plt.tight_layout()
plt.savefig(results_dir + 'realworld_acc.pdf', dpi=300, bbox_inches="tight")
plt.close()
