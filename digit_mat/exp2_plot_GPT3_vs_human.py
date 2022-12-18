import numpy as np
import matplotlib.pyplot as plt
import os

def check_path(path):
	if not os.path.exists(path):
		os.mkdir(path)

def hide_top_right(ax):
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')

# Digit matrices
# 1-rule, 2-rule, 3-rule, 4-rule, and 5-rule problems
# GPT-3: progressive
# Humans: progressive

# Load GPT-3 data
gpt3_gen_acc = np.load('./exp2_GPT3_data/all_probcat_gen_acc.npz')
gpt3_MC_acc = np.load('./exp2_GPT3_data/all_probcat_MC_acc.npz')
gpt3_gen_acc_mn = gpt3_gen_acc['acc']
gpt3_gen_acc_err = gpt3_gen_acc['err']
gpt3_MC_acc_mn = gpt3_MC_acc['acc']
gpt3_MC_acc_err = gpt3_MC_acc['err']

# Load human data
human_gen_acc = np.load('./exp2_behavioral_data/probcat_gen_acc_behavior.npz')
human_MC_acc = np.load('./exp2_behavioral_data/probcat_MC_acc_behavior.npz')
human_gen_acc_mn = human_gen_acc['acc']
human_gen_acc_err = human_gen_acc['err']
human_MC_acc_mn = human_MC_acc['acc']
human_MC_acc_err = human_MC_acc['err']

# Plot settings
N_conds = gpt3_MC_acc_mn.shape[0]
total_bar_width = 0.8
x_points = np.arange(N_conds)
gpt3_color = 'darkslateblue'
human_color = 'powderblue'
plot_fontsize = 14
title_fontsize = 16

# Directory
plot_dir = './exp2_GPT3_vs_human/'
check_path(plot_dir)

# Plot - generative 
ind_bar_width = total_bar_width / 2
ax = plt.subplot(111)
plt.bar(x_points - 0.2, gpt3_gen_acc_mn, yerr=gpt3_gen_acc_err, color=gpt3_color, edgecolor='black', width=ind_bar_width)
plt.bar(x_points + 0.2, human_gen_acc_mn, yerr=human_gen_acc_err, color=human_color, edgecolor='black', width=ind_bar_width)
plt.ylim([0,1])
plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','0.2','0.4','0.6','0.8','1'], fontsize=plot_fontsize)
plt.ylabel('Generative accuracy', fontsize=plot_fontsize)
plt.xticks(x_points, ['1-rule', '2-rule', '3-rule', '4-rule', '5-rule'], fontsize=plot_fontsize)
plt.xlabel('Problem type', fontsize=plot_fontsize)
hide_top_right(ax)
plt.legend(['GPT-3','Human'],fontsize=plot_fontsize,frameon=False,bbox_to_anchor=(0.95,1))
results_fname = plot_dir + 'gen_gpt3_vs_human.png'
ax.set_aspect(3)
plt.tight_layout()
plt.savefig(results_fname, dpi=300, bbox_inches="tight")
plt.close()

# Plot - multiple-choice 
ax = plt.subplot(111)
plt.bar(x_points - 0.2, gpt3_MC_acc_mn, yerr=gpt3_MC_acc_err, color=gpt3_color, edgecolor='black', width=ind_bar_width)
plt.bar(x_points + 0.2, human_MC_acc_mn, yerr=human_MC_acc_err, color=human_color, edgecolor='black', width=ind_bar_width)
plt.ylim([0,1])
plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','0.2','0.4','0.6','0.8','1'], fontsize=plot_fontsize)
plt.ylabel('Multiple choice accuracy', fontsize=plot_fontsize)
plt.xticks(x_points, ['1-rule', '2-rule', '3-rule', '4-rule', '5-rule'], fontsize=plot_fontsize)
plt.xlabel('Problem type', fontsize=plot_fontsize)
hide_top_right(ax)
plt.legend(['GPT-3','Human'],fontsize=plot_fontsize,frameon=False,bbox_to_anchor=(0.95,1))
results_fname = plot_dir + 'MC_gpt3_vs_human.png'
ax.set_aspect(3)
plt.tight_layout()
plt.savefig(results_fname, dpi=300, bbox_inches="tight")
plt.close()

## Rule type analysis for one-rule problems

# Load GPT-3 one-rule data
gpt3_gen_acc_onerule = np.load('./exp2_GPT3_data/all_onerule_gen_acc.npz')
gpt3_gen_acc_onerule_mn = gpt3_gen_acc_onerule['acc']
gpt3_gen_acc_onerule_err = gpt3_gen_acc_onerule['err']

# Load human one-rule data
human_gen_acc_onerule = np.load('./exp2_behavioral_data/probcat_gen_acc_behavior_onerule.npz')
human_gen_acc_onerule_mn = human_gen_acc_onerule['acc']
human_gen_acc_onerule_err = human_gen_acc_onerule['err']

# Plot settings
N_conds = gpt3_gen_acc_onerule_mn.shape[0]
x_points = np.arange(N_conds)
ind_bar_width = total_bar_width / 2

# Plot - generative
ax = plt.subplot(111)
plt.bar(x_points - 0.2, gpt3_gen_acc_onerule_mn, yerr=gpt3_gen_acc_onerule_err, color=gpt3_color, edgecolor='black', width=ind_bar_width)
plt.bar(x_points + 0.2, human_gen_acc_onerule_mn, yerr=human_gen_acc_onerule_err, color=human_color, edgecolor='black', width=ind_bar_width)
plt.ylim([0,1])
plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','0.2','0.4','0.6','0.8','1'], fontsize=plot_fontsize)
plt.ylabel('Generative accuracy', fontsize=plot_fontsize)
plt.xticks(x_points, ['Constant', 'Distribution', 'Progression'], fontsize=plot_fontsize)
plt.xlabel('Rule type', fontsize=plot_fontsize)
hide_top_right(ax)
plt.legend(['GPT-3','Human'],fontsize=plot_fontsize,frameon=False,bbox_to_anchor=(0.65,1))
plt.title('One-rule problems', fontsize=title_fontsize)
results_fname = plot_dir + 'onerule_gen_gpt3_vs_human.png'
ax.set_aspect(3)
plt.tight_layout()
plt.savefig(results_fname, dpi=300, bbox_inches="tight")
plt.close()


