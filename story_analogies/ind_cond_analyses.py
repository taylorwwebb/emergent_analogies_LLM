import csv
import numpy as np
from scipy.stats import binomtest, ttest_1samp

# Human vs. GPT-3 data
human_data_file = open('./human_vs_gpt3_data.csv')
csvreader = csv.reader(human_data_file)
header = []
header = np.array(next(csvreader))
rows = []
for row in csvreader:
	rows.append(row)
rows = np.array(rows).astype(int)

# Human data analyses
human_vs_gpt = np.where(header == 'human_vs_gpt')[0][0]
human_data = rows[rows[:,human_vs_gpt]==0, :]
human_acc = []
subjID = np.where(header == 'subjID')[0][0]
N_subj = np.max(np.unique(human_data[:,subjID]))
for s in range(N_subj):
	subj_data = human_data[human_data[:,subjID] == s+1, :]
	correct_pred = np.where(header == 'correct_pred')[0][0]
	subj_acc = np.mean(subj_data[:,correct_pred])
	human_acc.append(subj_acc)
print('\nHuman:')
print('Combined:')
print(ttest_1samp(human_acc, 0.5))
# Near analogy
analogy_vs_similarity = np.where(header == 'analogy_vs_similarity')[0][0]
human_similarity_data = human_data[human_data[:,analogy_vs_similarity]==1, :]
human_similarity_acc = []
for s in range(N_subj):
	subj_data = human_similarity_data[human_similarity_data[:,subjID] == s+1, :]
	subj_acc = np.mean(subj_data[:,correct_pred])
	human_similarity_acc.append(subj_acc)
print('Near analogy:')
print(ttest_1samp(human_similarity_acc, 0.5))
# Far analogy
human_analogy_data = human_data[human_data[:,analogy_vs_similarity]==0, :]
human_analogy_acc = []
for s in range(N_subj):
	subj_data = human_analogy_data[human_analogy_data[:,subjID] == s+1, :]
	subj_acc = np.mean(subj_data[:,correct_pred])
	human_analogy_acc.append(subj_acc)
print('Far analogy:')
print(ttest_1samp(human_analogy_acc, 0.5))

# GPT-3 data analyses
gpt3_data = rows[rows[:,human_vs_gpt]==1, :]
print('\nGPT-3:')
print('Combined:')
print(binomtest(gpt3_data[:,correct_pred].sum(), n=gpt3_data.shape[0], p=0.5))
# Near analogy
gpt3_similarity_data = gpt3_data[gpt3_data[:,analogy_vs_similarity]==1, :]
print('Near analogy:')
print(binomtest(gpt3_similarity_data[:,correct_pred].sum(), n=gpt3_similarity_data.shape[0], p=0.5))
# Far analogy
gpt3_analogy_data = gpt3_data[gpt3_data[:,analogy_vs_similarity]==0, :]
print('Far analogy:')
print(binomtest(gpt3_analogy_data[:,correct_pred].sum(), n=gpt3_analogy_data.shape[0], p=0.5))

# GPT-4 data
human_data_file = open('./gpt4_data.csv')
csvreader = csv.reader(human_data_file)
header = []
header = np.array(next(csvreader))
rows = []
for row in csvreader:
	rows.append(row)
rows = np.array(rows).astype(int)

# GPT-4 data analyses
gpt4_data = rows
correct_pred = np.where(header == 'correct_pred')[0][0]
print('\nGPT-4:')
print('Combined:')
print(binomtest(gpt4_data[:,correct_pred].sum(), n=gpt4_data.shape[0], p=0.5))
# Near analogy
analogy_vs_similarity = np.where(header == 'analogy_vs_similarity')[0][0]
gpt4_similarity_data = gpt4_data[gpt4_data[:,analogy_vs_similarity]==1, :]
print('Near analogy:')
print(binomtest(gpt4_similarity_data[:,correct_pred].sum(), n=gpt4_similarity_data.shape[0], p=0.5))
# Far analogy
gpt4_analogy_data = gpt4_data[gpt4_data[:,analogy_vs_similarity]==0, :]
print('Far analogy:')
print(binomtest(gpt4_analogy_data[:,correct_pred].sum(), n=gpt4_analogy_data.shape[0], p=0.5))
print(' ')

