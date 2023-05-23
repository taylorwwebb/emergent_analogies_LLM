import openai
import numpy as np
import pandas as pd
import builtins
import time

# GPT-3 settings
openai.api_key = "FILL_IN_API_KEY_HERE"
kwargs = { "engine":"text-davinci-003", "temperature":0, "max_tokens":10, "stop":"\n", "echo":True, "logprobs":1, }

# Load problems
df = pd.read_excel (r'./UCLA_VAT.xlsx', sheet_name='UCLA_VAT')
# Extract data
A = builtins.list(df['A'])
B = builtins.list(df['B'])
C = builtins.list(df['C'])
D = builtins.list(df['D'])
D_prime = builtins.list(df["D'"])

# Initialize storage for results
all_synonym_correct_pred = []
all_opposite_correct_pred = []
all_function_correct_pred = []
all_category_correct_pred = []
results_fname = './UCLA_VAT_results.npz'
# Evaluate 
N_prob = len(A)
prob_order = np.arange(N_prob)
np.random.shuffle(prob_order)
for p in range(N_prob):
	print(str(p+1) + ' of ' + str(N_prob) + '...')
	if p == 0:
		prompt = A[prob_order[p]] + ' : ' + B[prob_order[p]] + ' :: ' + C[prob_order[p]] + ' : '
	else:
		prompt = context + '\n\n' + A[prob_order[p]] + ' : ' + B[prob_order[p]] + ' :: ' + C[prob_order[p]] + ' : '
	# Correct answer
	d_prompt = prompt + D[prob_order[p]]
	response = []
	while len(response) == 0:
		try:
			response = openai.Completion.create(prompt=d_prompt, **kwargs)
		except:
			print('trying again...')
			time.sleep(5)
	first_token_ind = np.where(np.array(response['choices'][0]['logprobs']['text_offset']) <= len(prompt))[0][-1]
	if len(d_prompt) > np.array(response['choices'][0]['logprobs']['text_offset'])[0]:
		d_avg_logprob = np.mean(response['choices'][0]['logprobs']['token_logprobs'][first_token_ind:])
	else:
		last_token_ind = np.where(np.array(response['choices'][0]['logprobs']['text_offset']) == len(d_prompt))[0][0]
		d_avg_logprob = np.mean(response['choices'][0]['logprobs']['token_logprobs'][first_token_ind:last_token_ind])
	# Foil
	d_prime_prompt = prompt + D_prime[prob_order[p]]
	response = []
	while len(response) == 0:
		try:
			response = openai.Completion.create(prompt=d_prime_prompt, **kwargs)
		except:
			print('trying again...')
			time.sleep(5)
	first_token_ind = np.where(np.array(response['choices'][0]['logprobs']['text_offset']) <= len(prompt))[0][-1]
	if len(d_prime_prompt) > np.array(response['choices'][0]['logprobs']['text_offset'])[0]:
		d_prime_avg_logprob = np.mean(response['choices'][0]['logprobs']['token_logprobs'][first_token_ind:])
	else:
		last_token_ind = np.where(np.array(response['choices'][0]['logprobs']['text_offset']) == len(d_prime_prompt))[0][0]
		d_prime_avg_logprob = np.mean(response['choices'][0]['logprobs']['token_logprobs'][first_token_ind:last_token_ind])
	# Correct
	correct_pred = d_avg_logprob > d_prime_avg_logprob
	if prob_order[p] < 20:
		all_synonym_correct_pred.append(correct_pred)
	elif prob_order[p] >= 20 and prob_order[p] < 40:
		all_opposite_correct_pred.append(correct_pred)
	elif prob_order[p] >= 40 and prob_order[p] < 60:
		all_function_correct_pred.append(correct_pred)
	elif prob_order[p] >= 60:
		all_category_correct_pred.append(correct_pred)
	# Add problem to context
	if correct_pred:
		context = d_prompt
	else:
		context = d_prime_prompt
	# Save results
	np.savez(results_fname, synonym=all_synonym_correct_pred, opposite=all_opposite_correct_pred, function=all_function_correct_pred, category=all_category_correct_pred, context=context, prob_order=prob_order, allow_pickle=True)
