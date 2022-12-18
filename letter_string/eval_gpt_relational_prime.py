import openai
import numpy as np
import builtins
import os

# GPT-3 settings
openai.api_key = "FILL_IN_API_KEY_HERE"
kwargs = { "engine":"text-davinci-003", "temperature":0, "max_tokens":20, "stop":"\n", "echo":True, "logprobs":1, }

# Load all problems
all_prob = np.load('./letter_string_analogies.npz', allow_pickle=True)['all_prob']
N_prob_type = all_prob.shape[0]

# Load data if it already exists
all_data_fname = './relational_prime_results.npz'
if os.path.exists(all_data_fname):
    all_context_responses = builtins.list(np.load('./relational_prime_results.npz', allow_pickle=True)['all_context_responses'])
    all_target_responses = builtins.list(np.load('./relational_prime_results.npz', allow_pickle=True)['all_target_responses'])
# Otherwise create data structure for responses        
else:
    all_context_responses = []
    all_target_responses = []

# Evalute GPT-3
N_trials_per_prob_type = 17
for t in range(N_trials_per_prob_type):
    print('trial ' + str(t) + '...')
    if t >= len(all_context_responses):
        # Context problem
        context_prob = all_prob[6][t]
        prompt = "Let's try to complete the pattern:\n\n" + context_prob
        response = openai.Completion.create(prompt=prompt, **kwargs)
        completed_prob = response['choices'][0]['text']
        completion = completed_prob[len(prompt):]
        all_context_responses.append(completion)
        # Target problem
        target_prob = all_prob[5][t]
        prompt = completed_prob + '\n\n' + target_prob
        response = openai.Completion.create(prompt=prompt, **kwargs)
        completed_prob = response['choices'][0]['text']
        completion = completed_prob[len(prompt):]
        all_target_responses.append(completion)
        np.savez('./relational_prime_results.npz', all_context_responses=np.array(all_context_responses), all_target_responses=np.array(all_target_responses))
    else:
        print('    already ran problem ' + str(p) + '...')