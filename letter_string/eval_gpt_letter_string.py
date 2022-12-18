import openai
import numpy as np
import builtins
import os

# GPT-3 settings
openai.api_key = "FILL_IN_API_KEY_HERE"
kwargs = { "engine":"text-davinci-003", "temperature":0, "max_tokens":20, "stop":"\n", "echo":True, "logprobs":1, }

# Load all problems
all_prob = np.load('./letter_string_analogies.npz', allow_pickle=True)['all_prob']
N_prob_type = all_prob.shape[0] - 1

# Load data if it already exists
all_data_fname = './letter_string_results.npz'
if os.path.exists(all_data_fname):
    previous_data = np.load('./letter_string_results.npz', allow_pickle=True)['all_responses']
    all_responses = []
    for p in range(N_prob_type):
        all_responses.append(builtins.list(previous_data[p]))
# Otherwise create data structure for responses        
else:
    all_responses = []
    for p in range(N_prob_type):
        all_responses.append([])

# Evalute GPT-3
N_trials_per_prob_type = 18
for t in range(N_trials_per_prob_type):
    print('trial ' + str(t) + '...')
    for p in range(N_prob_type):
        if t >= len(all_responses[p]):
            if t < all_prob[p].shape[0]:
                print('    problem ' + str(p) + '...')
                prob = all_prob[p][t]
                prompt = "Let's try to complete the pattern:\n\n" + prob
                response = openai.Completion.create(prompt=prompt, **kwargs)
                completed_prob = response['choices'][0]['text']
                completion = completed_prob[len(prompt):]
                all_responses[p].append(completion)
                np.savez('./letter_string_results.npz', all_responses=np.array(all_responses, dtype=object))
            else:
                print('    no more problems left for problem type ' + str(p) + '...')
        else:
            print('    already ran problem ' + str(p) + '...')