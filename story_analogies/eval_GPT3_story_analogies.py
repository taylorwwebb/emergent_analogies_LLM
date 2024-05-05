import openai
import numpy as np
import builtins
import time

# GPT-3 settings
openai.api_key = "FILL_IN_API_KEY_HERE"
kwargs = { "engine":"text-davinci-003", "temperature":0, "max_tokens":256, "echo":False, "logprobs":1, }

# Load problems
df = pd.read_excel (r'./Rattermann.xlsx', sheet_name='Rattermann')
source_story = builtins.list(df['Base'])[1:19]
true_analogy = builtins.list(df['True Analogy Story'])[1:19]
false_analogy = builtins.list(df['False Analogy Story'])[1:19]
literal_similarity = builtins.list(df['Literally similar story'])[1:19]
mere_appearance = builtins.list(df['Mere-Appearance Match'])[1:19]

# Initialize results
all_analogy_results = []
all_analogy_correct = []
all_similarity_results = []
all_similarity_correct = []
N_source_stories = 18
for s in range(N_source_stories):
	print('Source story ' + str(s+1) + ' of ' + str(N_source_stories) + '...')
	# A. True analogy B. False analogy
	print('True analogy vs. false analogy')
	print(' ')
	prompt = 'Consider the following story:\n\nStory 1: ' + source_story[s] + '\n\nNow consider two more stories:\n\nStory A: ' + true_analogy[s] + '\n\nStory B: ' + false_analogy[s] + '\n\n'
	prompt += 'Which of Story A and Story B is a better analogy to Story 1? Is the best answer Story A, Story B, or both are equally analogous?'
	print(prompt)
	response = []
	while len(response) == 0:
		try:
			response = openai.Completion.create(prompt=prompt, **kwargs)
		except:
			print('trying again...')
			time.sleep(5)
	all_analogy_results.append(response['choices'][0]['text'])
	print('response: ' + response['choices'][0]['text'])
	all_analogy_correct.append('A')
	print('correct_answer: A')
	print(' ')
	# A. False analogy B. True analogy
	print('False analogy vs. true analogy')
	print(' ')
	prompt = 'Consider the following story:\n\nStory 1: ' + source_story[s] + '\n\nNow consider two more stories:\n\nStory A: ' + false_analogy[s] + '\n\nStory B: ' + true_analogy[s] + '\n\n'
	prompt += 'Which of Story A and Story B is a better analogy to Story 1? Is the best answer Story A, Story B, or both are equally analogous?'
	print(prompt)
	response = []
	while len(response) == 0:
		try:
			response = openai.Completion.create(prompt=prompt, **kwargs)
		except:
			print('trying again...')
			time.sleep(5)
	all_analogy_results.append(response['choices'][0]['text'])
	print('response: ' + response['choices'][0]['text'])
	all_analogy_correct.append('B')
	print('correct_answer: B')
	print(' ')
	# A. Literal similarity B. Mere appearance
	print('Literal similarity vs. mere appearance')
	print(' ')
	prompt = 'Consider the following story:\n\nStory 1: ' + source_story[s] + '\n\nNow consider two more stories:\n\nStory A: ' + literal_similarity[s] + '\n\nStory B: ' + mere_appearance[s] + '\n\n'
	prompt += 'Which of Story A and Story B is a better analogy to Story 1? Is the best answer Story A, Story B, or both are equally analogous?'
	print(prompt)
	response = []
	while len(response) == 0:
		try:
			response = openai.Completion.create(prompt=prompt, **kwargs)
		except:
			print('trying again...')
			time.sleep(5)
	all_similarity_results.append(response['choices'][0]['text'])
	print('response: ' + response['choices'][0]['text'])
	all_similarity_correct.append('A')
	print('correct_answer: A')
	print(' ')
	# A. Mere appearance B. Literal similarity
	print('Mere appearance vs. Literal similarity')
	print(' ')
	prompt = 'Consider the following story:\n\nStory 1: ' + source_story[s] + '\n\nNow consider two more stories:\n\nStory A: ' + mere_appearance[s] + '\n\nStory B: ' + literal_similarity[s] + '\n\n'
	prompt += 'Which of Story A and Story B is a better analogy to Story 1? Is the best answer Story A, Story B, or both are equally analogous?'
	print(prompt)
	response = []
	while len(response) == 0:
		try:
			response = openai.Completion.create(prompt=prompt, **kwargs)
		except:
			print('trying again...')
			time.sleep(5)
	all_similarity_results.append(response['choices'][0]['text'])
	print('response: ' + response['choices'][0]['text'])
	all_similarity_correct.append('B')
	print('correct_answer: B')
	print(' ')
	# Save results
	np.savez('./gpt3_rattermann_results.npz', all_analogy_results=all_analogy_results, all_analogy_correct=all_analogy_correct, all_similarity_results=all_similarity_results, all_similarity_correct=all_similarity_correct)

