import numpy as np

# Load data
results = np.load('./gpt3_rattermann_results.npz')
all_analogy_results = results['all_analogy_results']
all_analogy_correct = results['all_analogy_correct']
all_similarity_results = results['all_similarity_results']
all_similarity_correct = results['all_similarity_correct']

# Score
all_analogy_correct_pred = []
for i in range(len(all_analogy_results)):
	print('response: ' + all_analogy_results[i])
	print('correct_answer: ' + all_analogy_correct[i])
	correct_pred = int(input("Correct? (correct=1, incorrect=0): "))
	print(' ')
	all_analogy_correct_pred.append(correct_pred)
all_similarity_correct_pred = []
for i in range(len(all_similarity_results)):
	print('response: ' + all_similarity_results[i])
	print('correct_answer: ' + all_similarity_correct[i])
	correct_pred = int(input("Correct? (correct=1, incorrect=0): "))
	print('')
	all_similarity_correct_pred.append(correct_pred)

# Report accuracy
analogy_acc = np.mean(all_analogy_correct_pred)
print('analogy acc. = ' + str(np.around(analogy_acc,4)))
similarity_acc = np.mean(all_similarity_correct_pred)
print('similarity acc. = ' + str(np.around(similarity_acc,4)))

# Save
np.savez('./gpt3_rattermann_results.npz', all_analogy_results=all_analogy_results, all_analogy_correct=all_analogy_correct, all_analogy_correct_pred=all_analogy_correct_pred,
										  all_similarity_results=all_similarity_results, all_similarity_correct=all_similarity_correct, all_similarity_correct_pred=all_similarity_correct_pred)

