## Letter String Analogies

To create letter string analogy problems, run:
```
python3 ./gen_letter_string_analogies.py
```
Problems will be located in:
```
./letter_string_analogies.npz
```
To evaluate GPT-3 on zero-shot letter string problems, run:
```
python3 ./eval_gpt_letter_string.py
```
Note that you will need to fill in your OpenAI API key (line 7).

To evaluate GPT-3 on relational priming test, run:
```
python3 ./eval_gpt_relational_prime.py
```
To analyze zero-shot letter string performance, run:
```
python3 ./analyze_letter_string_zeroshot.py
```
To analyze relational priming effect, run:
```
python3 ./analyze_relational_prime.py
```
Note that results for GPT-3 are already included in this repository.
