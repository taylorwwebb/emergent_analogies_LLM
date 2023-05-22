## Letter String Analogies

To create new letter string problems, run:
```
python3 ./gen_problems.py
```
Problems are contained in:
```
./all_prob.npz
```
To evaluate GPT-3 on letter string problems, run:
```
python3 ./eval_GPT3_letterstring_prob.py
```
Note that you will need to enter your OpenAI API key (line 15).

To analyze GPT-3's responses, run:
```
python3 ./analyze_gpt3_letterstring.py
```
To plot figures comparing GPT-3 with human performance, run:
```
python3 ./compare_behavior_gpt3.py
```
To perform regression analyses, run:
```
python3 ./create_regression_dsets.py
```
and run the following R script:
```
./letterstring_analysis.R
```
To perform correlation analyses, run:
```
python3 ./corr_analysis.py
```
To evaluate GPT-3 on letter string problems presented in alternative formats (without a prompt, or in the form of a setence), use the ```--noprompt``` or ```--sentence``` arguments for these scripts.

Note that results for human participants and GPT-3 are already included in this repository.
