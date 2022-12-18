## Digit Matrices

To create new digit matrix problems, run:
```
python3 ./gen_problems.py
python3 ./gen_4_5_rule_problems.py
python3 ./combine_problems_1thru5.py
```
Problems for experiment #1 (1-3 rule and logic problems) are contained in:
```
./all_problems.npz
```
and problems for experiment #2 (1-5 rule problems) are contained in:
```
./all_problems_1thru5.npz
```

To evaluate GPT-3 on experiment #1, run:
```
python3 ./eval_gpt_matprob.py
```
Note that you will need to enter your OpenAI API key (line 14).

To evaluate GPT-3 on experiment #2, run:
```
python3 ./eval_gpt_matprob_prog_1thru5.py
```
To analyze GPT-3's responses on these experiments, run:
```
python3 ./analyze_gpt3_exp1.py
python3 ./analyze_gpt3_exp2.py
```
To plot figures comparing GPT-3 with human performance, run:
```
python3 ./exp1_plot_GPT3_vs_human.py
python3 ./exp2_plot_GPT3_vs_human.py
```
To perform statistical analysis for experiment #1, run:
```
python3 ./exp1_create_stats_dset.py
```
and run the following R script:
```
./exp1_stats.r
```
To perform analysis comparing experiments #1 and #2, run:
```
python3 ./exp1_vs_exp2_create_stats_dset.py
```
and run the following R script:
```
./exp1_vs_exp2_stats.r
```
Note that results for human participants and GPT-3 are already provided.
