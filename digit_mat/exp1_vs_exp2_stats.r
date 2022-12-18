setwd("./")
data <-read.csv("./exp1_vs_exp2_all_data.csv")

# Generative task - problem type X experiment (progressive vs. random order) interaction
# Human only
human_gen <- glm(gen_correct_pred ~ prob_type + exp1_vs_exp2 + prob_type:exp1_vs_exp2, data=subset(data, human_vs_gpt==0), family="binomial")
summary(human_gen)
# GPT-3 only
GPT3_gen <- glm(gen_correct_pred ~ prob_type + exp1_vs_exp2 + prob_type:exp1_vs_exp2, data=subset(data, human_vs_gpt==1), family="binomial")
summary(GPT3_gen)
