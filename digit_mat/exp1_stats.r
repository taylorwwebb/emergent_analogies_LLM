# Load data
setwd("./")
data <-read.csv("./exp1_all_data.csv")

# Overall generative performance
all_gen <- glm(gen_correct_pred ~ prob_type + human_vs_gpt + prob_type:human_vs_gpt, data=data, family="binomial")
summary(all_gen)
# Overall multiple-choice performance
all_MC <- glm(MC_correct_pred ~ prob_type + human_vs_gpt + prob_type:human_vs_gpt, data=data, family="binomial")
summary(all_MC)

# Two-rule problems, generative performance, progression rule vs. no progression rule
# Human only
human_prog_vs_noprog <- glm(gen_correct_pred ~ tworule_prog_noprog, data=subset(subset(data, prob_type==1), human_vs_gpt==0), family="binomial")
summary(human_prog_vs_noprog)
# GPT-3 only
GPT3_prog_vs_noprog <- glm(gen_correct_pred ~ tworule_prog_noprog, data=subset(subset(data, prob_type==1), human_vs_gpt==1), family="binomial")
summary(GPT3_prog_vs_noprog)

# Analysis of relational complexity
# Human only
human_relcompl <- glm(gen_correct_pred ~ N_unique_rules, data=subset(subset(data, prob_type==2), human_vs_gpt==0), family="binomial")
summary(human_relcompl)
# GPT-3 only
GPT3_relcompl <- glm(gen_correct_pred ~ N_unique_rules, data=subset(subset(data, prob_type==2), human_vs_gpt==1), family="binomial")
summary(GPT3_relcompl)

# Aligned vs. permuted logic problems
# Human only
human_aligned_permute <- glm(gen_correct_pred ~ aligned_permuted, data=subset(subset(data, prob_type==3), human_vs_gpt==0), family="binomial")
summary(human_aligned_permute)
# GPT-3 only
GPT3_aligned_permute <- glm(gen_correct_pred ~ aligned_permuted, data=subset(subset(data, prob_type==3), human_vs_gpt==1), family="binomial")
summary(GPT3_aligned_permute)


