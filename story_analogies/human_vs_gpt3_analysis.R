setwd("./")

# Human vs. GPT-3
data <-read.csv("./human_vs_gpt3_data.csv")
human_vs_gpt3_model <- glm(correct_pred ~ human_vs_gpt, data=data, family="binomial")
summary(human_vs_gpt3_model)
human_vs_gpt3_OR <- exp(cbind(OR = coef(human_vs_gpt3_model), confint(human_vs_gpt3_model)))
summary(human_vs_gpt3_OR)


