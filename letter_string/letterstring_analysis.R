setwd("./")
library(dplyr)
library(Matrix)
library(lme4)
data <-read.csv("./letterstring_data.csv")

# Omnibus model
model <- glm(correct_pred ~ human_vs_gpt + N_gen + human_vs_gpt:N_gen, data=data, family="binomial")
summary(model)

# Zero-generalization problems
zerogen_model <- glm(correct_pred ~ human_vs_gpt, data=subset(data, N_gen==0), family="binomial")
summary(zerogen_model)

# Effect of number of generalizations
# Human
model <- glm(correct_pred ~ N_gen, data=subset(data, human_vs_gpt==0), family="binomial")
summary(model)
# GPT-3
model <- glm(correct_pred ~ N_gen, data=subset(data, human_vs_gpt==1), family="binomial")
summary(model)