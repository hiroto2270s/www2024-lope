###########################################################################
## Project: ACTG 175 Survival Analysis
## Script purpose: R code for project
## Date: 23/11/24
## Author: Rory O'Sullivan 
###########################################################################

setwd("C:/Users/Dell/OneDrive/Desktop/LK296/LK296 SEM1/MS6071 R for Data Science/Project")
library(tidyverse)
library(survival)
library(survminer)
library(car) # vif
library(GGally)

df <- read_csv('AIDS_ClinicalTrial_GroupStudy175.csv')
df.full <- df
###########################################################################
#
# Factoring and Labelling data
#
# Note: more is done after eda so that part of eda will work, it is noted below
###########################################################################

# trt treatment indicator (0 = ZDV only;
# 1 = ZDV + ddI, 2 = ZDV + Zal, 3 = ddI only)
# table(trt)
df$trt <- factor(df$trt, levels = c("0", "1", "2", "3"), 
                 labels = c(".ZDV", ".ZDV_ddI", ".ZDV_Zal", ".ddI"))
df$trt <- relevel(df$trt, ref = ".ZDV")

# karnof: Karnofsky score (on a scale of 0-100)
df$karnof <- factor(df$karnof, levels = c("70", "80", "90", "100"), 
                    labels = c(".70", ".80", ".90", ".100"))
df$karnof <- relevel(df$karnof, ref = ".100")

#  strat:  antiretroviral history stratification 
# 1 = 'Antiretroviral Naive',
# 2 = '> 1 but <= 52 weeks of prior antiretroviral therapy',
# 3 = '> 52 weeks'
df$strat <- factor(df$strat, levels = c("1", "2", "3"),
                   labels = c("Naive", "Under_year", "Year_plus"))
df$strat <- relevel(df$strat, ref = "Naive")


# label: censoring indicator (1 = failure, 0 = censoring)

################################ Removing outliers #############################
indices_zero <- which(df$cd40 == 0)
# Patients 217, 218 and 724 have CD40 of 0, these will be removed
df <- df[-c(217, 218,724), ]
attach(df)

# Very large outlier
#max(cd820)
cd820_index <- which.max(cd820)
#cd820_index
df <- df[-(cd820_index), ]

# Very large outlier again
attach(df)
#max(cd40)
cd40_index <- which.max(cd40)
#cd40_index
df <- df[-(cd40_index), ]

# remove one outlier from cd80
attach(df)
lowest_values <- sort(cd80, na.last = TRUE)[1:5]
index_of_min <- which.min(df$cd80)
#index_of_min
df <- df[-index_of_min, ]

# view final df
#dim(df)
#str(df)
attach(df)

###########################################################################
#
# Exploratory Data Analysis
#
###########################################################################

# gets mean and sd of all numeric variables
numeric_cols <- c("time", "age", "wtkg", "preanti", "cd40", "cd420",
                  "cd80", "cd820")
mean_sd <- sapply(numeric_cols, function(col) {
  mean <- mean(df[[col]], na.rm = TRUE)    
  sd <- sd(df[[col]], na.rm = TRUE)     
  sprintf("%.2f (%.2f)", mean, sd) 
})
print(mean_sd)

# prints count of all levels in multilevel categorical vars
categorical_cols <- c("trt", "karnof", "strat")
counts <- sapply(categorical_cols, function(col) table(df[[col]]))
print(counts)

# prints count of positive class in var along with proportion of df
binary_cols <- setdiff(names(df), c(numeric_cols, categorical_cols))
pos_counts <- sapply(binary_cols, function(col) {
  count <- sum(df[[col]] == 1, na.rm = TRUE)    
  percentage <- (count / 2133) * 100     
  sprintf("%d (%.1f)", count, percentage) 
})
print(pos_counts)

# eda plot
cd.df <- df[,c("cd40", "cd420", "cd80", "cd820", "label")]
cd.df$label <- factor(cd.df$label, levels = c("0", "1"), 
                      labels = c("Censored", "Failure"))

cd.pairs <- ggpairs(cd.df,
                    aes(colour=label) )
cd.pairs

########### more refactoring and relabelling ##################

# quite messy but refactoring and labelling binary vars here as if did before the
# eda, the binary cols wouldnt work as theyd be factors, not binary

# hemo: hemophilia (0=no, 1=yes)
df$hemo <- factor(df$hemo, levels = c("0", "1"), 
                  labels = c(".no", ".yes"))
# homo: sexual orientation, homosexual activity (0=no, 1=yes)
df$homo <- factor(df$homo, levels = c("0", "1"), 
                  labels = c(".no", ".yes"))
# drugs: history of IV drug use (0=no, 1=yes)
df$drugs <- factor(df$drugs, levels = c("0", "1"), 
                   labels = c(".no", ".yes"))

# race: race (0=White, 1=non-white)
df$race <- factor(df$race, levels = c("0", "1"), 
                  labels = c(".white", ".non-white"))

# gender: (0=F, 1=M)
df$gender <- factor(df$gender, levels = c("0", "1"), 
                    labels = c(".female", ".male"))

# str2: antiretroviral history (0=naive, 1=experienced
df$str2 <- factor(df$str2, levels = c("0", "1"), 
                  labels = c(".naive", ".experienced"))

# symptom: symptomatic indicator (0=asymp, 1=symp)
df$symptom <- factor(df$symptom, levels = c("0", "1"), 
                     labels = c(".asymptomatic", ".symptomatic"))

# offtrt:  indicator of off-trt before 96+/-5 weeks (0=no,1=yes)
df$offtrt <- factor(df$offtrt, levels = c("0", "1"), 
                    labels = c(".no", ".yes"))

attach(df)
###########################################################################
#
# Results
#
###########################################################################


############################## Kaplan Meier ###############################

# Kap meier for trt
fit.trt <- survfit(Surv(time, label) ~ trt, data = df)
ggsurvplot(
  fit.trt, 
  data = df, 
  size = 1.0,                 
  palette = 
    c("salmon", "limegreen", "#2E9FDF", "darkviolet"),
  legend.title = "Treatment",
  legend.labs = c("ZDV", "ZDV_ddl", "ZDV_Zal", "ddl"),
  xlab = ("Time(days)"),
  ylab = ("Survival Probability"),
  conf.int = FALSE,          
  pval = TRUE,              
  ggtheme = theme_classic(),
  censor.shape="|", 
  censor.size = 3.5,
)

# Kap meier for offtrt
fit.offtrt <- survfit(Surv(time, label) ~ offtrt, data = df)
ggsurvplot(
  fit.offtrt, 
  data = df, 
  size = 1.0,                 
  palette = 
    c("salmon","#2E9FDF"),
  legend.title = "Off-Treatment",
  legend.labs = c("No", "Yes"),
  xlab = ("Time(days)"),
  ylab = ("Survival Probability"),
  conf.int = FALSE,          
  pval = TRUE,              
  ggtheme = theme_classic(),
  censor.shape="|", 
  censor.size = 3.5,
)

############################## Modelling ###############################

# Log Rank for shortlisting
logrank.test <- survdiff(Surv(time, label) ~ str2, data = df)
logrank.test

# Cox prop hazards for shortlisting
fit.cox <- coxph(Surv(time, label) ~ drugs, data = df)
summary(fit.cox)

#################################################################
# Baseline Model

# logging these variables due to long tails
df$log_cd80 <- log(df$cd80)  
df$log_cd40 <- log(df$cd40)  
df$log_cd820 <- log(df$cd820)  
df$log_cd420 <- log(df$cd420)  
attach(df)

fit.baseline <- coxph(Surv(time, label) ~ trt + drugs + symptom  + log_cd40 +
                        log_cd80 + str2 + karnof  , data = df)
fit.baseline
summary(fit.baseline)

vif(fit.baseline) # nothing sig

AIC(fit.baseline)

# Check proportional hazards assumption
cox_zph_fit_baseline <- cox.zph(fit.baseline)
plot(cox_zph_fit_baseline, var = "trt", resid = TRUE)
print(cox_zph_fit_baseline)

##################################
# Updated model
fit.updated.full <- coxph(Surv(time, label) ~ trt + drugs + symptom + log_cd420 +
                            log_cd820 + str2 + karnof  + offtrt, data = df)
fit.updated.full
summary(fit.updated.full)

vif(fit.updated.full) # nothing sig
AIC(fit.updated.full)

cox_zph_fit_updated.full <- cox.zph(fit.updated.full)
print(cox_zph_fit_updated.full)

##################################
# Final model

fit.updated <- coxph(Surv(time, label) ~ trt  + symptom + log_cd420 +
                       log_cd820 + offtrt + log_cd40, data = df)
fit.updated
summary(fit.updated)

vif(fit.updated) # nothing sig
AIC(fit.updated)

cox_zph_fit_updated <- cox.zph(fit.updated)
print(cox_zph_fit_updated)
plot(cox_zph_fit_updated, var = "log_cd420")


