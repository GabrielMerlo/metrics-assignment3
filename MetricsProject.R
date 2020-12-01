#########################################################################################
################################ Project in Metrics: HW3 ################################
######################################################################################### 

#Install the packages we need
install.packages("hdm")
library(tidyverse)
library(GGally)
library(hdm)
install.packages("Hmisc")
library(Hmisc)

#Model
## Dependent:  logarithm of wages (lnw)
## Predictors: female + widowed + divorced + separated +nevermarried + hsd08 + hsd911 + hsg + cg + ad + mw + so + we + exp1 + exp2 + exp3

########################## a) Load, prepare and summarize data ##########################
######################################################################################### 

data("cps2012")
View(cps2012)
summary(cps2012)
summary(cps2012$lnw)
summary(cps2012$female)
#Hmisc describe for quantiles
describe(cps2012)
#Sapply for mean
sapply(cps2012, mean, na.rm=TRUE)

## Tables
#Table for female & Widowed
table(cps2012$female, cps2012$widowed)
#Table for female & divorced
table(cps2012$female, cps2012$divorced)
#Table for wages of females in 2012
table1 <- table(cps2012$year, cps2012$lnw, cps2012$female)
View(table1)
#Table for wages of married females
table2 <- table(cps2012$lnw, cps2012$married)
prop.table(table2)
View(table2)

#################### b) Ridge-Regression with cross-validation (CV) #####################
######################################################################################### 

library(glmnet)
library(tidyverse)
library(broom)
#Default grid of values of lambda 
#This sets the random seed for reproducibility of results
set.seed(100)
#Now we create an index to randomly sample observations in order to partition tour data.
index = sample(1:nrow(cps2012), 0.7*nrow(cps2012)) 
#Create the training and test set
train = cps2012[index,] # Create the training data 
test = cps2012[-index,] # Create the test data

#Dimensions of the training and test set
dim(train)
dim(test)
### The train set contains 70% of the data; test set contains the remaining 30%

#Set our Y and X
#Set y
y <- cps2012$lnw
#Set x [a matrix x with all the data of the predictors inside]
x <- cps2012 %>% select(female, widowed, divorced, separated, nevermarried, hsd08, hsd911, hsg, cg, ad, mw, so, we, exp1, exp2, exp3) %>% data.matrix()

## TRAINING SET
#Set our Y_train and X_train
y_train <- train$lnw
x_train <- train %>% select(female, widowed, divorced, separated, nevermarried, hsd08, hsd911, hsg, cg, ad, mw, so, we, exp1, exp2, exp3) %>% data.matrix()

## TEST SET
#Set our Y_test and X_test
y_test <- test$lnw
x_test <- test %>% select(female, widowed, divorced, separated, nevermarried, hsd08, hsd911, hsg, cg, ad, mw, so, we, exp1, exp2, exp3) %>% data.matrix()

# i) Ridge Regression with default grid of lamda values | 10-fold CV MSE Plot
lambdas <- 10^seq(3, -2, by = -.1)
ridge_reg <- glmnet(x, y, alpha = 0, lambda = lambdas)
summary(ridge_reg)
# Use training data 
cv_ridge <- cv.glmnet(x_train, y_train, type.measure = "mse", alpha = 0)
##Plot
plot(cv_ridge)

# ii) Optimal value for lambda λ by cross validation (CV)
opt_lambda <- cv_ridge$lambda.min #log(λ) that best minimised the error in CV is the lowest point in the curve
opt_lambda #[1] 0.0183073
fit <- cv_ridge$glmnet.fit
summary(fit)

#Predicted values for the teting
y_predicted <- predict(cv_ridge, s = opt_lambda, newx = x_test)
#Statistically lamda.1se is indistinguishable with lamda.min but results in a model with fewer parameters | We will use in all regressions lamda.min
y_predicted <- predict(fit, s = cv_ridge$lambda.1se, newx = x_test)

#MSE of the difference between the true values of y_test and y_predicted 
mean((y_test - y_predicted)^2)

# Sum of Squares Total and Error
sst <- sum((y - mean(y))^2)
sse <- sum((y_predicted - y)^2)

# R squared
rsq <- 1 - sse / sst
rsq #[1] 0.2401431

#-------------------------------------------------#

# Compute R^2 from true and predicted values
eval_results <- function(true, predicted, df) 
{
  SSE <- sum((predicted - true)^2)
  SST <- sum((true - mean(true))^2)
  R_square <- 1 - SSE / SST
  RMSE = sqrt(SSE/nrow(df))
  
  # Model performance metrics
  data.frame(RMSE = RMSE, Rsquare = R_square)
}

# Prediction and evaluation on train data
predictions_train <- predict(ridge_reg, s = opt_lambda, newx = x_train)
eval_results(y_train, predictions_train, train)

# Prediction and evaluation on test data
predictions_test <- predict(ridge_reg, s = opt_lambda, newx = x_test)
eval_results(y_test, predictions_test, test)

#-------------------------------------------------#

# iii) Why is the test MSE for Ridge often smaller than for OLS when lambda is not zero?
Ordinary least squares (OLS) minimizes the residual sum of squares (RSS)
𝑅𝑆𝑆=∑𝑖(𝜀𝑖)2=𝜀′𝜀=∑𝑖(𝑦𝑖−𝑦̂𝑖)2

The mean squared deviation equals 𝑀𝑆𝐸=𝑅𝑆𝑆𝑛, where 𝑛 is the number of observations.
Since 𝑛 is a constant, minimizing the RSS is equivalent to minimizing the MSE.
It is for this reason, that the Ridge-MSE cannot be smaller than the OLS-MSE. 
Ridge minimizes the RSS as well but under a constraint and as long 𝜆>0, this constraint is binding.
My version of the MSE is the in-sample MSE. When we calculate the mean squared error of a Ridge regression, we usually mean a different MSE. 
We are typically interested in how well the Ridge estimator allows us to predict out-of-sample. 
It is here, where Ridge may for certain values of 𝜆 outperform OLS.

We usually do not have out-of-sample observations so we split our sample into two parts.

Training sample, which we use to estimate the coefficients, say 𝛽̂𝑇𝑟𝑎𝑖𝑛𝑖𝑛𝑔

Test sample, which we use to assess our prediction 𝑦̂𝑇𝑒𝑠𝑡𝑖=𝑋𝑇𝑒𝑠𝑡𝑖𝛽̂𝑇𝑟𝑎𝑖𝑛𝑖𝑛𝑔

The test sample plays the role of the out-of-sample observations. The test-MSE is then given by
𝑀𝑆𝐸𝑇𝑒𝑠𝑡=∑𝑖(𝑦𝑇𝑒𝑠𝑡𝑖−𝑦̂𝑇𝑒𝑠𝑡𝑖)ˆ2

Ridge regression on test sample overfitts the data. The slope changes by introducing a small bias but the variance is oftn eliminaed. 
Hence the test MSE can often be smaller than the OLS. 

# iv) What is the optimal value of lambda? Is unrestricted OLS optimal here, in a test MSE sense?
opt_lambda #[1] 0.0183073


######################### b) LASSO with cross-validation (CV) ###########################
######################################################################################### 

# i) Lasso Regression with default grid of lamda values | 10-fold CV MSE Plot
##lambdas <- 10^seq(3, -2, by = -.1)
# Use training data 
cv_lasso <- cv.glmnet(x_train, y_train, type.measure = "mse", alpha = 1)
##Plot
plot(cv_lasso)

# ii) Optimal value for lambda λ by cross validation (CV)
opt_lambda_lasso <- cv_lasso$lambda.min #log(λ) that best minimised the error in CV is the lowest point in the curve
opt_lambda_lasso #[1] 0.0003944188

#Predicted values for the teting
y_pred.lasso <- predict(cv_lasso, s = opt_lambda, newx = x_test)
#Statistically lamda.1se is indistinguishable with lamda.min but results in a model with fewer parameters | We will use in all regressions lamda.min
y_pred.lasso <- predict(fit, s = cv_ridge$lambda.1se, newx = x_test)

#MSE of the difference between the true values of y_test and y_predicted 
mean((y_test - y_pred.lasso)^2)

#(iii) How many variables are used in the optimal Lasso fit? What are their coeficients? Is there a big difference here between Ridge and Lasso (in terms of test MSE)?
fit.alpha1 <- cv_lasso$glmnet.fit
summary(fit.alpha1)
The difference of MSE between Ridge and Lasso are: [MSE.R] 0.3443933 > [MSE.L] 0.3492436, so there is no big difference

#(iv) Which method of prediction would you choose and why? Is gender an important factor in the prediction model? Interpret the coeficient of female.
I would choose Ridge because the MSE is saller hence the variance is less and however the bias I get better results. 


















