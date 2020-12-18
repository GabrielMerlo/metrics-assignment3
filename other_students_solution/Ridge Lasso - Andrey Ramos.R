#Preparation

install.packages("hdm")
install.packages("magrittr")
install.packages("dplyr")
require("hdm")
require("MASS")
require("glmnet")
require("magrittr")
require("dplyr")

#1. 
data <- cps2012
attach(data)
summary(data)
x <- as.matrix(data[,3:18])
y <- as.matrix(data[,2])

# Define train and test samples
set.seed(1)
train <-  data %>% sample_frac(0.7)
test <- data %>% setdiff(train)
y_train = as.matrix(train[,2])
y_test = as.matrix(test[,2])
x_train = as.matrix(train[,3:18])
x_test = as.matrix(test[,3:18])

#Fit ridge regression using cross validation
cvridge = cv.glmnet(x_train, y_train, alpha = 0, family = "gaussian")
plot(cvridge) #Plot the 10-fold CV MSE as a function of lambda
bestlamda = cvridge$lambda.min  # Select lamda that minimizes training MSE
bestlamda

#Now, lets fit the model using the best lambda
ridgereg = glmnet(x_train, y_train, alpha = 0, lambda = bestlamda, family = "gaussian")
coef(ridgereg) #The number of variables used in ridge fit is 16, the same as in OLS since Ridge does not make variables selection


#Lets calculate the test MSE
ridge_pred = predict(ridgereg, s = bestlamda, newx = x_test)
mean((ridge_pred - y_test)^2)

#Comparing with OLS
olsreg = glmnet(x_train, y_train, alpha = 0, lambda = 0)
coef(olsreg)
olspred = predict(olsreg, newx = x_test)
mean((olspred - y_test)^2) #In this case, the test MSE is similar using Ridge and OLS

#Finally, lets estimate the model in the full sample using the best lambda
ridgeregfull = glmnet(x, lnw, alpha = 0, lambda = bestlamda, family = "gaussian")
coef(ridgeregfull)

#Fit lasso regression using cross validation
cvlasso = cv.glmnet(x_train, y_train, alpha = 1, family = "gaussian")
plot(cvlasso) #Plot the 10-fold CV MSE as a function of lambda
bestlassomin= cvlasso$lambda.min  # Select lamda that minimizes training MSE
bestlassomin
bestlasso1se = cvlasso$lambda.1se 
bestlasso1se # Lambda 1se makes a stronger variable selection

#Now, lets fit the model using the best lambda
reglasso = glmnet(x_train, y_train, alpha = 1, lambda = bestlassomin, family = "gaussian")
coef(reglasso) # In this case, the number of variables used in lasso fit is 16, the same as in OLS. For this value of lambda, lasso does not make variables selection
reglasso2 = glmnet(x_train, y_train, alpha = 1, lambda = bestlasso1se, family = "gaussian")
coef(reglasso2) # With a more strict lambda, the model set 6 coefficients to be exactly zero.

#Lets calculate the test MSE
lasso1_pred = predict(reglasso, s = bestlassomin, newx = x_test)
mean((lasso1_pred - y_test)^2)
lasso2_pred = predict(reglasso2, s = bestlasso1se, newx = x_test)
mean((lasso2_pred - y_test)^2) #There is not much difference between Lasso, Ridge and OLS MSE test

#Finally, lets estimate the model in the full sample using the best lambda
lassofull = glmnet(x, y, alpha = 1, lambda = bestlassomin, family = "gaussian")
coef(lassofull)
lassofull2 = glmnet(x, y, alpha = 1, lambda = bestlasso1se, family = "gaussian")
coef(lassofull2)
