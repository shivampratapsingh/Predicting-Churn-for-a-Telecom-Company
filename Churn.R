library(dummy)

# Adding training and testing datasets
train = read.csv('C:/Users/Shivam/Downloads/DeZyre/Telecom_Train.csv')
test = read.csv('C:/Users/Shivam/Downloads/DeZyre/Telecom_Test.csv')

# Removing the first column as there is no use of it
train_df = train[,-1]
View(train_df)
test_df = test[,-1]

# Taking all numerical and target feature into one dataset
traindf = train_df[,c(2,6:20)]
testdf = test_df[,c(2,6:20)]

# Visualizing in boxplot for all the numeric columns to check for the outliers
par(mfrow=c(3,3))
for(i in 1:ncol(traindf)) {
  boxplot(traindf[,i], main=names(traindf)[i])
}

# Extracting the variables like state, area_code in order to make them dummy variables
train_cat = train_df[,c(-2,-4:-20)]
test_cat = test_df[,c(-2,-4:-20)]

#Visualizing the variables
par(mfrow=c(2,1))
barplot(table(train_df$state))
barplot(table(train_df$area_code))
barplot(table(train_df$international_plan))
barplot(table(train_df$voice_mail_plan))
barplot(table(train_df$churn))

# Creating dummy variables
cat1 = dummy(train_cat,p='all')
cat2 = dummy(test_cat,p='all')

# final training dataset
final_train = data.frame(traindf,cat1)
final_test = data.frame(testdf,cat2)

library(caret)
install.packages("mlr")
library(mlr)

# basic logistic regression based classifier
# RUN 1:
fit_glm = glm(churn~.,data=final_train,
              family = binomial(link='logit'))
summary(fit_glm)

final_train$pred = ifelse(predict(fit_glm,
                                  final_train,
                                  type = 'response')>0.5,'yes','no')

confusionMatrix(final_train$churn,as.factor(final_train$pred))

final_test$pred = ifelse(predict(fit_glm,
                                 final_test,
                                 type = 'response')>0.5,'yes','no')

confusionMatrix(final_test$churn,as.factor(final_test$pred))

#If I change the threshold probability it will change the reliability statistics
# The split of probability should be in confirmity with the train dataset

(table(final_train$churn)/nrow(final_train)*100)
(table(final_train$pred)/nrow(final_train)*100)
(table(final_test$churn)/nrow(final_test)*100)
(table(final_test$pred)/nrow(final_test)*100)
#----------------------------------------------
# RUN 2:
final_train$pred = ifelse(predict(fit_glm,
                                  final_train,
                                  type = 'response')>0.6,'yes','no')

confusionMatrix(final_train$churn,as.factor(final_train$pred))

final_test$pred = ifelse(predict(fit_glm,
                                 final_test,
                                 type = 'response')>0.6,'yes','no')

confusionMatrix(final_test$churn,as.factor(final_test$pred))

# the split of probability should be in confirmity with the train dataset

(table(final_train$churn)/nrow(final_train)*100)
(table(final_train$pred)/nrow(final_train)*100)
(table(final_test$churn)/nrow(final_test)*100)
(table(final_test$pred)/nrow(final_test)*100)

#--------------------------------------------------
# RUN 3:
final_train$pred = ifelse(predict(fit_glm,
                                  final_train,
                                  type = 'response')>0.4,'yes','no')

confusionMatrix(final_train$churn,as.factor(final_train$pred))

final_test$pred = ifelse(predict(fit_glm,
                                 final_test,
                                 type = 'response')>0.4,'yes','no')

confusionMatrix(final_test$churn,as.factor(final_test$pred))

# the split of probability should be in confirmity with the train dataset
(table(final_train$churn)/nrow(final_train)*100)
(table(final_train$pred)/nrow(final_train)*100)
(table(final_test$churn)/nrow(final_test)*100)
(table(final_test$pred)/nrow(final_test)*100)
#--------------------------------------------------------------
# RUN 4:
final_train$pred = ifelse(predict(fit_glm,
                                  final_train,
                                  type = 'response')>0.3,'yes','no')

confusionMatrix(final_train$churn,as.factor(final_train$pred))

final_test$pred = ifelse(predict(fit_glm,
                                 final_test,
                                 type = 'response')>0.3,'yes','no')

confusionMatrix(final_test$churn,as.factor(final_test$pred))

# the split of probability should be in confirmity with the train dataset
(table(final_train$churn)/nrow(final_train)*100)
(table(final_train$pred)/nrow(final_train)*100)
(table(final_test$churn)/nrow(final_test)*100)
(table(final_test$pred)/nrow(final_test)*100)

#Accuracy : 0.8368  
#Kappa : 0.3065 

# Trying to stabilize the result using cross validation
library(caret)
library(rpart)
train_control <- trainControl(method = 'cv',number = 10)

# Decision tree based methods
fit_dt <- caret::train(churn~.,data=train_df, trControl=train_control, method='rpart',metric= 'Accuracy')
fit_dt
predictions <- predict(fit_dt,test_df)
pred = cbind(test_df,predictions)

confusionMatrix(pred$churn,pred$predictions)

#Accuracy : 0.8872 
#Kappa : 0.3413

varImp(fit_dt)
#----------------

# Decision tree based methods
fit_dtc50 <- caret::train(churn~.,data=train_df,
                   trControl=train_control,
                   method='C5.0')

fit_dtc50
predictions <- predict(fit_dtc50,test_df)
pred = cbind(test_df,predictions)

confusionMatrix(pred$churn,pred$predictions)
varImp(fit_dtc50)

#Accuracy : 0.961   
#Kappa : 0.8146

#------------------------------------
#logistic regression model
fit_glm <- caret::train(churn~.,data=train_df,
                 trControl=train_control,
                 method='glm')

fit_glm
predictions <- predict(fit_glm,test_df)
pred = cbind(test_df,predictions)

confusionMatrix(pred$churn,pred$predictions)
varImp(fit_glm)

#Accuracy : 0.8692 
#Kappa : 0.2699 

#----------------------------------------------
fit_bstTree <- caret::train(churn~.,data=train_df,
                     trControl=train_control,
                     method='bstTree')

fit_bstTree
predictions <- predict(fit_bstTree,test_df)
pred = cbind(test_df,predictions)

confusionMatrix(pred$churn,pred$predictions)
varImp(fit_bstTree)

#Accuracy : 0.9442 
#Kappa : 0.7182  
#----------------------------------------
fit_C5.0Cost <- caret::train(churn~.,data=train_df,
                      trControl=train_control,
                      method='C5.0Cost')

fit_C5.0Cost
predictions <- predict(fit_C5.0Cost,test_df)
pred = cbind(test_df,predictions)

confusionMatrix(pred$churn,pred$predictions)
varImp(fit_C5.0Cost)

#Accuracy : 0.961  
#Kappa : 0.8146

#----------------------------------
fit_C5.0Rules <- caret::train(churn~.,data=train_df,
                       trControl=train_control,
                       method='C5.0Rules')

fit_C5.0Rules
predictions <- predict(fit_C5.0Rules,test_df)
pred = cbind(test_df,predictions)

confusionMatrix(pred$churn,pred$predictions)
varImp(fit_C5.0Rules)

#Accuracy : 0.9472 
#Kappa : 0.7441   

#--------------------------------
fit_treebag <- caret::train(churn~.,data=train_df,
                     trControl=train_control,
                     method='treebag')

fit_treebag
predictions <- predict(fit_treebag,test_df)
pred = cbind(test_df,predictions)

confusionMatrix(pred$churn,pred$predictions)
varImp(fit_treebag)

#Accuracy : 0.9526
#Kappa : 0.7718   
#--------------------------------
fit_xgbTree <- caret::train(churn~.,data=train_df,
                     trControl=train_control,
                     method='xgbTree')

fit_xgbTree
predictions <- predict(fit_xgbTree,test_df)
pred = cbind(test_df,predictions)

confusionMatrix(pred$churn,pred$predictions)
varImp(fit_xgbTree)

#Accuracy : 0.8656 
#Kappa : 0.1155 

#--------------------------------
# Ensemble methods
# Bagging- boost strap aggregation (random forest)
# Boosting- (gradient boosting)
# stacking- (different models)
# 500 observations to train the grid search model for ensemble learning
# takes time to finish one run

control <- trainControl(method = 'repeatedcv',
                        number = 10,
                        repeats = 3)
seed <- 1234
metric <- 'Accuracy'
set.seed(seed)
mtry <- sqrt(ncol(train_df))
tunegrid <- expand.grid(.mtry = mtry)
fit_rf_default <- caret::train(churn~.,
                        data=train_df[1:500,],
                        method='rf',
                        metric=metric,
                        trControl=control)
fit_rf_default

predictions <- predict(fit_rf_default,test_df)
pred = cbind(test_df,predictions)

confusionMatrix(pred$churn,pred$predictions)
varImp(fit_rf_default)

#Accuracy : 0.931
#Kappa : 0.6546

#-----------------------------------------
control <- trainControl(method = 'repeatedcv',
                        number = 10,
                        repeats = 3,
                        search = 'random')
seed <- 1234
metric <- 'Accuracy'
set.seed(seed)
mtry <- sqrt(ncol(train_df))
#tunegrid <- expand.grid(.mtry = mtry)
fit_rf_random <- caret::train(churn~.,
                       data=train_df[1:1000,],
                       method='rf',
                       metric=metric,
                       tuneLength=15,
                       trControl=control)
fit_rf_random
predictions <- predict(fit_rf_random,test_df)
pred = cbind(test_df,predictions)

confusionMatrix(pred$churn,pred$predictions)
varImp(fit_rf_random)

#Accuracy : 0.9424 
#Kappa : 0.7072

plot(fit_rf_random)


#-----------------------------------------
control <- trainControl(method = 'repeatedcv',
                        number = 10,
                        repeats = 3,
                        search = 'grid')
seed <- 1234
metric <- 'Accuracy'
set.seed(seed)
#mtry <- sqrt(ncol(train_df))
tunegrid <- expand.grid(.mtry = c(1:15))
fit_rf_grid <- caret::train(churn~.,
                     data=train_df[1:1000,],
                     method='rf',
                     metric=metric,
                     tuneGrid=tunegrid,
                     trControl=control)
fit_rf_grid
predictions <- predict(fit_rf_grid,test_df)
pred = cbind(test_df,predictions)

confusionMatrix(pred$churn,pred$predictions)
varImp(fit_rf_grid)

plot(fit_rf_grid)

#---------------
#boosting model:
control <- trainControl(method = 'repeatedcv',
                        number = 10,
                        repeats = 3,
                        search = 'grid')
seed <- 1234
metric <- 'Accuracy'
set.seed(seed)
fit_gbm <- caret::train(churn~.,
                 data=train_df,
                 method='gbm',
                 metric=metric,
                 trControl=control)
fit_gbm
predictions <- predict(fit_gbm,test_df)
pred = cbind(test_df,predictions)

confusionMatrix(pred$churn,pred$predictions)

library(gbm)
varImp(fit_gbm)
plot(fit_gbm)
