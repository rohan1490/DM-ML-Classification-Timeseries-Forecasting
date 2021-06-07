mydata <- read.csv("D:/Study/Data Mining and Machine Learning/Project/Datasets/Cardio/cardio_train.csv",header = TRUE, sep = ";")
head(mydata)
summary(mydata)
str(mydata)
#Removing id column
mydata$id <- NULL
head(mydata)

library(Amelia)
missmap(mydata, main = "Missing values vs observed")

mydata$age <- round(mydata$age/365.25)
head(mydata)

#Checking correclation between predictors and response variable
cor(mydata$cardio,mydata)

mydata$cardio <- as.factor(mydata$cardio)

#Visualizing data

#AGE

#ggplot(mydata, aes(age, colour = cardio)) +
  #geom_freqpoly(binwidth = 1) + labs(title="Age Distribution by presence of CDV")

library(ggplot2)
ggplot(mydata, aes(x=age, , fill=cardio ,color=cardio)) +
  geom_histogram(binwidth = 1,alpha=0.5) + labs(title="Presence of CDV according to age")+
  xlim(37 ,max(mydata$age))+
  theme_minimal()

#Range of ages
min(mydata$age)
max(mydata$age)


#GENDER
table(mydata$gender)

ggplot(mydata, aes(x = gender, fill = cardio)) + 
  geom_bar(position = "dodge")



#HEIGHT
#Removing values greater than 214 7ft and less than 130 cm 4'3 ft
mydata <- mydata[!(mydata$height <140) & !(mydata$height > 187),]

c <- ggplot(mydata, aes(x=height, fill=cardio,color=cardio)) +
  geom_histogram(binwidth = 5) + labs(title="Pregnancy Distribution by Outcome")
c + theme_gray() + xlim(min(mydata$weight), max(mydata$weight))
table(mydata$height)


#WEIGHT
#Removing weight less than 37 kgs
mydata <- mydata[!(mydata$weight <37),]


c <- ggplot(mydata, aes(x=weight, fill=cardio, color=cardio)) +
  geom_histogram(binwidth = 5) + labs(title="Pregnancy Distribution by Outcome")
c + theme_gray() + xlim(min(mydata$weight), max(mydata$weight))
table(mydata$weight)

#ap_hi systolic blood pressure

mydata <- mydata[!(mydata$ap_hi <90) & !(mydata$ap_hi >250),]

c <- ggplot(mydata, aes(x=ap_hi, fill=cardio, color=cardio)) +
  geom_histogram(binwidth = 10) + labs(title="Pregnancy Distribution by Outcome")
c + theme_gray() + xlim(min(mydata$weight), max(mydata$weight))
table(mydata$ap_hi)


#ap_low diastolic blood pressure

mydata <- mydata[!(mydata$ap_lo <40) & !(mydata$ap_lo >200),]

c <- ggplot(mydata, aes(x=ap_lo, fill=cardio,color=cardio)) +
  geom_histogram(binwidth = 10) + labs(title="Pregnancy Distribution by Outcome")
c + theme_gray() + xlim(min(mydata$weight), max(mydata$weight))

table(mydata$ap_lo)

#Cholesterol
table(mydata$cholesterol)
ggplot(mydata, 
       aes(x = cholesterol, 
           fill = cardio)) + 
  geom_bar(position = "dodge")

#Collapsing above normal and well above normal to one level

table(mydata$cholesterol)
mydata$cholesterol[(mydata$cholesterol == 3)]<-2
table(mydata$cholesterol)


#Glucose
table(mydata$gluc)
mydata$gluc[(mydata$gluc == 3)]<-2
table(mydata$gluc)

ggplot(mydata, 
       aes(x = gluc, 
           fill = cardio,color=cardio)) + 
  geom_bar(position = "dodge")



#Smoke
table(mydata$smoke)

ggplot(mydata, 
       aes(x = smoke, 
           fill = cardio,color=cardio)) + 
  geom_bar(position = "dodge")


#active
table(mydata$active)
ggplot(mydata, 
       aes(x = active, 
           fill = cardio,color=cardio)) + 
  geom_bar(position = "dodge")




#cardio
table(mydata$cardio)
ggplot(mydata, 
       aes(x = cardio, 
           fill = cardio,color=cardio)) + 
  geom_bar(position = "dodge")



#Converting the dichotomous variables to factors

mydata$gender <- as.factor(mydata$gender)
mydata$cholesterol <- as.factor(mydata$cholesterol)
mydata$gluc <- as.factor(mydata$gluc)
mydata$smoke <- as.factor(mydata$smoke)
mydata$alco <- as.factor(mydata$alco)
mydata$active <- as.factor(mydata$active)
mydata$cardio <- as.factor(mydata$cardio)

str(mydata)


library(splitstackshape)
library(caTools)
set.seed(42)

#Splitting data using stratified sampling
train_index <- stratified(mydata, c('gender','cholesterol', 'gluc','smoke', 'alco','active', 'cardio'), 0.7,keep.rownames = T)$rn
test_index <-setdiff(row.names(mydata),train_index)

#Creating train and test data with 70:30 ration
train_data <- mydata[train_index,]
test_data <- mydata[test_index,]

#Checking ratio of response variable after stratified sampling
prop.table(table(train_data$cardio)) * 100
prop.table(table(test_data$cardio)) * 100

#logistic regression

library(InformationValue)
library(caret)
library(e1071)
library(pROC)

model_logistic <- glm(cardio ~ ., data = train_data, family = "binomial")
summary(model_logistic)
par(mfrow = c(2, 2))
plot(model_logistic)
exp(coef(model_logistic))

#Step function
model_logistic_step <- step(model_logistic, approximation=FALSE)

library(car)
Anova(model_logistic_step, type="II", test="Wald")

library(rcompanion)
nagelkerke(model_logistic_step)

#Predicting using the model on test dataset
res_test <- predict(model_logistic_step, test_data, type="response")
str(res_test)

#To get a square plot
par(pty = 's')

library(pROC)
#With the train dataset
roc(train_data$cardio,model_logistic_step$fitted.values, plot= TRUE)

#With the train dataset
roc(test_data$cardio,res_test, plot= TRUE)

#Converting trained model fitted values as binary
res_train <- model_logistic_step$fitted.values
res_train[res_train>0.5] <- 1
res_train[res_train<=0.5] <- 0
res_train
res_train <- as.factor(res_train)

#Converting results from test and train to binary
res_test[res_test>0.5] <- 1
res_test[res_test<=0.5] <- 0
res_test
str(res_test)
res_test <- as.factor(res_test)

#Getting confusion matrices
#For train data
confusionMatrix(res_train,train_data$cardio)
#For test data
confusionMatrix(res_test,test_data$cardio)

#Get important variables according to logistic regression
library(caret)
varImp(model_logistic_step)


library(e1071)

model_naive=naiveBayes(cardio ~., data=train_data)
summary(model_naive)


str(model_naive)

res_train <- predict(model_naive, train_data,"raw")
res_test <- predict(model_naive, test_data,"raw")

x <- as.numeric(0)
for(i in 1:dim(res_train)[1]) {
  x[i] <- res_train[i,2]
}

y <- as.numeric(0)
for(i in 1:dim(res_test)[1]) {
  y[i] <- res_test[i,2]
}

roc(train_data$cardio,x, plot= TRUE)
roc(test_data$cardio,y, plot= TRUE)


res_train <- predict(model_naive, train_data)
res_test <- predict(model_naive, test_data)


#Getting confusion matrices
#For train data
confusionMatrix(res_train,train_data$cardio)
#For test data
confusionMatrix(res_test,test_data$cardio)




#Random forest
library(randomForest)

set.seed(42)

#Searching the best mtry value (1 to 10) by selecting 300 trees and 14 nodesize

trControl <- trainControl(method = "cv",
                          number = 5,
                          search = "grid")
tuneGrid <- expand.grid(.mtry = c(1: 10))
rf_mtry <- train(cardio~.,
                 data = train_data,
                 method = "rf",
                 metric = "Accuracy",
                 tuneGrid = tuneGrid,
                 trControl = trControl,
                 importance = TRUE,
                 nodesize = 14,
                 ntree = 100)
print(rf_mtry)

best_mtry <- rf_mtry$bestTune$mtry 
best_mtry

#Searching the best nodes

store_maxnode <- list()
tuneGrid <- expand.grid(.mtry = best_mtry)
for (maxnodes in c(5: 15)) {
  set.seed(42)
  rf_maxnode <- train(cardio~.,
                      data = train_data,
                      method = "rf",
                      metric = "Accuracy",
                      tuneGrid = tuneGrid,
                      trControl = trControl,
                      importance = TRUE,
                      nodesize = 14,
                      maxnodes = maxnodes,
                      ntree = 100)
  current_iteration <- toString(maxnodes)
  store_maxnode[[current_iteration]] <- rf_maxnode
}
results_mtry <- resamples(store_maxnode)
summary(results_mtry)

#for 16 to 25
for (maxnodes in c(16: 25)) {
  set.seed(42)
  rf_maxnode <- train(cardio~.,
                      data = train_data,
                      method = "rf",
                      metric = "Accuracy",
                      tuneGrid = tuneGrid,
                      trControl = trControl,
                      importance = TRUE,
                      nodesize = 14,
                      maxnodes = maxnodes,
                      ntree = 100)
  current_iteration <- toString(maxnodes)
  store_maxnode[[current_iteration]] <- rf_maxnode
}
results_mtry <- resamples(store_maxnode)
summary(results_mtry)

#for 26 to 35
for (maxnodes in c(26: 35)) {
  set.seed(42)
  rf_maxnode <- train(cardio~.,
                      data = train_data,
                      method = "rf",
                      metric = "Accuracy",
                      tuneGrid = tuneGrid,
                      trControl = trControl,
                      importance = TRUE,
                      nodesize = 14,
                      maxnodes = maxnodes,
                      ntree = 100)
  current_iteration <- toString(maxnodes)
  store_maxnode[[current_iteration]] <- rf_maxnode
}
results_mtry <- resamples(store_maxnode)
summary(results_mtry)

#Gettng the best number of tress
store_maxtrees <- list()
for (ntree in c(100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650)) {
  set.seed(42)
  rf_maxtrees <- train(cardio~.,
                       data = train_data,
                       method = "rf",
                       metric = "Accuracy",
                       tuneGrid = tuneGrid,
                       trControl = trControl,
                       importance = TRUE,
                       nodesize = 14,
                       maxnodes = 33,
                       ntree = ntree)
  key <- toString(ntree)
  store_maxtrees[[key]] <- rf_maxtrees
}

results_tree <- resamples(store_maxtrees)
summary(results_tree)


#Final model with parameters tuned
trControl <- trainControl(method = "cv",
                          number = 10,
                          search = "grid")
tuneGrid <- expand.grid(.mtry = 2)

#fitting the final model
fit_rf_train <- train(cardio~.,
                train_data,
                method = "rf",
                metric = "Accuracy",
                tuneGrid = tuneGrid,
                trControl = trControl,
                importance = TRUE,
                nodesize = 14,
                ntree = 100,
                maxnodes = 33)

#To get AUC of test data
fit_rf_test <- train(cardio~.,
                      test_data,
                      method = "rf",
                      metric = "Accuracy",
                      tuneGrid = tuneGrid,
                      trControl = trControl,
                      importance = TRUE,
                      nodesize = 14,
                      ntree = 100,
                      maxnodes = 33)

#With test data
prediction_test <-predict(fit_rf_train, test_data)
confusionMatrix(prediction_test, test_data$cardio)

#With train data
prediction_train <-predict(fit_rf_train, train_data)
confusionMatrix(prediction_train, train_data$cardio)

#ROC curves with train
roc(train_data$cardio,fit_rf_train$finalModel$votes[,2],plot= TRUE)

#ROC curves with test
roc(test_data$cardio,fit_rf_test$finalModel$votes[,2],plot= TRUE)

library(caret)
#To get the most important variables
varImp(fit_rf_train)



#New kNN
library(ISLR)
library(caret)

set.seed(42)
ctrl <- trainControl(method="cv", number = 5) 
knnFit <- train(cardio ~ ., data = train_data, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 4)

#Output of kNN fit
knnFit

knn_Predict_test <- predict(knnFit,newdata = test_data , type="prob")
knn_Predict_train <- predict(knnFit,newdata = train_data , type="prob")

x1 <- as.numeric(0)
for(i in 1:dim(knn_Predict_test)[1]) {
  x1[i] <- knn_Predict_test[i,2]
}

y1 <- as.numeric(0)
for(i in 1:dim(knn_Predict_train)[1]) {
  y1[i] <- knn_Predict_train[i,2]
}

#Roc with train
roc(train_data$cardio,y1, plot= TRUE)
#ROC with test
roc(test_data$cardio,x1, plot= TRUE)

knn_Predict_test_np <- predict(knnFit,newdata = test_data)
knn_Predict_train_np <- predict(knnFit,newdata = train_data)

#Confusion matrix
#With test
confusionMatrix(knn_Predict_test_np, test_data$cardio)
#With train
confusionMatrix(knn_Predict_train_np, train_data$cardio)










#SVM

#Splitting data using stratified sampling
train_index <- stratified(mydata_scaled, c('gender','cholesterol', 'gluc','smoke', 'alco','active', 'cardio'), 0.7,keep.rownames = T)$rn
test_index <-setdiff(row.names(mydata_scaled),train_index)

#Creating train and test data with 70:30 ration
train_data <- mydata[train_index,]
test_data <- mydata[test_index,]


trctrl <- trainControl(method = "cv", number = 5)
set.seed(42)

svm_Linear <- train(cardio ~., data = train_data, method = "svmLinear",
                    trControl=trctrl,
                    tuneLength = 4)
svm_Linear


#With test data
svm_Linear_test <-predict(svm_Linear, test_data)
confusionMatrix(svm_Linear_test, test_data$cardio)

#With train data
svm_Linear_train <-predict(svm_Linear, train_data)
confusionMatrix(svm_Linear_train, train_data$cardio)

#With Radial Kernel
svm_Radial <- train(cardio ~., data = train_data, method = "svmRadial",
                    trControl=trctrl,
                    tuneLength = 4)
svm_Radial

#With test data
svm_radial_test <-predict(svm_Radial, test_data)
confusionMatrix(svm_radial_test, test_data$cardio)

#With train data
svm_radial_train <-predict(svm_Radial, train_data)
confusionMatrix(svm_radial_train, train_data$cardio)

