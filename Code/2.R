
mydata <- read.csv("D:/Study/Data Mining and Machine Learning/Project/Datasets/Credit card payment/default.csv")
head(mydata)
summary(mydata)
str(mydata)
#Removing id column
mydata$ï..<- NULL
head(mydata)
#Removing first row
mydata <- mydata[-1,]
#renaming the rows from index 1
row.names(mydata) <- 1:nrow(mydata)
str(mydata)
head(mydata)
str(mydata)

table(mydata$Y)

#For checking missing data
#missmap(mydata, main = "Missing values vs observed")

#All to numeric
cols.num <- c(1:24)
mydata[cols.num] <- sapply(mydata[cols.num],as.numeric)

cor(mydata)

mydata <- mydata[,c(1,6:11,24)]

str(mydata)
cor(mydata)

#Collapsing factors 

table(mydata$X6)
mydata$X6[mydata$X6 > 1 ] <- 1
table(mydata$X6)

table(mydata$X7)
mydata$X7[mydata$X7 > 1 ] <- 1

table(mydata$X8)
mydata$X8[mydata$X8 > 1 ] <- 1

table(mydata$X9)
mydata$X9[mydata$X9 > 1 ] <- 1

table(mydata$X10)
mydata$X10[mydata$X10 > 1 ] <- 1

table(mydata$X11)
mydata$X11[mydata$X11 > 1 ] <- 1


#Convertig to factors

mydata$X6 <- as.factor(mydata$X6)
mydata$X7 <- as.factor(mydata$X7)
mydata$X8 <- as.factor(mydata$X8)
mydata$X9 <- as.factor(mydata$X9)
mydata$X10 <- as.factor(mydata$X10)
mydata$X11 <- as.factor(mydata$X11)
mydata$Y <- as.numeric(mydata$Y)

mydata$Y <- as.factor(mydata$Y)

str(mydata)

library(splitstackshape)
library(caTools)
set.seed(555)

train_index <- stratified(mydata, c('X2','X3', 'X4','X5', 'X6','X7', 'X8','X9', 'X10','X11'), 0.6,keep.rownames = T)$rn
train_index <- stratified(mydata, c('X6','X7', 'X8','X9', 'X10','X11'), 0.5,keep.rownames = T)$rn

test_index <-setdiff(row.names(mydata),train_index)
train_data <- mydata[train_index,]
test_data <- mydata[test_index,]



#Split into test and train data without stratifying

#library(caTools)
#set.seed(101) 
#sample = sample.split(mydata$Y, SplitRatio = .70)
#train = subset(mydata, sample == TRUE)
#test  = subset(mydata, sample == FALSE)
#row.names(test) <- 1:nrow(test)
#row.names(train) <- 1:nrow(train)
#str(test)
#logistic regression

model <- glm(Y ~ ., data = train_data, family = "binomial")
summary(model)
par(mfrow = c(2, 2))
plot(model)

modelstep <- step(model)
summary(modelstep)
plot(modelstep)

res_test <- predict(model, test_data, type="response")
res_test
head(res_test)

res_test[res_test>0.5] <- 1
res_test[res_test<=0.5] <- 0
res_test <- as.numeric(res_test)
res_test <- as.factor(res_test)
table(res_test)
table(test_data$Y)

str(res_test)
library(caret)
confusionMatrix(res_test,test_data$Y)
####
res_train <- modelstep$fitted.values
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
confusionMatrix(res_train,train_data$Y)
#For test data
confusionMatrix(res_test,test_data$Y)


library(pROC)
roc(as.numeric(test_data$Y), as.numeric(res_test), plot = TRUE, legacy.axes=TRUE, percent = TRUE, xlab="False Positive Percentage", ylab="True Postive percentage")
roc(as.numeric(train_data$Y), as.numeric(res_train), plot = TRUE, legacy.axes=TRUE, percent = TRUE, xlab="False Positive Percentage", ylab="True Postive percentage")
res <- predict(modelstep, test_data, type="response")


#Random forest
library(randomForest)
set.seed(42)

model2 <- randomForest(Y~., data=train_data, mtry=4, ntree=100, importance=TRUE)
par(mfrow = c(1, 1))
plot(model2)
p1 <- predict(model2, test_data)
p2 <- predict(model2, train_data)

str(p1)
str(test_data$Y)
par(pty='s')
roc(as.numeric(test_data$Y), as.numeric(p1), plot = TRUE, legacy.axes=TRUE, percent = TRUE, xlab="False Positive Percentage", ylab="True Postive percentage")
roc(as.numeric(train_data$Y), as.numeric(p2), plot = TRUE, legacy.axes=TRUE, percent = TRUE, xlab="False Positive Percentage", ylab="True Postive percentage")
confusionMatrix(p1, test_data$Y)
confusionMatrix(p2, train_data$Y)


#KNN
set.seed(42)
ctrl <- trainControl(method="cv", number = 5)
knnFit <- train(Y~., data = train_data, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 4)

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
roc(train_data$Y,y1, plot= TRUE)
#ROC with test
roc(test_data$Y,x1, plot= TRUE)

knn_Predict_test_np <- predict(knnFit,newdata = test_data)
knn_Predict_train_np <- predict(knnFit,newdata = train_data)

#Confusion matrix
#With test
confusionMatrix(knn_Predict_test_np, test_data$Y)
#With train
confusionMatrix(knn_Predict_train_np, train_data$Y)







