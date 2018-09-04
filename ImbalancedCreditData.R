library(DMwR) #For smote
library(ROSE) #For rose
library(caret)
library(data.table)
card.data <- fread("creditcard.csv",header=TRUE,sep=",",na.strings = '?')
card.data <- card.data[,2:31]
card.data <- apply(card.data, 2, function(x) as.numeric(x))
card.data <- as.data.frame(card.data)
head(card.data,3)
card.data <- na.omit(card.data)
head(card.data,3)
card.data$Class <- as.factor(card.data$Class)
levels(card.data$Class) <- c("Ok", "Fraud")
table(card.data$Class)

inTrainRows <- createDataPartition(card.data$Class,p=0.1,list=FALSE)
trainData <- card.data[inTrainRows,]
testData <-  card.data[-inTrainRows,]
table(trainData$Class)
table(testData$Class)

down_train <- downSample(x = trainData[, -ncol(trainData)], y = trainData$Class)
table(down_train$Class)
up_train <- upSample(x = trainData[, -ncol(trainData)],  y = trainData$Class)
table(up_train$Class)

smote_train <- SMOTE(Class ~ ., data  = trainData)                         
table(smote_train$Class) 

rose_train <- ROSE(Class ~ ., data  = trainData)$data                         
table(rose_train$Class) 

ctrl <- trainControl(method = "repeatedcv", repeats = 5,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary)

orig_fit <- train(Class ~ ., data = trainData, 
                  method = "lda",
                  metric = "ROC",
                  trControl = ctrl)
origPP <- predict(orig_fit, testData[,-ncol(testData)],type="raw")
testCNFM <- confusionMatrix(origPP,testData[,ncol(testData)])

down_fit <- train(Class ~ ., data = down_train, 
                      method = "lda",
                      metric = "ROC",
                      tuneLength=15,
                      trControl = ctrl)
downPP <- predict(down_fit, testData[,-ncol(testData)],type="raw")
confusionMatrix(downPP,testData[,ncol(testData)])

up_fit <- train(Class ~ ., data = up_train, 
                    method = "rpart",
                    metric = "ROC",
                    trControl = ctrl)

rose_fit <- train(Class ~ ., data = rose_train, 
                      method = "rpart",
                      metric = "ROC",
                      trControl = ctrl)

smote_fit <- train(Class ~ ., data = smote_train, 
                       method = "lda",
                       metric = "ROC",
                       trControl = ctrl)
smotePP <- predict(smote_fit, testData[,-ncol(testData)],type="raw")
confusionMatrix(smotePP,testData[,ncol(testData)])

models <- list(original = orig_fit,
                       down = down_fit,
                       up = up_fit,
                       SMOTE = smote_fit,
                       ROSE = rose_fit)

resampling <- resamples(models)

test_roc <- function(model, data) {
  library(pROC)
  roc_obj <- roc(data$Class, 
                 predict(model, data, type = "prob")[, "Ok"],
                 levels = c("Fraud", "Ok"))
}



test <- lapply(models, test_roc, data = testData)
test <- lapply(test, as.vector)
test <- do.call("rbind", test)
colnames(test) <- c("lower", "ROC", "upper")
test <- as.data.frame(test)

summary(resampling, metric = "ROC")
test
