library(plyr)
library(caret)
library(foreach)
library(doMC)
library(DMwR) #For smote
library(ROSE) #For rose

MyOwnDownSampling <- function(trainingData, percentageSampling){
  PositiveData <- trainingData[trainingData$Class=='Ok',]
  FraudData <- trainingData[trainingData$Class=='Fraud',]
  nbrOK <- dim(PositiveData)[1]
  nbrSamplePoints <- as.integer(percentageSampling*nbrOK, replace=FALSE)
  iu<-sample(seq(1,nbrOK),nbrSamplePoints)
  subedData <- PositiveData[iu,]
  
  newTrainData <- rbind(FraudData, subedData)
  return(newTrainData)
}
  

BaggingMethodDownSample <- function(trainData, method, B){
  registerDoMC(4)
  ctrl <- trainControl(method = "repeatedcv", repeats = 5,
                       classProbs = TRUE,
                       summaryFunction = twoClassSummary)
  listOfClassifiers = list()
  for (b in (1:B)) {
    down_train <- downSample(x = trainData[, -ncol(trainData)], y = trainData$Class)
    fit<-train(Class~., data=down_train, method=method, tuneLength=15, trControl=ctrl)
    listOfClassifiers[[b]] = fit
    print(b)
  }
  registerDoMC(NULL)
  return(listOfClassifiers)
}
BaggingMethodSmoteSample <- function(trainData, method, B){
  registerDoMC(4)
  ctrl <- trainControl(method = "repeatedcv", repeats = 5,
                       classProbs = TRUE,
                       summaryFunction = twoClassSummary)
  listOfClassifiers = list()
  for (b in (1:B)) {
    smote_train <- SMOTE(Class ~ ., data  = trainData)  
    fit<-train(Class~., data=smote_train, method=method, tuneLength=15, trControl=ctrl)
    listOfClassifiers[[b]] = fit
    print(b)
  }
  registerDoMC(NULL)
  return(listOfClassifiers)
}

PredictUsingMultipleClassifiers <- function(listOfClassifiers, testData){
  nbrClassifiers = length(listOfClassifiers)
  for (i in (1:nbrClassifiers)){
    tmpPredictor <- predict(listOfClassifiers[[i]], newdata=testData, type="raw")
    tmpFrame <- data.frame(tmpPredictor)
    if (i==1){
      listOfPreds = cbind(tmpFrame)
    } else {
      listOfPreds = cbind(listOfPreds, tmpFrame)
    }
  }
  return(listOfPreds)
}

MajorityVote <- function(classifications){
  # point on row, methods on col
  # binary count...
  classes = levels(classifications[,1])
  isFirstClass = classifications == classes[1]
  nbrMethods = dim(classifications)[2]
  meanVote = rowSums(isFirstClass) / nbrMethods
  result = factor(meanVote >= 0.5)
  mapvalues(result, from = c(FALSE, TRUE), to = c("Fraud", "Ok"))
}

test_roc <- function(model, data) {
  library(pROC)
  roc_obj <- roc(data$Class, 
                 predict(model, data, type = "prob")[, "Ok"],
                 levels = c("Fraud", "Ok"))
}

### BAGGING with downsampling
listOfClassifiers = BaggingMethodSmoteSample(trainData = trainData, 'lda', 50)
listOfClassifiers = BaggingMethodDownSample(trainData = trainData, 'lda', 50)
listOfPreds = PredictUsingMultipleClassifiers(listOfClassifiers, testData)
majVote = MajorityVote(listOfPreds)
confusionMatrix(majVote, testData[,31])

### Investigate how percentage of sampling works
ctrl <- trainControl(method = "cv", classProbs = TRUE, summaryFunction = twoClassSummary)
listOfPercentage = c(0.002, 0.01, 0.05, 0.10, 0.2, 0.5, 1)
listOfPercentage = c(0.1)
reps <- 5
MatOfKappas <- matrix(0,nrow = reps, ncol = length(listOfPercentage))
MatOfAUC <- matrix(0,nrow = reps, ncol = length(listOfPercentage))

spec_values = matrix(0,reps,length(listOfPercentage))
neg_pred_values = matrix(0,reps,length(listOfPercentage))

fractionTrain = 0.7
for (j in (1:reps)) {
  inTrainRows <- createDataPartition(card.data$Class,p=fractionTrain,list=FALSE)
  trainData <- card.data[inTrainRows,]
  testData <-  card.data[-inTrainRows,]
  for (i in (1:length(listOfPercentage))) {

    tmpTrainData <- MyOwnDownSampling(trainData, listOfPercentage[i])
    tmp_fit <- train(Class ~ ., data = tmpTrainData, 
                     method = "rpart",
                     metric = "ROC",
                     trControl = ctrl)
    tmpPP <- predict(tmp_fit, testData[,-ncol(testData)])
    tmpCnfMatrix <- confusionMatrix(tmpPP,testData$Class)
    MatOfKappas[j,i] <- tmpCnfMatrix$overall[2]
    tmpROCObj <- test_roc(tmp_fit, testData)
    MatOfAUC[j,i] <- tmpROCObj$auc
    spec_values[j,i] = tmpCnfMatrix$byClass[2]
    neg_pred_values[j,i] = tmpCnfMatrix$byClass[4]
    print(j)
  }
  print('#######')
}
table(trainData$Class)
ratioOKFraud = listOfPercentage*199021/345
boxplot(spec_values, ylab='Specificity', xlab='Ratio: OK/Fraud', names = round(ratioOKFraud,0))
boxplot(neg_pred_values, ylab='Negative prediction value', xlab='Ratio: OK/Fraud', names = round(ratioOKFraud,0))


