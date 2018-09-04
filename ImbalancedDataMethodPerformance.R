library(caret)
library(data.table)
library(tictoc)
library(plyr)
library(GGally)
library(DMwR) #For smote

test_roc <- function(model, data) {
  library(pROC)
  roc_obj <- roc(data$Class, 
                 predict(model, data[,-ncol(data)], type = "prob")[, "NoHD"],
                 levels = c("NoHD", "HD"))
}

card.data <- fread("creditcard.csv",header=TRUE,sep=",",na.strings = '?')
card.data <- card.data[,2:31]
#card.data <- apply(card.data, 2, function(x) as.numeric(x))
card.data <- as.data.frame(card.data)
#card.data <- na.omit(card.data)
card.data$Class <- as.factor(card.data$Class)
levels(card.data$Class) <- c("Ok", "Fraud")
head(card.data)


heart.data <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",header=FALSE,sep=",",na.strings = '?')
names(heart.data) <- c( "age", "sex", "cp", "trestbps", "chol","fbs", "restecg",
                        "thalach","exang", "oldpeak","slope", "ca", "thal", "num")
head(heart.data,3)
heart.data$num[heart.data$num > 0] <- 1
heart.data$num[heart.data$num == 0] <- "NoHD"
heart.data$num[heart.data$num == 1] <- "HD"
barplot(table(heart.data$num),
        main="Fate", col="black")
# change a few predictor variables from integer to factors (make dummies)
chclass <-c("numeric","factor","factor","numeric","numeric","factor","factor","numeric","factor","numeric","factor","factor","factor","factor")

convert.magic <- function(obj,types){
  for (i in 1:length(obj)){
    FUN <- switch(types[i],character = as.character, 
                  numeric = as.numeric, 
                  factor = as.factor)
    obj[,i] <- FUN(obj[,i])
  }
  obj
}
heart.data <- na.omit(heart.data)
heart.data <- convert.magic(heart.data,chclass)
heart = heart.data #add labels only for plot
levels(heart$num) = c("No disease","Disease")
levels(heart$sex) = c("female","male","")

## 

colnames(heart.data)[ncol(heart.data)] = 'Class'
fractionTrain = 0.7

table(trainData$num)
table(testData$num)

ctrl <- trainControl(method = "cv", classProbs = TRUE,verboseIter = FALSE, summaryFunction = multiClassSummary)

listOfMethods = c('lda','mda','knn','rpart','ranger','regLogistic')
nbrReps = 5
accMatrix = matrix(0,nbrReps,length(listOfMethods))
kappaMatrix = matrix(0,nbrReps,length(listOfMethods))
sensMatrix = matrix(0,nbrReps,length(listOfMethods))
specMatrix = matrix(0,nbrReps,length(listOfMethods))
aucMatrix = matrix(0, nbrReps,length(listOfMethods))
neg_pred_values = matrix(0, nbrReps,length(listOfMethods))

tic()
for (i_rep in 1:nbrReps){
  print('rep:')
  print(i_rep)
  tic()
  inTrainRows <- createDataPartition(card.data$Class,p=fractionTrain,list=FALSE)
  trainData <- card.data[inTrainRows,]
  smote_train <- SMOTE(Class ~ ., data  = trainData)   
  testData <-  card.data[-inTrainRows,]
  print('train:')
  for (i_method in 1:length(listOfMethods)){
    train_method <- listOfMethods[i_method]
    print(train_method)
    fit <- train(Class ~ ., data = smote_train, method = train_method, trControl = ctrl)
    prediction <- predict(fit, testData[,-ncol(testData)])
    cnf <-confusionMatrix(prediction, testData[,ncol(testData)])
    #ROCObj = test_roc(fit, testData)
    
    #aucMatrix[i_rep,i_method] = ROCObj$auc
    accMatrix[i_rep,i_method] = cnf$overall[1]
    kappaMatrix[i_rep,i_method] = cnf$overall[2]
    sensMatrix[i_rep,i_method] = cnf$byClass[1]
    specMatrix[i_rep,i_method] = cnf$byClass[2]
    neg_pred_values[i_rep,i_method] = cnf$byClass[4]
  }
  toc()
}
toc()

ggpairs(data=heart.data, # data.frame with variables
        #columns=2:6, # columns to plot, default to all.
        title="Synthetic data", # title of the plot
        mapping = ggplot2::aes(color = Class))

method_names = c('lda','mda','knn','cart','random\nforest','logistic\nregression')
#boxplot(accMatrix, names = method_names, ylab = 'Accuracy')
#boxplot(kappaMatrix, names = method_names, ylab = 'Kappa')
#boxplot(aucMatrix, names = method_names, ylab = 'AUC')
#boxplot(sensMatrix, names = method_names, ylab = 'Sensitivity')
boxplot(specMatrix, names = method_names, ylab = 'Specificity')
boxplot(neg_pred_values, names = method_names, ylab = 'Negative predictive value')
#save.image(file='allRegMEthods.RData')


###################################
### create bagging ----------------
BaggingMethod <- function(dataMat, method1, partitionFraction, B){
  ctrl<-trainControl(method="cv", classProbs = TRUE,summaryFunction=multiClassSummary)
  listOfClassifiers = list()
  nbrPoints = dim(dataMat)[1]
  for (b in (1:B)) {
    inTrain<-createDataPartition(dataMat$Class, p=partitionFraction, list=FALSE)
    fit<-train(Class~., data=dataMat[inTrain,], method=method1, tuneLength=15, trControl=ctrl)
    listOfClassifiers[[b]] = fit
  }
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
  mapvalues(result, from = c(FALSE, TRUE), to = c("NoHD", "HD"))
}
bagPartitionFraction = 0.5
B = 5
listOfMethods = c('pda','nb','knn','rpart','ranger')
nbrReps = 2
accMatrix = matrix(0,nbrReps,length(listOfMethods))
kappaMatrix = matrix(0,nbrReps,length(listOfMethods))
sensMatrix = matrix(0,nbrReps,length(listOfMethods))
specMatrix = matrix(0,nbrReps,length(listOfMethods))
aucMatrix = matrix(0, nbrReps,length(listOfMethods))

tic()
for (i_rep in 1:nbrReps){
  print('rep:')
  print(i_rep)
  tic()
  inTrainRows <- createDataPartition(heart.data$Class,p=fractionTrain,list=FALSE)
  trainData <- heart.data[inTrainRows,]
  testData <-  heart.data[-inTrainRows,]
  print('train:')
  for (i_method in 1:length(listOfMethods)){
    train_method <- listOfMethods[i_method]
    print(train_method)

    baggedM = BaggingMethod(trainData, train_method, bagPartitionFraction, B)
    predBAg = PredictUsingMultipleClassifiers(baggedM, testData)
    majvote_bag = MajorityVote(predBAg)

    cnf <-confusionMatrix(majvote_bag, testData[,ncol(testData)])
    ROCObj = test_roc(fit, testData)
    
    aucMatrix[i_rep,i_method] = ROCObj$auc
    accMatrix[i_rep,i_method] = cnf$overall[1]
    kappaMatrix[i_rep,i_method] = cnf$overall[2]
    sensMatrix[i_rep,i_method] = cnf$byClass[1]
    specMatrix[i_rep,i_method] = cnf$byClass[2]
  }
  toc()
}
toc()
c('lda','knn','rpart','ranger')
method_names = c('lda','knn','cart','random\nforest')
load('bagTestCreditCard.RData')
boxplot(accMatrix, names = method_names, ylab = 'Accuracy')
boxplot(kappaMatrix, names = method_names, ylab = 'Kappa')
boxplot(aucMatrix, names = method_names, ylab = 'AUC')
boxplot(sensMatrix, names = method_names, ylab = 'Sensitivity')
boxplot(specMatrix, names = method_names, ylab = 'Specificity')
boxplot(neg_pred_values, names = method_names, ylab = 'Negative predictive value')


