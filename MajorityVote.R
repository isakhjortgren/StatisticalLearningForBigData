library(plyr)
library(caret)
BaggingMethod <- function(dataMat, method, partitionFraction, B){
  ctrl<-trainControl(method="cv",summaryFunction=multiClassSummary)
  listOfClassifiers = list()
  nbrPoints = dim(datMat)[1]
  for (b in (1:B)) {
    inTrain <- sample(1:nbrPoints, nbrPoints, replace = TRUE)
    #inTrain<-createDataPartition(dataMat$Class, p=partitionFraction, list=FALSE)
    fit<-train(Class~., data=dataMat[inTrain,], method=method, tuneLength=15, trControl=ctrl)
    #pp<-predict(fit, newdata=dataMat[-inTrain,], type="raw")
    #ppFrame = data.frame(pp)
    listOfClassifiers[[b]] = fit
    print(b)
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
  mapvalues(result, from = c(FALSE, TRUE), to = c("OK", "Fraud"))
}
mpTest = data.frame(mp)
lpTest = data.frame(lp)
NBpTest = data.frame(NBp)
qpTest = data.frame(qp)
cartpTest = data.frame(cartp)
test.data = cbind(mpTest,lpTest, NBpTest, qpTest, cartpTest)

test = MajorityVote(test.data)
table(test,dataToUse$Class[-inTrain])

test

###
inTrain<-createDataPartition(dataToUse$Class, p=3/4, list=FALSE)
datMat = dataToUse[inTrain,]
ListOfClassifiers = BaggingMethod(datMat, 'knn', 3/4, 11)
listOfPreds = PredictUsingMultipleClassifiers(ListOfClassifiers, restData)#dataToUse[-inTrain,-31])
majVote = MajorityVote(listOfPreds)
table(majVote,restData$Class)#[-inTrain])


