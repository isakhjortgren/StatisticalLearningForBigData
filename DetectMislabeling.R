library(caret)


BaggingMethod <- function(dataMat, method, partitionFraction, B){
  ctrl<-trainControl(method="cv", classProbs = TRUE,summaryFunction=multiClassSummary)
  listOfClassifiers = list()
  nbrPoints = dim(dataMat)[1]
  for (b in (1:B)) {
    inTrain<-createDataPartition(dataMat$TCGAclassstr, p=partitionFraction, list=FALSE)
    fit<-train(TCGAclassstr~., data=dataMat[inTrain,], method=method,  metric = "Kappa", tuneLength=15, trControl=ctrl)
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

CalculateVoteProb <- function(listOfPreds){
  nobs = nrow(listOfPreds)
  nclassif = ncol(listOfPreds)
  prob =  data.frame(matrix(0,nobs,6))
  colNameArr <- c("BC",  "GBM", "KI",  "LU",  "OV",  "U")
  colnames(prob) <- colNameArr
  
  for (i in 1:length(colNameArr)){
    currentClass = colNameArr[i]
    for (j in 1:nobs){
      for (k in 1:nclassif){
        if (listOfPreds[j,k] == currentClass) {
          prob[j,i] = prob[j,i] + 1
        }
      }
    }
  }
  prob = prob / nclassif
  return(prob)
}

CalculateMajVote <- function(vote_prob_f){
  maxCols = max.col(vote_prob_f)
  nobs = nrow(vote_prob_f)
  votes = c(1:nobs)
  votes[maxCols == 1] = "BC"
  votes[maxCols == 2] = "GBM"
  votes[maxCols == 3] = "KI"
  votes[maxCols == 4] = "LU"
  votes[maxCols == 5] = "OV"
  votes[maxCols == 6] = "U"
  return(factor(votes))
}
CalculateMajVoteProb <- function(vote_prob_f){
  maxCols = max.col(vote_prob_f)
  nobs = nrow(vote_prob_f)
  votes_prob = c(1:nobs)
  for (i in 1:nobs){
    votes_prob[i] = vote_prob_f[i, maxCols[i]]
  }
  return(votes_prob)
}



## create mislabeled data:
library(irlba)
# fast svd for large data sets
library(plot3D)
library(rgl)
library(caret)
setwd("~/Documents/Skola/StatisticalLearningForBigData/FinalProject")
load('TCGAdata.RData')
ss<-irlba(TCGA,15,15)


componentsVec <- c(1:10)
dataFirst10pc <- ss$u[,componentsVec]
dataFirst10pc <- as.data.frame(dataFirst10pc)


fullDataSet <- cbind(dataFirst10pc, TCGAclassstr)
head(fullDataSet)
fractionTrain = 0.7
inTrainRows <- createDataPartition(fullDataSet$TCGAclassstr,p=fractionTrain,list=FALSE)
trainData <- fullDataSet[inTrainRows,]

testData <-  fullDataSet[-inTrainRows,componentsVec]
testClasses <- fullDataSet[-inTrainRows,ncol(fullDataSet)]





# generate randomly mislabeled data
data = trainData
list_of_fractions = c(1:10)/10
result_fraction_mislab = matrix(0,10,2) # frac mis lab found, frac falsley acc
for (i in 1:10){
  fractionMislab = list_of_fractions[i]
  nbrPoints <- nrow(data)
  nbrRandomMislab <- as.integer(nbrPoints*fractionMislab)
  current_labels <- levels(data$TCGAclassstr)
  random_label_array <- sample(current_labels, nbrRandomMislab, replace=TRUE)
  random_position_array <- sample(c(1:nbrPoints), nbrRandomMislab, replace=FALSE)
  tmp_data <- data.frame(data)
  tmp_data[random_position_array, ncol(data)] <- random_label_array
  
  # create a list of weak learners
  print('create ensamble for fraction:')
  print(fractionMislab)
  list_classif = BaggingMethod(tmp_data, 'lda', 0.2, 10)
  pred_on_train = PredictUsingMultipleClassifiers(list_classif, data[,-ncol(data)])
  vote_on_train = CalculateVoteProb(pred_on_train)
  maj_vote_train = CalculateMajVote(vote_on_train)
  maj_vote_prob_train = CalculateMajVoteProb(vote_on_train)
  all_lab_incl_mislab = tmp_data[, ncol(tmp_data)]
  
  
  alpha = 0.7
  predicted_mislabelings = (maj_vote_train != all_lab_incl_mislab) & (maj_vote_prob_train > alpha)
  predicted_actual_mislabeling = predicted_mislabelings[tmp_data[,ncol(data)] != data[,ncol(data)]]
  predicted_real_as_mislabeled = predicted_mislabelings[-random_position_array]
  
  fraction_of_mis_lab_found = sum(predicted_actual_mislabeling)/length(predicted_actual_mislabeling)
  fraction_falseley_accused = sum(predicted_real_as_mislabeled)/length(predicted_real_as_mislabeled)
  result_fraction_mislab[i,1] = fraction_of_mis_lab_found
  result_fraction_mislab[i,2] = fraction_falseley_accused
}

plot(list_of_fractions, result_fraction_mislab[,1], xlim=c(-0.05, 1.05), ylim=c(-.05, 1.05), col = 'red', lty=1, type="b", xlab='fraction of data randomly labeled', ylab="")
points(list_of_fractions, result_fraction_mislab[,2], col = 'blue',lty=2, type="b")
legend(0, 0.5, legend=c("Fraction of mislabeled observations found", "Fraction of correct labels classified as mislabeled"), col=c("red", "blue"), lty=c(1,2))



## Systematic mislabeling
## create new data set
radius_array = c(0.7,0.8,0.95,1, 1.05,1.1,1.2,1.3,1.4,1.5,1.6,1.8,2,2.2) # 14 rads
result_matrix = matrix(0,length(radius_array),7) # fraction, frac mis lab found, frac falsley acc
for (j in 1:length(radius_array)){
  radius = radius_array[j]
  logicalBC = trainData$TCGAclassstr == 'BC'
  BCdata = trainData[logicalBC,1:10]
  BCdata_lab = trainData[logicalBC,11]
  mean_vec = colMeans(BCdata)
  std_vec = apply(BCdata, 2, sd)
  nbrChanges = 0
  for (i in 1:nrow(BCdata)){
    tmpx = BCdata[i,]
    tmp = tmpx - mean_vec
    tmp_abs = abs(tmp)
    if (all(tmp_abs<(radius*std_vec))){
      BCdata_lab[i] = 'OV'
      nbrChanges = nbrChanges +1
    }
  }
  result_matrix[j,1] = nbrChanges/nrow(BCdata)
  trainData_mislab = data.frame(trainData)
  trainData_mislab[logicalBC, 11] = BCdata_lab
  
  print('create ensamble for fraction:')
  print(result_matrix[j,1])
  ### start creating a bagging methods to detect mislabeling
  list_classif = BaggingMethod(trainData_mislab, 'lda', 0.2, 10)
  pred_on_train = PredictUsingMultipleClassifiers(list_classif, trainData_mislab[,-ncol(trainData_mislab)])
  vote_on_train = CalculateVoteProb(pred_on_train)
  maj_vote_train = CalculateMajVote(vote_on_train)
  maj_vote_prob_train = CalculateMajVoteProb(vote_on_train)
  all_lab_incl_mislab = trainData_mislab[, ncol(trainData_mislab)]
  
  alpha = 0.7
  predicted_mislabelings = (maj_vote_train != all_lab_incl_mislab) & (maj_vote_prob_train > alpha) # this vector is a logical vector where 1 is a predicted mislabel
  ##
  
  #compare labels
  predicted_actual_mislabeling = predicted_mislabelings[trainData_mislab[,ncol(trainData_mislab)] != trainData[,ncol(trainData)]]
  predicted_real_as_mislabeled = predicted_mislabelings[trainData_mislab[,ncol(trainData_mislab)] == trainData[,ncol(trainData)]]
  fraction_of_mis_lab_found = sum(predicted_actual_mislabeling)/length(predicted_actual_mislabeling)
  fraction_falseley_accused = sum(predicted_real_as_mislabeled)/length(predicted_real_as_mislabeled)
  result_matrix[j,2] = fraction_of_mis_lab_found
  result_matrix[j,3] = fraction_falseley_accused
  print('frac mislab found')
  print(fraction_of_mis_lab_found)
  
}
confusionMatrix(maj_vote_train,trainData_mislab[,ncol(trainData_mislab)] )
plot(result_matrix[,1], result_matrix[,2], xlim=c(-0.05, 1.05), ylim=c(-.05, 1.05), col = 'red', lty=1, type="b", xlab='fraction of class BC labeled as OV', ylab="")
points(result_matrix[,1], result_matrix[,3], col = 'blue',lty=2, type="b")
legend(0.5, 1, legend=c("Fraction of mislabeled observations found", "Fraction of correct labels classified as mislabeled"), col=c("red", "blue"), lty=c(1,2))
