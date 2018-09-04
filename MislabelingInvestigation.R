library(irlba)
# fast svd for large data sets
library(plot3D)
library(rgl)
library(caret)
setwd("~/Documents/Skola/StatisticalLearningForBigData/FinalProject")
load('TCGAdata.RData')
load('RandomMisLab.RData')
TCGA <- sweep(TCGA,2,colMeans(TCGA))
ss<-irlba(TCGA,10,10)


componentsVec <- c(1:10)
dataFirst10pc <- ss$u[,componentsVec]
plot3d(dataFirst10pc[,1:3],col=as.numeric(as.factor(TCGAclassstr)))
legend3d("topright", legend = paste('Type', c(unique(TCGAclassstr))[sort.list(as.numeric(as.factor(unique(TCGAclassstr))))]), pch = 5, col=seq(1,6), cex=1, inset=c(0.02))
dataFirst10pc <- as.data.frame(dataFirst10pc)


fullDataSet <- cbind(dataFirst10pc, TCGAclassstr)
head(fullDataSet)
fractionTrain = 0.7
inTrainRows <- createDataPartition(fullDataSet$TCGAclassstr,p=fractionTrain,list=FALSE)
trainData <- fullDataSet[inTrainRows,]

testData <-  fullDataSet[-inTrainRows,componentsVec]
testClasses <- fullDataSet[-inTrainRows,ncol(fullDataSet)]
table(trainData$TCGAclassstr)
ctrl <- trainControl(method = "cv", classProbs = TRUE,verboseIter = FALSE, summaryFunction = multiClassSummary)

# train with no misslabeled data
train_method <- 'lda'
lda_fit <- train(TCGAclassstr ~ ., data = trainData, method = train_method,  metric = "Kappa", trControl = ctrl)
lda_prediction <- predict(lda_fit, testData)
lda_cnf <-confusionMatrix(lda_prediction, testClasses)

train_method <- 'ranger'
rf_fit <- train(TCGAclassstr ~ ., data = trainData, method = train_method,  metric = "Kappa", trControl = ctrl)
rf_prediction <- predict(rf_fit, testData)
rf_cnf <- confusionMatrix(rf_prediction, testClasses)

train_method <- 'mda'
mda_fit <- train(TCGAclassstr ~ ., data = trainData, method = train_method,  metric = "Kappa", trControl = ctrl)
mda_prediction <- predict(mda_fit, testData)
mda_cnf <- confusionMatrix(mda_prediction, testClasses)



###################
# Random Misslabeling
###################
GenerateRandomMisslabeling <- function(data, fractionMislab) {
  # function inserts random labels
  nbrPoints <- nrow(data)
  nbrRandomMislab <- as.integer(nbrPoints*fractionMislab)
  current_labels <- levels(data$TCGAclassstr)
  random_label_array <- sample(current_labels, nbrRandomMislab, replace=TRUE)
  random_position_array <- sample(c(1:nbrPoints), nbrRandomMislab, replace=FALSE)
  tmp_data <- data.frame(data)
  tmp_data[random_position_array, ncol(data)] <- random_label_array
  return(tmp_data)
}

list_off_frac_mislab = c(0:10)/10
result_matrix = matrix(0,11,6) # lda acc, lda kappa, rf acc, rf kappa, mda acc, mda kappa
# investigate random mislabeled data

nbr_mean_runs = 5
for (i in 1:11) {
  # generate avg result
  print('fraction mislabeled')
  fractionMisLab = list_off_frac_mislab[i]
  print(fractionMisLab)
  print('iteration:')
  for (j in 1:nbr_mean_runs) {
    print(j)
    inTrainRows <- createDataPartition(fullDataSet$TCGAclassstr,p=fractionTrain,list=FALSE)
    trainData <- fullDataSet[inTrainRows,]
    
    testData <-  fullDataSet[-inTrainRows,componentsVec]
    testClasses <- fullDataSet[-inTrainRows,ncol(fullDataSet)]
    
    
    trainData_mislab <- GenerateRandomMisslabeling(trainData, fractionMisLab)
    #plot3d(trainData_mislab[,1:3],col=as.numeric(as.factor(trainData_mislab$TCGAclassstr)))
    
    train_method <- 'lda'
    lda_fit_mis <- train(TCGAclassstr ~ ., data = trainData_mislab, method = train_method,  metric = "Kappa", trControl = ctrl)
    lda_prediction_mis <- predict(lda_fit_mis, testData)
    lda_cnf_mis <- confusionMatrix(lda_prediction_mis, testClasses)
    
    train_method <- 'ranger'
    rf_fit_mis <- train(TCGAclassstr ~ ., data = trainData_mislab, method = train_method,  metric = "Kappa", trControl = ctrl)
    rf_prediction_mis <- predict(rf_fit_mis, testData)
    rf_cnf_mis <- confusionMatrix(rf_prediction_mis, testClasses)
    
    train_method <- 'mda'
    mda_fit_mis <- train(TCGAclassstr ~ ., data = trainData_mislab, method = train_method,  metric = "Kappa", trControl = ctrl)
    mda_prediction_mis <- predict(mda_fit_mis, testData)
    mda_cnf_mis <- confusionMatrix(mda_prediction_mis, testClasses)
    result_matrix[i,2] <- mda_cnf_mis$overall[2] + result_matrix[i,2]
    result_matrix[i,1] <- mda_cnf_mis$overall[1] + result_matrix[i,1]
    
    result_matrix[i,4] <- lda_cnf_mis$overall[2] + result_matrix[i,4]
    result_matrix[i,3] <- lda_cnf_mis$overall[1] + result_matrix[i,3]
    
    result_matrix[i,6] <- rf_cnf_mis$overall[2] + result_matrix[i,6]
    result_matrix[i,5] <- rf_cnf_mis$overall[1] + result_matrix[i,5]
  }
  
}
result_matrix <- result_matrix/nbr_mean_runs

plot(list_off_frac_mislab, result_matrix[,1], xlim=c(-0.05, 1.05), ylim=c(-.05, 1.05), col = 'red', lty=1, type="b", xlab='fraction of data randomly labeled', ylab="")
points(list_off_frac_mislab, result_matrix[,2], col = 'red',lty=2, type="b")
points(list_off_frac_mislab, result_matrix[,3], col = 'blue',lty=1, type="b")
points(list_off_frac_mislab, result_matrix[,4], col = 'blue',lty=2, type="b")
points(list_off_frac_mislab, result_matrix[,5], col = 'green',lty=1, type="b")
points(list_off_frac_mislab, result_matrix[,6], col = 'green',lty=2, type="b")
legend(0, 0.5, legend=c("LDA accuracy", "LDA kappa", "RF accuracy", "RF kappa", "MDA accuracy", "MDA kappa"), col=c("red", "red", "blue", "blue", "green", "green"), lty=c(1,2,1,2,1,2))



##############
# Systematic Mislabeling
##############

table(trainData$TCGAclassstr)
radius_array = c(0,0.7,0.8,0.95,1, 1.05,1.1,1.2,1.3,1.4,1.5,1.6,1.8,2,2.2) # 14 rads
radius_array = c(1.1) # 14 rads
result_matrix = matrix(0,length(radius_array),7) # fraction, lda ba BC, lda ba OV, rf ba BC, rf ba OV, mda ba BC, mda ba OV
nbrReps = 5
for (j in 1:length(radius_array)){
  print('radius iteration:')
  print(j)
  print('rep: ')
  for (k in 1:nbrReps) {
    print(k)
    inTrainRows <- createDataPartition(fullDataSet$TCGAclassstr,p=fractionTrain,list=FALSE)
    trainData <- fullDataSet[inTrainRows,]
    
    testData <-  fullDataSet[-inTrainRows,componentsVec]
    testClasses <- fullDataSet[-inTrainRows,ncol(fullDataSet)]
    ## create new data set
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
    
    
    # train on new data
    train_method <- 'lda'
    lda_fit_mis <- train(TCGAclassstr ~ ., data = trainData_mislab, method = train_method,  metric = "Kappa", trControl = ctrl)
    lda_prediction_mis <- predict(lda_fit_mis, testData)
    lda_cnf_mis <- confusionMatrix(lda_prediction_mis, testClasses)
    
    train_method <- 'ranger'
    rf_fit_mis <- train(TCGAclassstr ~ ., data = trainData_mislab, method = train_method,  metric = "Kappa", trControl = ctrl)
    rf_prediction_mis <- predict(rf_fit_mis, testData)
    rf_cnf_mis <- confusionMatrix(rf_prediction_mis, testClasses)
    
    train_method <- 'mda'
    mda_fit_mis <- train(TCGAclassstr ~ ., data = trainData_mislab, method = train_method,  metric = "Kappa", trControl = ctrl)
    mda_prediction_mis <- predict(mda_fit_mis, testData)
    mda_cnf_mis <- confusionMatrix(mda_prediction_mis, testClasses)
    result_matrix[j,3] <- mda_cnf_mis$overall[2] + result_matrix[j,3]
    result_matrix[j,2] <- mda_cnf_mis$overall[1] + result_matrix[j,2]
    
    result_matrix[j,5] <- lda_cnf_mis$overall[2] + result_matrix[j,5]
    result_matrix[j,4] <- lda_cnf_mis$overall[1] + result_matrix[j,4]
    
    result_matrix[j,7] <- rf_cnf_mis$overall[2] + result_matrix[j,7]
    result_matrix[j,6] <- rf_cnf_mis$overall[1] + result_matrix[j,6]
  }
  
  
}
result_matrix[,2:7] = result_matrix[,2:7]/nbrReps

#misLabTrain = data.frame(trainData)
#misLabTrain[logicalBC, 11] = BCdata_lab
#table(misLabTrain$TCGAclassstr)
#plot3d(misLabTrain[,1:3],col=as.numeric(as.factor(misLabTrain$TCGAclassstr)))
plot(result_matrix[,1], result_matrix[,2], xlim=c(-0.05, .95), ylim=c(.5, 1.05), col = 'red', lty=1, type="b", xlab='fraction of class BC labeled as OV', ylab="balanced accuracy")
points(result_matrix[,1], result_matrix[,3], col = 'red',lty=2, type="b")
points(result_matrix[,1], result_matrix[,4], col = 'blue',lty=1, type="b")
points(result_matrix[,1], result_matrix[,5], col = 'blue',lty=2, type="b")
points(result_matrix[,1], result_matrix[,6], col = 'green',lty=1, type="b")
points(result_matrix[,1], result_matrix[,7], col = 'green',lty=2, type="b")
legend(0, 0.8, legend=c("LDA - class BC", "LDA - class OV", "RF - class BC", "RF - class OV", "MDA - class BC", "MDA - class OV"), col=c("red", "red", "blue", "blue", "green", "green"), lty=c(1,2,1,2,1,2))


sum(predict(mda_fit_mis, BCdata[BCdata_lab == "OV",]) == "OV")
sum(BCdata_lab == "OV")




