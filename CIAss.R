###############
# Script to do Multiple Neural Networks
#
##############

library(RSNNS)

data<-read.csv("Diabetes.csv",header=F)

diabetesValues <-  data[,1:8] 
diabetesTargets <- data[,9] 
diabetesDecTargets <-  decodeClassLabels(diabetesTargets) 
data <-  splitForTrainingAndTest(diabetesValues, diabetesDecTargets, ratio = 0.3) 
data <-  normTrainingAndTestSet(data)


model <-  mlp(data$inputsTrain, data$targetsTrain, size = 4, learnFuncParams = c(0.1,0.001), maxit = 50, inputsTest = data$inputsTest, targetsTest = data$targetsTest) 
predictions <-  predict(model, data$inputsTest)

SingleTrainerror<- sqrt(sum((model$fitted.values - data$targetsTrain)^2))

SingleTesterror <-  sqrt(sum((predictions - data$targetsTest)^2))
SingleTrainerror
SingleTesterror

plotIterativeError(model) 
plotRegressionError(predictions[,2], data$targetsTest[,2], pch = 3) 
plotROC(fitted.values(model)[,2], data$targetsTrain[,2]) 
plotROC(predictions[,2], data$targetsTest[,2])

diabetes_single_train<-confusionMatrix(data$targetsTrain, fitted.values(model))
diabetes_single_test<-confusionMatrix(data$targetsTest, predictions)
diabetes_single_train
diabetes_single_test

#FineTuning

parameterGrid <-  expand.grid(c(3,5,9), c(0.003, 0.01, 0.1)) 
colnames(parameterGrid)<-c("nHidden", "learnRate") 
rownames(parameterGrid)<-paste("nnet-", apply(parameterGrid, 1, function(x) {paste(x,sep="", collapse="-")}), sep="") 

models <-apply(parameterGrid, 1, function(p) { mlp(data$inputsTrain, data$targetsTrain, size=p[1], learnFunc="Std_Backpropagation",learnFuncParams=c(p[2], 0.1), maxit=200, inputsTest=data$inputsTest,targetsTest=data$targetsTest)})


#Plotting Iterative Error rate
par(mfrow=c(4,3))
for(modInd in 1:length(models)) { plotIterativeError(models[[modInd]], main=names(models)[modInd]) }


plotIterativeError(models[[1]])
plotIterativeError(models[[2]])



trainErrors <-  data.frame(lapply(models, function(mod) { error<- sqrt(sum((mod$fitted.values - data$targetsTrain)^2))
                                                          error })) 

testErrors <-  data.frame(lapply(models, function(mod) { pred <- predict(mod,data$inputsTest) 
                                                         error <-  sqrt(sum((pred - data$targetsTest)^2)) 
                                                         error })) 

ConfuseMatrix_train<-lapply(models,function(mod){confusionMatrix(data$targetsTrain,fitted.values(mod))})

ConfuseMatrix_test<-lapply(models,function(mod){ pred<-predict(mod,data$inputsTest)
                                                 confusionMatrix(data$targetsTest,pred)})



trainErrors[which(min(trainErrors) == trainErrors)]

testErrors[which(min(testErrors) == testErrors)]

model1<-models[[which(min(testErrors) == testErrors)]]

model1


#RBF

model_rbf<- rbf(data$inputsTrain,data$targetsTrain, size=c(150), maxit=10000, initFunc="RBF_Weights", initFuncParams=c(0, 1, 0, 0.02, 0.04), learnFunc="RadialBasisLearning",learnFuncParams=c(1e-05, 0, 1e-05, 0.1, 0.8),updateFunc="Topological_Order", updateFuncParams=c(0),shufflePatterns=TRUE, linOut=FALSE)

summary(model_rbf)
plotIterativeError(model_rbf)

rbf_pred<-predict(model_rbf,data$inputsTest)
confusionMatrix(data$targetsTrain, fitted.values(model_rbf))
confusionMatrix(data$targetsTest,rbf_pred)

#Ensemble Function 
#Can be done in excel ; looks easier at the moment.

## Changing continuous labels into classes. Original Class 0 will be labeled 1 as Column 1 has Class 0 and 
## Original Class 1 will be labeled 2 as Column 2 of prediction results has Class 1.
testTargets<-encodeClassLabels(data$targetsTest, method="WTA", l=0, h=0)
modelTargets_mlp1<-encodeClassLabels(predictions, method="WTA", l=0, h=0)
modelTargets_mlp2<-encodeClassLabels(predict(models$`nnet-3-0.01`,data$inputsTest), method="WTA", l=0, h=0)
modelTargets_rbf<-encodeClassLabels(rbf_pred, method="WTA", l=0, h=0)

## Replacing 1 with correct class 0 and 2 with correct class 1.
testTarget_decode<-replace(testTargets,list=c(which(testTargets==1),which(testTargets==2)),c(0,1))
modelTargets_mlp1_decode<-replace(modelTargets_mlp1,list=c(which(modelTargets_mlp1==1),which(modelTargets_mlp1==2)),c(0,1))
modelTargets_mlp2_decode<-replace(modelTargets_mlp2,list=c(which(modelTargets_mlp2==1),which(modelTargets_mlp2==2)),c(0,1))
modelTargets_rbf_decode<-replace(modelTargets_rbf,list=c(which(modelTargets_rbf==1),which(modelTargets_rbf==2)),c(0,1))

##Creating data frame using the results of individual models.

for_voting<-data.frame(modelTargets_mlp1_decode,modelTargets_mlp2_decode,modelTargets_rbf_decode)

## This is the ensemble functions. Here we are using simple voting to create our hybrid model.
ensemble<-function(var1,var2,var3)
{
  sum<-var1+var2+var3
  if(sum<=1)
  {
    vote<-0
  } else {
    vote <- 1
  }
}

##Applying the voting to individual model results.
for_voting$vote<-mapply(ensemble,for_voting$modelTargets_mlp1_decode,for_voting$modelTargets_mlp2_decode,for_voting$modelTargets_rbf_decode)

table(testTarget_decode,for_voting$vote)

####Wine Data#####
wine = read.csv("winequality-white.csv", header=TRUE)

wineValues <-  wine[,1:11] 
wineTargets <- wine[,12] 
wineDecTargets <-  decodeClassLabels(wineTargets) 
wine <-  splitForTrainingAndTest(wineValues, wineDecTargets, ratio = 0.15) 
wine <-  normTrainingAndTestSet(wine)

#Wine initial MLP .  sqrt(nin*non) is initial number of hidden nodes

model_wine <-  mlp(wine$inputsTrain, wine$targetsTrain, size = 9, learnFuncParams = c(0.0147), maxit = 200, inputsTest = wine$inputsTest, targetsTest = wine$targetsTest) 
predictions_wine <-  predict(model_wine, wine$inputsTest)

model_wine2 <-  mlp(wine$inputsTrain, wine$targetsTrain, size = 10, learnFuncParams = c(0.0147), maxit = 500, inputsTest = wine$inputsTest, targetsTest = wine$targetsTest) 
predictions_wine2 <-  predict(model_wine2, wine$inputsTest)
write.csv(predictions_wine, file="MLP10out.csv")

model_wine3 <-  mlp(wine$inputsTrain, wine$targetsTrain, size = 15, learnFuncParams = c(0.01), maxit = 200, inputsTest = wine$inputsTest, targetsTest = wine$targetsTest) 
predictions_wine3 <-  predict(model_wine3, wine$inputsTest)
write.csv(predictions_wine, file="MLP15out.csv")

SingleTrainerror_wine<- sqrt(sum((model_wine$fitted.values - wine$targetsTrain)^2))

SingleTesterror_wine <-  sqrt(sum((predictions_wine - wine$targetsTest)^2))
                         
                         
plotIterativeError(model_wine) 
plotRegressionError(predictions_wine[,2], wine$targetsTest[,2], pch = 3) 
plotROC(fitted.values(model_wine)[,2], wine$targetsTrain[,2]) 
plotROC(predictions_wine[,2], wine$targetsTest[,2])
                         
confusionMatrix(wine$targetsTest, predictions_wine)
confusionMatrix(wine$targetsTrain, fitted.values(model_wine))

## Export Predicted results to CSV
write.csv(predictions_wine, file="MLP9out.csv")

#########Wine with bagging

wine1 = read.csv("winequality-white.csv", header=TRUE)
wine1Values<-wine1[,1:11]
wine1Targets<-wine1[,12]
wine1DecTargets<-decodeClassLabels(wineTargets) 
newWineVal<-normalizeData(wine1Values,type="center")
NormParams<-getNormParameters(newWineVal)
newWine<-data.frame(newWineVal,wine1DecTargets)

library(foreach)

set.seed(12345)
indexes<-sample(1:nrow(newWine),size=0.75*nrow(newWine))
Wine_train<-newWine[indexes,]
Wine_test<-newWine[-indexes,]

bagging<-function(training,testing,trainIP,trainOP,testIP,testOP,length_divisor=10,iterations=1000)
{
  predictions<-foreach(m=1:iterations,.combine=cbind) %do% {
    training_positions <- sample(nrow(training), size=floor((nrow(training)/length_divisor)))
    train_pos<-1:nrow(training) %in% training_positions
    model_Newwine <-  mlp(trainIP, trainOP , size = 9, learnFuncParams = c(0.0147), maxit = 200, inputsTest = testIP, targetsTest = testOP) 
    predict(model_Newwine, testIP)
  }
  rowMeans(predictions)
}

bagging(Wine_train,Wine_test,as.matrix(Wine_train[,1:11]),as.matrix(Wine_train[,12:18]),as.matrix(Wine_test[,1:11]),as.matrix(Wine_test[,12:18]),length_divisor=10,iterations=1000)
                         