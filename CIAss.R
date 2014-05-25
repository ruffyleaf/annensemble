###############
# Script to do Multiple Neural Networks
# JFK 2014
##############

library(RSNNS)

data<-read.csv("Diabetes.csv",header=F)

diabetesValues <-  data[,1:8] 
diabetesTargets <- data[,9] 
diabetesDecTargets <-  decodeClassLabels(diabetesTargets) 
data <-  splitForTrainingAndTest(diabetesValues, diabetesDecTargets, ratio = 0.15) 
data <-  normTrainingAndTestSet(data)


model <-  mlp(data$inputsTrain, data$targetsTrain, size = 4, learnFuncParams = c(0.0147), maxit = 200, inputsTest = data$inputsTest, targetsTest = data$targetsTest) 
predictions <-  predict(model, data$inputsTest)

SingleTrainerror<- sqrt(sum((model$fitted.values - data$targetsTrain)^2))

SingleTesterror <-  sqrt(sum((predictions - data$targetsTest)^2))


                         
plotIterativeError(model) 
plotRegressionError(predictions[,2], data$targetsTest[,2], pch = 3) 
plotROC(fitted.values(model)[,2], data$targetsTrain[,2]) 
plotROC(predictions[,2], data$targetsTest[,2])

confusionMatrix(data$targetsTrain, fitted.values(model))
confusionMatrix(data$targetsTest, predictions)


#FineTuning

parameterGrid <-  expand.grid(c(3,5,9), c(0.00316, 0.0147, 0.1)) 
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

model_rbf<- rbf(data$inputsTrain,data$targetsTrain, size=c(7), maxit=10000, initFunc="RBF_Weights", initFuncParams=c(0, 1, 0, 0.02, 0.04), learnFunc="RadialBasisLearning",learnFuncParams=c(1e-05, 0, 1e-05, 0.1, 0.8),updateFunc="Topological_Order", updateFuncParams=c(0),shufflePatterns=TRUE, linOut=FALSE)

summary(model_rbf)
plotIterativeError(model_rbf)

rbf_pred<-predict(model_rbf,data$inputsTest)
confusionMatrix(data$targetsTrain, fitted.values(model_rbf))
confusionMatrix(data$targetsTest,rbf_pred)

#Ensemble Function 
#Can be done in excel ; looks easier at the moment.

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

SingleTrainerror_wine<- sqrt(sum((model_wine$fitted.values - wine$targetsTrain)^2))

SingleTesterror_wine <-  sqrt(sum((predictions_wine - wine$targetsTest)^2))
                         
                         
plotIterativeError(model_wine) 
plotRegressionError(predictions_wine[,2], wine$targetsTest[,2], pch = 3) 
plotROC(fitted.values(model_wine)[,2], wine$targetsTrain[,2]) 
plotROC(predictions_wine[,2], wine$targetsTest[,2])
                         
confusionMatrix(wine$targetsTrain, fitted.values(model_wine))
confusionMatrix(wine$targetsTest, predictions_wine)



                         