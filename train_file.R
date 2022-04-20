## train file
df = read.csv('F:/OneDrive - University Of Houston/Study/11th semester/MATH 4322/GP/creditcard.csv')
df$Time = NULL
df$Amount = scale(df$Amount)
df$Class = factor(df$Class, levels = rev(levels(factor(df$Class))))
levels(df$Class) = c("Fraud", "NonFraud")
library(ranger)
library(rpart)
library(caret) # for cross validation training and hyper-parameter tuning
library(doSNOW) # for parallel computing
library(PRROC) # for evaluation metrics

set.seed(1)
cvIndex = createFolds(df$Class, k = 5, returnTrain = T)
trControl = trainControl(index = cvIndex, method = "cv", classProbs = TRUE, summaryFunction = prSummary, allowParallel = TRUE, verboseIter = TRUE, savePredictions = TRUE, search = "random")
cl = makeCluster(7, outfile="")
registerDoSNOW(cl)

set.seed(1)
NN = train(Class ~ ., data = df, method = "nnet", tuneLength = 50, metric = "AUC", trControl = trControl, verbose = TRUE)
stopCluster(cl)
NN