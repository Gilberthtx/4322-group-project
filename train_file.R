## train file for neural network
options(java.parameters = c("-XX:+UseConcMarkSweepGC", "-Xmx8192m"))
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
cvIndex = createFolds(df$Class, k = 10, returnTrain = T)
cl = makeCluster(8, outfile="")
registerDoSNOW(cl)

trControl = trainControl(index = cvIndex, method = "cv", returnData = FALSE, classProbs = TRUE, summaryFunction = prSummary, allowParallel = TRUE, verboseIter = TRUE, savePredictions = TRUE)
set.seed(1)
NN_last2 = train(Class ~ ., data = df, method = "nnet", maxit = 200, metric = "AUC", trControl = trControl, verbose = TRUE, tuneGrid = expand.grid(size = sample(15:30, size = 400, replace = TRUE), decay = 10^runif(400, min = -5, 1)))
stopCluster(cl)
save(NN_last2, file = "NN_last2.Rdata")