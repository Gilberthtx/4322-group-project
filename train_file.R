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
trControl = trainControl(index = cvIndex, method = "cv", classProbs = TRUE, summaryFunction = prSummary, allowParallel = TRUE, verboseIter = TRUE, indexFinal = 1:200)

cl = makeCluster(4, outfile="")
registerDoSNOW(cl)

set.seed(1)
fit = train(Class ~ ., data = df, method = "ranger", metric = "AUC", trControl = trControl, tuneGrid = expand.grid(mtry = 5, splitrule = "gini", min.node.size=10))
stopCluster(cl)
fit

fit = train(Class ~ ., data = df, method = "treebag", metric = "AUC", trControl = trControl, tuneGrid = expand.grid(mtry = 5, splitrule = "gini", min.node.size=10))
