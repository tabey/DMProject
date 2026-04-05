library(caret)
library(dplyr)
library(tidyr)
library(randomForest)
library(tree)
library(factoextra)
library(xtable)
library(UBL)
library(ggplot2)

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# setting seed for deterministic behavior
# the seed only sets once, so make sure to run this before performing any "random" action
# this includes data splitting and model training
seed <- 42
set.seed(seed)

data <- read.csv('data.csv') # reading data
sapply(data, class)

# net income flag only has one value, so we remove it
single_value_columns <- data %>% summarise(across(everything(), ~n_distinct(.) == 1))
single_value_column_names <- names(single_value_columns)[as.logical(single_value_columns[1,])]
print(single_value_column_names)
data %>% count(Net.Income.Flag)

data <- subset(data, select = -Net.Income.Flag)
data

ncol(data)
nrow(data)

data$Bankrupt. <- as.factor(data$Bankrupt.) # converting to factor for classification
data$Bankrupt.

data %>% count(Bankrupt.)

train_idx <- createDataPartition(data$Bankrupt., p=0.5, list=FALSE) # stratified 50/50 split by bankrupt column

train <- data[train_idx,] # training set
test <- data[-train_idx,] # test set

# calculate mean and standard deviation on test set
mu <- apply(subset(train, select = -Bankrupt.), 2, mean)
std <- apply(subset(train, select = -Bankrupt.), 2, sd)

# scale training set
train_scaled <- data.frame(y=train$Bankrupt.,
                           x=as.data.frame(scale(subset(train, select = -Bankrupt.), center=mu, scale=std)))
colnames(train_scaled) <- colnames(train)

# scale testing set
test_scaled <- data.frame(y=test$Bankrupt.,
                           x=as.data.frame(scale(subset(test, select = -Bankrupt.), center=mu, scale=std)))
colnames(test_scaled) <- colnames(test)

train %>% count(Bankrupt.)
test %>% count(Bankrupt.)


# single tree
tree_model <- tree(Bankrupt. ~ ., data = train_scaled)

train_pred <- predict(tree_model, train_scaled, type='class')
confusion_matrix <- confusionMatrix(train_pred, train_scaled$Bankrupt., positive='1')
print(confusion_matrix)
confusion_matrix$byClass

test_pred <- predict(tree_model, test_scaled, type='class')
confusion_matrix <- confusionMatrix(test_pred, test_scaled$Bankrupt., positive='1')
print(confusion_matrix)
confusion_matrix$byClass

# plc <- vector("list", ntrees)
# plc[[2]] <- tree(Bankrupt. ~ ., data = train_scaled)

# random forest
set.seed(seed)
ntrees <- 50
seeds <- sample.int(ntrees*2, ntrees, replace=FALSE)
trees <- vector("list", ntrees)
train_features <- subset(train_scaled, select = -Bankrupt.)
train_target <- train_scaled$Bankrupt.

for (i in 1:ntrees){
  set.seed(seeds[i])
  row_ids <- sample(nrow(train_features), nrow(train_features), replace=TRUE)
  col_ids <- sample(ncol(train_features), floor(sqrt(ncol(train_features))), replace=FALSE)
  
  temp_data <- data.frame(y=train_target[row_ids],
                          x=train_features[row_ids, col_ids])
  colnames(temp_data) <- c('Bankrupt.', names(train_features)[col_ids])
  trees[[i]] <- tree(Bankrupt. ~ ., data = temp_data)
}

train_preds <- matrix(nrow=nrow(train_scaled), ncol=ntrees)
test_preds <- matrix(nrow=nrow(test_scaled), ncol=ntrees)

test_features <- subset(test_scaled, select = -Bankrupt.)
test_target <- test_scaled$Bankrupt.

for (i in 1:ntrees){
  set.seed(seeds[i])
  row_ids <- sample(nrow(train_features), nrow(train_features), replace=TRUE)
  col_ids <- sample(ncol(train_features), floor(sqrt(ncol(train_features))), replace=FALSE)
  
  temp_data <- data.frame(y=train_target,
                          x=train_features[,col_ids])
  colnames(temp_data) <- c('Bankrupt.', names(train_features)[col_ids])
  train_preds[,i] <- predict(trees[[i]], temp_data, type='class')
  
  temp_data <- data.frame(y=test_target,
                          x=test_features[, col_ids])
  colnames(temp_data) <- c('Bankrupt.', names(test_features)[col_ids])
  test_preds[,i] <- predict(trees[[i]], temp_data, type='class')
}

train_pred <- as.factor(apply(train_preds, 1, Mode) - 1)
test_pred <- as.factor(apply(test_preds, 1, Mode) - 1)

confusion_matrix <- confusionMatrix(train_pred, train_scaled$Bankrupt., positive='1')
print(confusion_matrix)
confusion_matrix$byClass

confusion_matrix <- confusionMatrix(test_pred, test_scaled$Bankrupt., positive='1')
print(confusion_matrix)
confusion_matrix$byClass

# random undersampling
set.seed(seed)
ntrees <- 50
seeds <- sample.int(ntrees*2, ntrees, replace=FALSE)
trees <- vector("list", ntrees)
train_features <- subset(train_scaled, select = -Bankrupt.)
train_target <- train_scaled$Bankrupt.

for (i in 1:ntrees){
  set.seed(seeds[i])
  col_ids <- sample(ncol(train_features), floor(sqrt(ncol(train_features))), replace=FALSE) + 1
  
  temp_data <- RandUnderClassif(Bankrupt. ~ ., dat=train_scaled, C.perc='balance', repl=TRUE)[,c(1,col_ids)]
  trees[[i]] <- tree(Bankrupt. ~ ., data = temp_data)
}

train_preds <- matrix(nrow=nrow(train_scaled), ncol=ntrees)
test_preds <- matrix(nrow=nrow(test_scaled), ncol=ntrees)

test_features <- subset(test_scaled, select = -Bankrupt.)
test_target <- test_scaled$Bankrupt.

for (i in 1:ntrees){
  set.seed(seeds[i])
  col_ids <- sample(ncol(train_features), floor(sqrt(ncol(train_features))), replace=FALSE) + 1
  
  temp_data <- train_scaled[,c(1,col_ids)]
  #colnames(temp_data) <- c('Bankrupt.', names(train_features)[col_ids])
  train_preds[,i] <- predict(trees[[i]], temp_data, type='class')
  
  temp_data <- test_scaled[,c(1,col_ids)]
  #colnames(temp_data) <- c('Bankrupt.', names(test_features)[col_ids])
  test_preds[,i] <- predict(trees[[i]], temp_data, type='class')
}

train_pred <- as.factor(apply(train_preds, 1, Mode) - 1)
test_pred <- as.factor(apply(test_preds, 1, Mode) - 1)

confusion_matrix <- confusionMatrix(train_pred, train_scaled$Bankrupt., positive='1')
print(confusion_matrix)
confusion_matrix$byClass

confusion_matrix <- confusionMatrix(test_pred, test_scaled$Bankrupt., positive='1')
print(confusion_matrix)
confusion_matrix$byClass

# gaussian over-sampling
set.seed(seed)
ntrees <- 50
seeds <- sample.int(ntrees*2, ntrees, replace=FALSE)
trees <- vector("list", ntrees)
train_features <- subset(train_scaled, select = -Bankrupt.)
train_target <- train_scaled$Bankrupt.

for (i in 1:ntrees){
  set.seed(seeds[i])
  col_ids <- sample(ncol(train_features), floor(sqrt(ncol(train_features))), replace=FALSE) + 1
  
  temp_data <- GaussNoiseClassif(Bankrupt. ~ ., dat=train_scaled, C.perc='balance', pert=(0.05)^(1/2), repl=TRUE)[,c(1,col_ids)]
  trees[[i]] <- tree(Bankrupt. ~ ., data = temp_data)
}
                                                      
train_preds <- matrix(nrow=nrow(train_scaled), ncol=ntrees)
test_preds <- matrix(nrow=nrow(test_scaled), ncol=ntrees)

test_features <- subset(test_scaled, select = -Bankrupt.)
test_target <- test_scaled$Bankrupt.
# temp_data
for (i in 1:ntrees){
  set.seed(seeds[i])
  col_ids <- sample(ncol(train_features), floor(sqrt(ncol(train_features))), replace=FALSE) + 1
  
  temp_data <- train_scaled[,c(1,col_ids)]
  #colnames(temp_data) <- c('Bankrupt.', names(train_features)[col_ids])
  train_preds[,i] <- predict(trees[[i]], temp_data, type='class')
  
  temp_data <- test_scaled[,c(1,col_ids)]
  #colnames(temp_data) <- c('Bankrupt.', names(test_features)[col_ids])
  test_preds[,i] <- predict(trees[[i]], temp_data, type='class')
}

train_pred <- as.factor(apply(train_preds, 1, Mode) - 1)
test_pred <- as.factor(apply(test_preds, 1, Mode) - 1)

confusion_matrix <- confusionMatrix(train_pred, train_scaled$Bankrupt., positive='1')
print(confusion_matrix)
confusion_matrix$byClass

confusion_matrix <- confusionMatrix(test_pred, test_scaled$Bankrupt., positive='1')
print(confusion_matrix)
confusion_matrix$byClass

# pca-based
set.seed(seed)
ntrees <- 50
seeds <- sample.int(ntrees*2, ntrees, replace=FALSE)
trees <- vector("list", ntrees)
train_features <- subset(train_scaled, select = -Bankrupt.)
train_target <- train_scaled$Bankrupt.

results <- svd(train_features)
k <- 50 # number of principal components
rec <- results$u[,1:k] %*% diag(results$d[1:k]) %*% t(results$v[,1:k])
eps <- train_features - rec

n_samples <- ceiling(nrow(rec)/2)

for (i in 1:ntrees){
  set.seed(seeds[i])
  col_ids <- sample(ncol(train_features), floor(sqrt(ncol(train_features))), replace=FALSE)
  rnames <- c(sample(which(train_scaled$Bankrupt. == 0), n_samples, replace=TRUE),
              sample(which(train_scaled$Bankrupt. == 1), n_samples, replace=TRUE))
  nnames <- c(sample(which(train_scaled$Bankrupt. == 0), n_samples, replace=TRUE),
              sample(which(train_scaled$Bankrupt. == 1), n_samples, replace=TRUE))
  temp_data <- data.frame(y=train_target[rnames]
                          ,x=rec[rnames,col_ids] + eps[nnames,col_ids])
  colnames(temp_data) <- c('Bankrupt.', names(test_features)[col_ids])
  trees[[i]] <- tree(Bankrupt. ~ ., data = temp_data)
}

train_preds <- matrix(nrow=nrow(train_scaled), ncol=ntrees)
test_preds <- matrix(nrow=nrow(test_scaled), ncol=ntrees)

test_features <- subset(test_scaled, select = -Bankrupt.)
test_target <- test_scaled$Bankrupt.

for (i in 1:ntrees){
  set.seed(seeds[i])
  col_ids <- sample(ncol(train_features), floor(sqrt(ncol(train_features))), replace=FALSE) + 1
  
  temp_data <- train_scaled[,c(1,col_ids)]
  #colnames(temp_data) <- c('Bankrupt.', names(train_features)[col_ids])
  train_preds[,i] <- predict(trees[[i]], temp_data, type='class')
  
  temp_data <- test_scaled[,c(1,col_ids)]
  #colnames(temp_data) <- c('Bankrupt.', names(test_features)[col_ids])
  test_preds[,i] <- predict(trees[[i]], temp_data, type='class')
}

train_pred <- as.factor(apply(train_preds, 1, Mode) - 1)
test_pred <- as.factor(apply(test_preds, 1, Mode) - 1)

confusion_matrix <- confusionMatrix(train_pred, train_scaled$Bankrupt., positive='1')
print(confusion_matrix)
confusion_matrix$byClass

confusion_matrix <- confusionMatrix(test_pred, test_scaled$Bankrupt., positive='1')
print(confusion_matrix)
confusion_matrix$byClass

# svd
results <- svd(subset(train_scaled, select=-Bankrupt.))
var_exp <- cumsum(results$d^2) / sum(results$d^2)

plot(results$d^2 / sum(results$d^2),
     xlab='# of principal components',
     ylab='explained variance',
     type="l",
     lwd=2,
     main='Scree plot')
abline(v=50, lwd=2, lty="dashed", col='red')

scree_data <- data.frame(
  component = 1:length(results$d),
  explained_variance = results$d^2 / sum(results$d^2)
)

ggplot(scree_data, aes(x = component, y = explained_variance)) +
  geom_area(fill = "steelblue", alpha = 0.4) +  # Add filled area with transparency
  geom_line(size = 1, color = "steelblue") +
  geom_vline(xintercept = 50, linetype = "dashed", color = "red", size = 1) +
  labs(
    x = "Principal Component",
    y = "Explained Variance",
    title = "Scree Plot",
    subtitle = "Explained Variance by Principal Components"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5),
    axis.title = element_text(face = "bold")
  ) + theme(text=element_text(size=15))

k <- 50 # number of principal components
rec <- results$u[,1:k] %*% diag(results$d[1:k]) %*% t(results$v[,1:k])
eps <- subset(train_scaled, select=-Bankrupt.) - rec
rec
rec + eps
# 
# X <- data.frame(y=data$Bankrupt., x=results$u[,1:k] %*% diag(results$d[1:k]) %*% t(results$v[,1:k]))
# colnames(X) <- colnames(data)
# 
# X
# # sample.int(100, ntrees) to get array of random seeds i guess?
# set.seed(42)
# sample.int(100, 10, replace=FALSE)
# 
# floor(sqrt(ncol(data))) # for feature subsampling
