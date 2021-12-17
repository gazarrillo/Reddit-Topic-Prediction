# library -----------------------------------------------------------------

library(data.table)
library(caret)
library(xgboost)

# load train and test -----------------------------------------------------

train <- fread('./project/volume/data/interim/train.csv')
test <- fread('./project/volume/data/interim/test.csv')

# split y and x -----------------------------------------------------------

id.train <- train$id
id.test <- test$id
train$id <- NULL
test$id <- NULL

y.train <- as.integer(as.factor(train$topic))-1
topics <- sort(unique(train$topic))
num.class <- length(topics)

dummies <- dummyVars(topic~., data=train)
x.train <- predict(dummies, newdata=train)
x.test <- predict(dummies, newdata=test)

# make objects for XGBoost ------------------------------------------------

dtrain <- xgb.DMatrix(x.train, label=y.train, missing=NA)
dtest <- xgb.DMatrix(x.test, missing=NA)

tuning_log <- NULL

# DO NOT RUN ABOVE WHEN PARAMETER TUNING
# parameter tuning using cross-validation ---------------------------------

### details of parameters can be found here
### https://xgboost.readthedocs.io/en/latest/parameter.html

### set tuning parameters
params <- list(booster          = 'gbtree',
               tree_method      = 'hist',
               objective        = 'multi:softprob',
               num_class        = num.class,
               # complexity
               max_depth        = 5,
               min_child_weight = 1,
               gamma            = 0.1,
               # diversity
               eta              = 0.1,
               subsample        = 1,
               colsample_bytree = 1
)

### cross-validation
set.seed(1)
XGBm <- xgb.cv(params                = params,
               data                  = dtrain,
               missing               = NA,
               nfold                 = 5,
               # diversity
               nrounds               = 10000,
               early_stopping_rounds = 25,
               # whether it shows error at each round
               verbose               = 1
)

### save tuning parameters
tuning_new <- data.table(t(params))

### save the best number of rounds
best_nrounds <- unclass(XGBm)$best_iteration
tuning_new$best_nrounds <- best_nrounds

### save the test set error
error_cv <- unclass(XGBm)$evaluation_log[best_nrounds,]$test_mlogloss_mean
tuning_new$error_cv <- error_cv

### keep the tuning log
tuning_log <- rbind(tuning_log,tuning_new)
tuning_log

### try different parameters following the order below
### 1. max_depth
### 2. eta
### 3. min_child_weight
### 4. gamma
### 5. subsample
### 6. colsample_bytree

# fit a full model --------------------------------------------------------

### find the parameterss with the smallest cv error
tuning_best <- tuning_log[which.min(tuning_log$error_cv),]

### call back the best parameters
params <- list(booster          = 'gbtree',
               tree_method      = 'hist',
               objective        = 'multi:softprob',
               num_class        = num.class,
               max_depth        = tuning_best$max_depth,
               min_child_weight = tuning_best$min_child_weight,
               gamma            = tuning_best$gamma,
               eta              = tuning_best$eta,
               subsample        = tuning_best$subsample,
               colsample_bytree = tuning_best$colsample_bytree
)
nrounds <- tuning_best$best_nrounds

### define watchlist
watchlist <- list(train=dtrain)

### Fit XGBoost using the best parameters
set.seed(1)
XGBm <- xgb.train(params        = params,
                  data          = dtrain,
                  missing       = NA,
                  nrounds       = nrounds,
                  print_every_n = TRUE,
                  watchlist     = watchlist
)

# predict test ------------------------------------------------------------

pred <- predict(XGBm, newdata=dtest, reshape=T)

# save the model ----------------------------------------------------------

xgb.save(XGBm,"./project/volume/models/model.model")

# make a submission file --------------------------------------------------

submit <- data.table(id=id.test)
submit <- cbind(submit, pred)
setnames(submit, paste0('V',1:num.class), topics)

fwrite(submit, "./project/volume/data/processed/submit.csv")
