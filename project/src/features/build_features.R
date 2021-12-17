# library -----------------------------------------------------------------

library(data.table)
library(httr)
library(Rtsne)
library(ggplot2)
library(gridExtra)

# load train and test -----------------------------------------------------

train <- fread('./project/volume/data/raw/train.csv')
test <- fread('./project/volume/data/raw/test.csv')

# reformat topic ----------------------------------------------------------

topics <- names(train[,-c('id','text')])
train$topic <- 'NA'
for(i in 1:length(topics)) train[get(topics[i])==1]$topic <- topics[i]
train[,topics] <- NULL

# combine train and test as master ----------------------------------------

train$train <- 1
test$train <- 0
master <- rbind(train, test, fill=T)

# embedding ---------------------------------------------------------------

train_emb <- fread('./project/volume/data/raw/train_emb.csv')
test_emb <- fread('./project/volume/data/raw/test_emb.csv')
emb_dt <- rbind(train_emb, test_emb)

# dimension reduction by t-SNE --------------------------------------------

tsne <- Rtsne(emb_dt, check_duplicates=F, verbose=T, dims=3)

### visualize
tsne_dt <- data.table(tsne$Y)
tsne_dt$topic <- master$topic

### plot when dim=3
grid.arrange(ggplot(tsne_dt[master$train==1],aes(x=V1,y=V2,col=topic))+geom_point(),
             ggplot(tsne_dt[master$train==1],aes(x=V1,y=V3,col=topic))+geom_point(),
             ggplot(tsne_dt[master$train==1],aes(x=V2,y=V3,col=topic))+geom_point(),
             nrow=1)

# save train and test -----------------------------------------------------

tsne_dt$id <- master$id
train <- tsne_dt[master$train==1, .(id,V1,V2,V3,topic)]
test <- tsne_dt[master$train==0, .(id,V1,V2,V3,topic)]

fwrite(train, './project/volume/data/interim/train.csv')
fwrite(test, './project/volume/data/interim/test.csv')