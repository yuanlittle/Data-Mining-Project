#library
library('gbm')
library('randomForest')
library('neuralnet')
library('e1071')
library('stats')

#Read original data
#hypothyroid data preprocessing
hypothyroid.train<-data.frame(read.csv("hypothyroid.train",header = FALSE,stringsAsFactors=FALSE))
hypothyroid.test<-data.frame(read.csv("hypothyroid.test",header = FALSE,stringsAsFactors=FALSE))
#data representing
for (i in 1:nrow(hypothyroid.train)){
  for (j in 1:ncol(hypothyroid.train)){
    if (hypothyroid.train[i,j]=="negative"||hypothyroid.train[i,j]=="f"||hypothyroid.train[i,j]=="n"||hypothyroid.train[i,j]=="F") {
      hypothyroid.train[i,j]<-0
    }
    else if(hypothyroid.train[i,j]=="hypothyroid"||hypothyroid.train[i,j]=="M"||hypothyroid.train[i,j]=="y"||hypothyroid.train[i,j]=="t"){ 
      hypothyroid.train[i,j]<-1
    }else if (hypothyroid.train[i,j]=="?"){
      hypothyroid.train[i,j]<--1
    }
  }
}
# #data cleaning
# hypothyroid.train <- subset(hypothyroid.train,select=c(1:25))
# hypothyroid.test <- subset(hypothyroid.test,select=c(1:25))
# #impute data with medium value
# for (i in 1:ncol(hypothyroid.train)){
#   for (j in 1:nrow(hypothyroid.train)){
#     if (is.na(hypothyroid.train[j,i])){
#       hypothyroid.train[j,i]<- median(as.numeric(hypothyroid.train[,i]),na.rm=T)
#     }
#   }
# }
hypothyroid.train<-data.frame(as.matrix(hypothyroid.train))
hypothyroid.train[,1:26] = sapply((hypothyroid.train[,1:26]),as.character)
hypothyroid.train[,1:26] = sapply((hypothyroid.train[,1:26]),as.numeric)

for (i in 1:nrow(hypothyroid.test)){
  for (j in 1:ncol(hypothyroid.test)){
    if (hypothyroid.test[i,j]=="negative"||hypothyroid.test[i,j]=="f"||hypothyroid.test[i,j]=="n"||hypothyroid.test[i,j]=="F") {
      hypothyroid.test[i,j]<-0
    }
    else if(hypothyroid.test[i,j]=="hypothyroid"||hypothyroid.test[i,j]=="M"||hypothyroid.test[i,j]=="y"||hypothyroid.test[i,j]=="t"){ 
      hypothyroid.test[i,j]<-1
    }else if (hypothyroid.test[i,j]=="?"){
      hypothyroid.test[i,j]<--1
    }
  }
}
# #impute data with medium value
# for (i in 1:ncol(hypothyroid.test)){
#   for (j in 1:nrow(hypothyroid.test)){
#     if (is.na(hypothyroid.test[j,i])){
#       hypothyroid.test[j,i]<- median(as.numeric(hypothyroid.test[,i]),na.rm=T)
#     }
#   }
# }
hypothyroid.test<-data.frame(as.matrix(hypothyroid.test))
hypothyroid.test[,1:26] = sapply((hypothyroid.test[,1:26]),as.character)
hypothyroid.test[,1:26] = sapply((hypothyroid.test[,1:26]),as.numeric)

#ionosphere data preprocessing
ionosphere.train<-data.frame(read.csv("ionosphere.train",header = FALSE,stringsAsFactors=FALSE))
ionosphere.test<-data.frame(read.csv("ionosphere.test",header = FALSE,stringsAsFactors=FALSE))
for (i in 1:length(ionosphere.train$V35)){
  if (ionosphere.train$V35[i]=="b"){
    ionosphere.train$V35[i]<-0
  }else{
    ionosphere.train$V35[i]<-1
  }
}
ionosphere.train<-data.frame(as.matrix(ionosphere.train))
ionosphere.train[,1:35] = sapply((ionosphere.train[,1:35]),as.character)
ionosphere.train[,1:35] = sapply((ionosphere.train[,1:35]),as.numeric)
for (i in 1:length(ionosphere.test$V35)){
  if (ionosphere.test$V35[i]=="b"){
    ionosphere.test$V35[i]<-0
  }else{
    ionosphere.test$V35[i]<-1
  }
}
ionosphere.test<-data.frame(as.matrix(ionosphere.test))
ionosphere.test[,1:35] = sapply((ionosphere.test[,1:35]),as.character)
ionosphere.test[,1:35] = sapply((ionosphere.test[,1:35]),as.numeric)

#wdbc data preprocessing
wdbc.train<-data.frame(read.csv("wdbc.train",header = FALSE,stringsAsFactors=FALSE))
wdbc.test<-data.frame(read.csv("wdbc.test",header = FALSE,stringsAsFactors=FALSE))
for (i in 1:length(wdbc.train$V2)){
  if (wdbc.train$V2[i]=="B"){
    wdbc.train$V2[i]<-0
  }else{
    wdbc.train$V2[i]<-1
  }
}
wdbc.train<-data.frame(as.matrix(wdbc.train))
wdbc.train[,1:32] = sapply((wdbc.train[,1:32]),as.character)
wdbc.train[,1:32] = sapply((wdbc.train[,1:32]),as.numeric)
for (i in 1:length(wdbc.test$V2)){
  if (wdbc.test$V2[i]=="B"){
    wdbc.test$V2[i]<-0
  }else{
    wdbc.test$V2[i]<-1
  }
}
wdbc.test<-data.frame(as.matrix(wdbc.test))
wdbc.test[,1:32] = sapply((wdbc.test[,1:32]),as.character)
wdbc.test[,1:32] = sapply((wdbc.test[,1:32]),as.numeric)
wdbc.train <- subset(wdbc.train,select=c(2:32))
wdbc.test <- subset(wdbc.test,select=c(2:32))



#loss function
mylossfunc<- function(predicty,y){
  if (predicty==y){
    error<-0
  }
  else{
    error<-1
  }
}

# partition data into training/validation sets
partition.cv <- function(dat, ratio)
{
  sample_size <- floor(ratio * nrow(dat))
  train_index <- sample(seq_len(nrow(dat)), size = sample_size)
  trn=dat[train_index,]
  val=dat[-train_index,]
  cv=list(train=trn,validation=val)
  return(cv)
}


#classification error
classifiError<-function(predy,Y){
  classifi.error<-0
  for (k in 1:length(predy)){
    classifi.error<-classifi.error+mylossfunc(Y[k],predy[k])/length(predy)
  }
  return (classifi.error)
}

#cross validation patition
set.seed(470)
hypothyroidcv=partition.cv(hypothyroid.train,0.7)
ionospherecv=partition.cv(ionosphere.train,0.7)
wdbccv=partition.cv(wdbc.train,0.7)


#=========GBM-validation&test==========
#parameter
shrinkage<-c(0.01)
depth=c(1)
n.minobsinnode=c(10)
#evaluate performance to tune para
GBMperformanceEval<-function(cv,tst,shrinkage,depth,dataType){
  trn=cv$train
  vali=cv$validation
  if (dataType=="hypothyroid"){
    print(paste("********GBM-hypothyroid*********"))
    tstx=tst[,-1]
    tsty=tst[,1]
    valix=vali[,-1]
    valiy=vali[,1]
    ns=length(shrinkage)
    nd=length(depth)
    for (i in 1:ns) {
      for (j in 1:nd){
        gbm1 <-gbm(trn$V1~.,
                   distribution = 'bernoulli',
                   data=trn, 
                   n.trees = 10000,
                   interaction.depth = depth[j],
                   shrinkage = shrinkage[i],
                   verbose = FALSE,
                   bag.fraction = 0.5,
                   cv.folds = 10/3)
        nTrees<-gbm.perf(gbm1)
        predy<-predict.gbm(gbm1,newdata=valix,type='response',n.trees=nTrees)
        Cerror<-classifiError(round(predy),valiy)
        print(paste("validation.error=",Cerror," shrinkage=",shrinkage[i]," interaction.depth = ",depth[j]," Iteration = ",nTrees))
        
        predy<-predict.gbm(gbm1,newdata=tstx,type='response',n.trees=nTrees)
        Cerror<-classifiError(round(predy),tsty)
        print(paste("test.error=",Cerror," ntrees = ",nTrees))
        
      }
    }
  }else if (dataType=="ionosphere"){
    print(paste("********GBM-ionosphere*********"))
    tstx=tst[,-35]
    tsty=tst[,35]
    valix=vali[,-35]
    valiy=vali[,35]
    ns=length(shrinkage)
    nd=length(depth)
    for (i in 1:ns) {
      for (j in 1:nd){
        gbm1 <- gbm(trn$V35~.,
                     distribution = 'bernoulli',
                     data=trn, 
                     n.trees = 10000,
                     interaction.depth = depth[j],
                     shrinkage = shrinkage[i],
                    verbose = FALSE,
                     bag.fraction = 0.5,
                     cv.folds = 10/3)
        nTrees<-gbm.perf(gbm1)
        predy<-predict.gbm(gbm1,newdata=valix,type='response',n.trees=nTrees)
        Cerror<-classifiError(round(predy),valiy)
        print(paste("validation.error=",Cerror," shrinkage=",shrinkage[i]," interaction.depth = ",depth[j]," Iteration = ",nTrees))
        predy<-predict.gbm(gbm1,newdata=tstx,type='response',n.trees=nTrees)
        Cerror<-classifiError(round(predy),tsty)
        print(paste("test.error=",Cerror," ntrees = ",nTrees))
      }
    }
  }else if (dataType=="wdbc"){
    print(paste("********GBM-wdbc*********"))
    valix=vali[,-1]
    valiy=vali[,1]
    tstx=tst[,-1]
    tsty=tst[,1]
    ns=length(shrinkage)
    nd=length(depth)
    for (i in 1:ns) {
      for (j in 1:nd){
        gbm1 <-gbm(trn$V2~.,
                   distribution = 'bernoulli',
                   data=trn, 
                   n.trees = 10000,
                   interaction.depth = depth[j],
                   shrinkage = shrinkage[i],
                   verbose = FALSE,
                   bag.fraction = 0.5,
                   cv.folds = 10/3)
        nTrees<-gbm.perf(gbm1)
        predy<-predict.gbm(gbm1,newdata=valix,type='response',n.trees=nTrees)
        Cerror<-classifiError(round(predy),valiy)
        print(paste("validation.error=",Cerror," shrinkage=",shrinkage[i]," interaction.depth = ",depth[j]," Iteration = ",nTrees))
        predy<-predict.gbm(gbm1,newdata=tstx,type='response',n.trees=nTrees)
        Cerror<-classifiError(round(predy),tsty)
        print(paste("test.error=",Cerror," ntrees = ",nTrees))
      }
    }
  }
}

#test the GBM performance in validation data
GBMperformanceEval(hypothyroidcv,hypothyroid.test,shrinkage,depth,"hypothyroid")
GBMperformanceEval(ionospherecv,ionosphere.test,shrinkage,depth,"ionosphere")
GBMperformanceEval(wdbccv,wdbc.test,shrinkage,depth,"wdbc")

#============Random Forest-validation==============
ntree=c(10000)
mtry=c()
#test the Random Forest performance in validation data
RFperformanceEval<-function(cv,tst,ntree,dataType){
  trn=cv$train
  vali=cv$validation
  if (dataType=="hypothyroid"){
    print(paste("********RF-hypothyroid*********"))
    valix=vali[,-1]
    valiy=vali[,1]
    tstx=tst[,-1]
    tsty=tst[,1]
    ns=length(ntree)
    for (i in 1:ns) {
        RF<-randomForest(as.factor(trn$V1)~.,data=trn,importance=TRUE,proximity=TRUE,ntree=ntree[i],mtry=2)
        predy<-predict(RF,newdata=valix,type='response')
        Cerror<-classifiError(predy,valiy)
        print(paste("validation.error=",Cerror," Ntree = ",ntree[i]))
        predy<-predict(RF,newdata=tstx,type='response')
        Cerror<-classifiError(predy,tsty)
        print(paste("test.error=",Cerror," Ntree = ",ntree[i]))
      
    }
  }else if (dataType=="ionosphere"){
    print(paste("********RF-ionosphere*********"))
    valix=vali[,-35]
    valiy=vali[,35]
    tstx=tst[,-35]
    tsty=tst[,35]
    ns=length(ntree)
    for (i in 1:ns) {
      RF<-randomForest(as.factor(trn$V35)~.,data=trn,importance=TRUE,proximity=TRUE,ntree=ntree[i],mtry=25)
      predy<-predict(RF,newdata=valix,type='response')
      Cerror<-classifiError(predy,valiy)
      print(paste("validation.error=",Cerror," Ntree = ",ntree[i]))
      predy<-predict(RF,newdata=tstx,type='response')
      Cerror<-classifiError(predy,tsty)
      print(paste("test.error=",Cerror," Ntree = ",ntree[i]))
      
    }
  }else if (dataType=="WDBC"){
    print(paste("********RF-WDBC*********"))
    valix=vali[,-1]
    valiy=vali[,1]
    tstx=tst[,-1]
    tsty=tst[,1]
    ns=length(ntree)
    for (i in 1:ns) {
      RF<-randomForest(as.factor(trn$V2)~.,data=trn,importance=TRUE,proximity=TRUE,ntree=ntree[i],mtry=25)
      predy<-predict(RF,newdata=valix,type='response')
      Cerror<-classifiError(predy,valiy)
      print(paste("validation.error=",Cerror," Ntree = ",ntree[i]))
      predy<-predict(RF,newdata=tstx,type='response')
      Cerror<-classifiError(predy,tsty)
      print(paste("test.error=",Cerror," Ntree = ",ntree[i]))
    }
  }
}
RFperformanceEval(hypothyroidcv,hypothyroid.test,ntree,"hypothyroid")
RFperformanceEval(ionospherecv,ionosphere.test,ntree,"ionosphere")
RFperformanceEval(wdbccv,wdbc.test,ntree,"WDBC")

#=============Neral Network-validation================
threshold=c(0.01)
hidden=c(1)
NNperformanceEval<-function(cv,tst,threshold,hidden,dataType){
  trn=cv$train
  vali=cv$validation
  if (dataType=="hypothyroid"){
    print(paste("********NN-hypothyroid*********"))
    n <- names(trn)
    f <- as.formula(paste("V1 ~", paste(n[!n %in% "V1"], collapse = " + ")))
    valix=vali[,-1]
    valiy=vali[,1]
    tstx=tst[,-1]
    tsty=tst[,1]
    ns=length(threshold)
    nh=length(hidden)
    for (i in 1:ns) {
      for(j in 1:nh){
      nn<-neuralnet(formula = f,data = trn, hidden=hidden,threshold=threshold[i],linear.output = FALSE,stepmax = 1000000)
      predy<-compute(nn,valix)$net.result
      Cerror<-classifiError(round(predy),valiy)
      print(paste("validation.error=",Cerror," threshold = ",threshold[i]," hidden=",hidden[j]))
#       predy<-compute(nn,tstx)$net.result
#       Cerror<-classifiError(round(predy),tsty)
#       print(paste("test.error=",Cerror," threshold = ",threshold[i]))
      }
    }
  }else if (dataType=="ionosphere"){
    print(paste("********NN-ionosphere*********"))
    n <- names(trn)
    f <- as.formula(paste("V35 ~", paste(n[!n %in% "V35"], collapse = " + ")))
    tstx=tst[,-35]
    tsty=tst[,35]
    valix=vali[,-35]
    valiy=vali[,35]
    ns=length(threshold)
    nh=length(hidden)
    for (i in 1:ns) {
      for(j in 1:nh){
      nn<-neuralnet(formula = f,data = trn, hidden=hidden,threshold=threshold[i],linear.output = FALSE,stepmax = 1000000)
      predy<-compute(x=nn,covariate =valix)$net.result
      Cerror<-classifiError(round(predy),valiy)
      print(paste("validation.error=",Cerror," threshold = ",threshold[i]," hidden=",hidden[j]))
#       predy<-compute(nn,tstx)$net.result
#       Cerror<-classifiError(round(predy),tsty)
#       print(paste("test.error=",Cerror," threshold = ",threshold[i]))
      }
    }
  }else if (dataType=="WDBC"){
    print(paste("********NN-WDBC*********"))
    n <- names(trn)
    f <- as.formula(paste("V2 ~", paste(n[!n %in% "V2"], collapse = " + ")))
    tstx=tst[,-1]
    tsty=tst[,1]
    valix=vali[,-1]
    valiy=vali[,1]
    ns=length(threshold)
    nh=length(hidden)
    for (i in 1:ns) {
      for(j in 1:nh){
      nh=length(hidden)
      nn<-neuralnet(formula = f,data = trn, hidden=hidden,threshold=threshold[i],linear.output = FALSE,stepmax = 1000000)
      predy<-compute(x=nn,covariate =valix)$net.result
      Cerror<-classifiError(round(predy),valiy)
      print(paste("validation.error=",Cerror," threshold = ",threshold[i]," hidden=",hidden[j]))
#       predy<-compute(nn,tstx)$net.result
#       Cerror<-classifiError(round(predy),tsty)
#       print(paste("test.error=",Cerror," threshold = ",threshold[i]))
    }
    }
  }
}
NNperformanceEval(hypothyroidcv,hypothyroid.test,threshold,hidden,"hypothyroid")
NNperformanceEval(ionospherecv,ionosphere.test,threshold,hidden,"ionosphere")
NNperformanceEval(wdbccv,wdbc.test,threshold,hidden,"WDBC")

#========================SVM-validation==================================
lambda=c(100,32,10,3.2,1,0.32,0.1,0.032,0.01,0.0032,0.001,0.00032,0.0001)
SVMperformanceEval<-function(cv,lambda,dataType,type){
  trn=cv$train
  vali=cv$validation
  if (dataType=="hypothyroid"){
    print(paste("********SVM-hypothyroid*********"))
    valix=vali[,-1]
    valiy=vali[,1]
    ls=length(lambda)
    for (i in 1:ls) {
      if (type=="linear"){
        mysvm <- svm(x=trn[,-1],y=trn[,1],kernel="linear",type="C-classification",cost=lambda[i])
        predy <- predict(mysvm,valix)
        Cerror<-classifiError(predy,valiy)
        print(paste("validation.error=",Cerror," lambda = ",lambda[i]))
      }
      else if (type == 'RBF') {
        # RBF kernel
        params<-c(0.05,0.1,0.2,0.3,0.4,0.5,0.7,1,2,5)
        for (sigma in params) {
          mysvm<-svm(x=trn[,-1],y=trn[,1],kernel="radial",gamma=1/(2*sigma^2),type="C-classification",cost=lambda[i])
          predy <- as.numeric(as.character(predict(mysvm,valix)))
          Cerror<-classifiError(round(predy),valiy)
          print(paste("validation.error=",Cerror," lambda = ",lambda[i],' sigma=',sigma))
        }

      }
      else {
        # polynomial kernel
        params<-c(1,2,3,4,5,7,10)
        for (sigma in params) {
          mysvm<-svm(x=trn[,-1],y=trn[,1],kernel="polynomial",gamma=1,coef0=1,degree=sigma,type="C-classification",cost=lambda[i])
          predy <- as.numeric(as.character(predict(mysvm,valix)))
          Cerror<-classifiError(round(predy),valiy)
          print(paste("validation.error=",Cerror," lambda = ",lambda[i],' sigma=',sigma))
        }

      }

      
    }
  }else if (dataType=="ionosphere"){
    print(paste("********SVM-ionosphere*********"))
    valix=vali[,-35]
    valiy=vali[,35]
    ls=length(lambda)
    for (i in 1:ls) {
      if (type=="linear"){
        mysvm <- svm(x=trn[,-35],y=trn[,35],kernel="linear",type="C-classification",cost=lambda[i])
        predy <- as.numeric(as.character(predict(mysvm,valix)))
        Cerror<-classifiError(round(predy),valiy)
        print(paste("validation.error=",Cerror," lambda = ",lambda[i]))
      }
      else if (type == 'RBF') {
        # RBF kernel
        params<-c(0.05,0.1,0.2,0.3,0.4,0.5,0.7,1,2,5)
        for (sigma in params) {
          mysvm<-svm(x=trn[,-35],y=trn[,35],kernel="radial",gamma=1/(2*sigma^2),type="C-classification",cost=lambda[i])
          predy <- predict(mysvm,valix)
          Cerror<-classifiError(predy,valiy)
          print(paste("validation.error=",Cerror," lambda = ",lambda[i],' sigma=',sigma))
        }
      }
      else {
        # polynomial kernel
        params<-c(1,2,3,4,5,7,10)
        for (sigma in params) {
          mysvm<-svm(x=trn[,-35],y=trn[,35],kernel="polynomial",gamma=1,coef0=1,degree=sigma,type="C-classification",cost=lambda[i])
          predy <- predict(mysvm,valix)
          Cerror<-classifiError(predy,valiy)
          print(paste("validation.error=",Cerror," lambda = ",lambda[i],' sigma=',sigma))
        }
 
      }
      
    }
  }else if (dataType=="WDBC"){
    print(paste("********SVM-WDBC*********"))
    valix=vali[,-1]
    valiy=vali[,1]
    ls=length(lambda)
    for (i in 1:ls) {
      if (type=="linear"){
        mysvm <- svm(x=trn[,-1],y=trn[,1],kernel="linear",type="C-classification",cost=lambda[i])
        predy <- predict(mysvm,valix)
        Cerror<-classifiError(predy,valiy)
        print(paste("validation.error=",Cerror," lambda = ",lambda[i]))
      }
      else if (type == 'RBF') {
        # RBF kernel
        params<-c(0.05,0.1,0.2,0.3,0.4,0.5,0.7,1,2,5)
        for (sigma in params) {
          mysvm<-svm(x=trn[,-1],y=trn[,1],kernel="radial",gamma=1/(2*sigma^2),type="C-classification",cost=lambda[i])
          predy <- predict(mysvm,valix)
          Cerror<-classifiError(predy,valiy)
          print(paste("validation.error=",Cerror," lambda = ",lambda[i],' sigma=',sigma))
        }
      }
      else {
        # polynomial kernel
        params<-c(1,2,3,4,5,7,10)
        for (sigma in params) {
          mysvm<-svm(x=trn[,-1],y=trn[,1],kernel="polynomial",gamma=1,coef0=1,degree=sigma,type="C-classification",cost=lambda[i])
          predy <- predict(mysvm,valix)
          Cerror<-classifiError(predy,valiy)
          print(paste("validation.error=",Cerror," lambda = ",lambda[i],' sigma=',sigma))

      }
      }
    }
  }
}
print(paste("********SVM-linear*********"))
SVMperformanceEval(hypothyroidcv,lambda,"hypothyroid","linear")
SVMperformanceEval(ionospherecv,lambda,"ionosphere","linear")
SVMperformanceEval(wdbccv,lambda,"WDBC","linear")

print(paste("********SVM-RBF*********"))
SVMperformanceEval(hypothyroidcv,lambda,"hypothyroid","RBF")
SVMperformanceEval(ionospherecv,lambda,"ionosphere","RBF")
SVMperformanceEval(wdbccv,lambda,"WDBC","RBF")

print(paste("********SVM-polynomial*********"))
SVMperformanceEval(hypothyroidcv,lambda,"hypothyroid","polynomial")
SVMperformanceEval(ionospherecv,lambda,"ionosphere","polynomial")
SVMperformanceEval(wdbccv,lambda,"WDBC","polynomial")


#=================logistic regression-validation=================
LRperformanceEval<-function(trn,tst,dataType){

  if (dataType=="hypothyroid"){
    print(paste("********LR-hypothyroid*********"))
    trnx=trn[,-1]
    trny=trn[,1]
    tstx=tst[,-1]
    tsty=tst[,1]
      LR <- glm(trn$V1~.,data=trn,family = "binomial")
      predy <- predict(LR,newdata=trnx,type="response")
      Cerror<-classifiError(round(predy),trny)
      print(paste("train.error=",Cerror))
      predy <- predict(LR,newdata=tstx,type="response")
      Cerror<-classifiError(round(predy),tsty)
      print(paste("test.error=",Cerror))

  }else if (dataType=="ionosphere"){
    print(paste("********LR-ionosphere*********"))
    trnx=trn[,-35]
    trny=trn[,35]
    tstx=tst[,-35]
    tsty=tst[,35]
    LR <- glm(trn$V35~.,data=trn,family = "binomial")
    predy <- predict(LR,newdata=trnx,type="response")
    Cerror<-classifiError(round(predy),trny)
      print(paste("train.error=",Cerror))
      predy <- predict(LR,newdata=tstx,type="response")
      Cerror<-classifiError(round(predy),tsty)
      print(paste("test.error=",Cerror))

  }else if (dataType=="WDBC"){
    print(paste("********LR-WDBC*********"))
    trnx=trn[,-1]
    trny=trn[,1]
    tstx=tst[,-1]
    tsty=tst[,1]
    LR <- glm(trn$V2~.,data=trn,family = "binomial")
    predy <- predict(LR,newdata=trnx,type="response")
    Cerror<-classifiError(round(predy),trny)
    print(paste("train.error=",Cerror))
    predy <- predict(LR,newdata=tstx,type="response")
    Cerror<-classifiError(round(predy),tsty)
    print(paste("test.error=",Cerror))
  }
}
LRperformanceEval(hypothyroid.train,hypothyroid.test,"hypothyroid")
LRperformanceEval(ionosphere.train,ionosphere.test,"ionosphere")
LRperformanceEval(wdbc.train,wdbc.test,"WDBC")


#======================test performance-SVM======================
testSVM<-function(trn,tst,datatype,kernel){
  if (datatype=="hypothyroid"){
    tsty=tst[,1]
    tstx=tst[,-1]
    tsty=tst[,1]
    if (kernel=="linear"){
      mysvm <- svm(x=trn[,-1],y=as.numeric(as.character(trn[,1])),kernel="linear",type="C-classification",cost=1)
      predy <- predict(mysvm,tstx)
      Cerror<-classifiError(predy,tsty)
      print(paste("test.error=",Cerror," lambda = ",1))
    }else if (kernel=="polynomial"){
      mysvm<-svm(x=trn[,-1],y=as.numeric(as.character(trn[,1])),kernel="polynomial",gamma=1,coef0=1,degree=4,type="C-classification",3.2)
      predy <- predict(mysvm,tstx)
      Cerror<-classifiError(predy,tsty)
      print(paste("test.error=",Cerror," lambda = ",3.2,' sigma=',4))
    }else if (kernel=="RBF"){
      mysvm<-svm(x=trn[,-1],y=as.numeric(as.character(trn[,1])),kernel="radial",gamma=1/(2*5^2),type="C-classification",cost=3.2)
      predy <- predict(mysvm,tstx)
      Cerror<-classifiError(predy,tsty)
      print(paste("test.error=",Cerror," lambda = ",3.2,' sigma=',5))
    }
  }else if (datatype=="ionosphere"){
    tsty=tst[,35]
    tstx=tst[,-35]
    tsty=tst[,35]
    if (kernel=="linear"){
      mysvm <- svm(x=trn[,-35],y=as.numeric(as.character(trn[,35])),kernel="linear",type="C-classification",cost=0.1)
      predy <- predict(mysvm,tstx)
      Cerror<-classifiError(predy,tsty)
      print(paste("test.error=",Cerror," lambda = ",0.1))
    }else if (kernel=="polynomial"){
      mysvm<-svm(x=trn[,-35],y=as.numeric(as.character(trn[,35])),kernel="polynomial",gamma=1,coef0=1,degree=2,type="C-classification",0.01)
      predy <- predict(mysvm,tstx)
      Cerror<-classifiError(predy,tsty)
      print(paste("test.error=",Cerror," lambda = ",0.01,' sigma=',2))
    }else if (kernel=="RBF"){
      mysvm<-svm(x=trn[,-35],y=as.numeric(as.character(trn[,35])),kernel="radial",gamma=1/(2*1^2),type="C-classification",cost=3.2)
      predy <- predict(mysvm,tstx)
      Cerror<-classifiError(predy,tsty)
      print(paste("test.error=",Cerror," lambda = ",3.2,' sigma=',1))
    }
  }else if (datatype=="WDBC"){
    tsty=tst[,1]
    tstx=tst[,-1]
    tsty=tst[,1]
    if (kernel=="linear"){
      mysvm <- svm(x=trn[,-1],y=as.numeric(as.character(trn[,1])),kernel="linear",type="C-classification",cost=0.1)
      predy <- predict(mysvm,tstx)
      Cerror<-classifiError(predy,tsty)
      print(paste("test.error=",Cerror," lambda = ",0.1))
    }else if (kernel=="polynomial"){
      mysvm<-svm(x=trn[,-1],y=as.numeric(as.character(trn[,1])),kernel="polynomial",gamma=1,coef0=1,degree=2,type="C-classification",3.2)
      predy <- predict(mysvm,tstx)
      Cerror<-classifiError(predy,tsty)
      print(paste("test.error=",Cerror," lambda = ",3.2,' sigma=',2))
    }else if (kernel=="RBF"){
      mysvm<-svm(x=trn[,-1],y=as.numeric(as.character(trn[,1])),kernel="radial",gamma=1/(2*5^2),type="C-classification",cost=3.2)
      predy <- predict(mysvm,tstx)
      Cerror<-classifiError(predy,tsty)
      print(paste("test.error=",Cerror," lambda = ",3.2,' sigma=',5))
    }

  }
}

testSVM(hypothyroidcv$train,hypothyroid.test,"hypothyroid","linear")
testSVM(hypothyroidcv$train,hypothyroid.test,"hypothyroid","polynomial")
testSVM(hypothyroidcv$train,hypothyroid.test,"hypothyroid","RBF")

testSVM(ionospherecv$train,ionosphere.test,"ionosphere","linear")
testSVM(ionospherecv$train,ionosphere.test,"ionosphere","polynomial")
testSVM(ionospherecv$train,ionosphere.test,"ionosphere","RBF")

testSVM(wdbccv$train,wdbc.test,"WDBC","linear")
testSVM(wdbccv$train,wdbc.test,"WDBC","polynomial")
testSVM(wdbccv$train,wdbc.test,"WDBC","RBF")


