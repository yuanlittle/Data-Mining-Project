#JY470 12/13/2015
#library
library("lars")
library('neuralnet')
library('gbm')
library('randomForest')

# compute normalization factor
normfact <- function (x) {
  # fill in code
  nfact =sqrt( 1 / mean(x^2))
  return (nfact)
}

# normalize
normalize <- function (x, nf) {
  # fill in code
  for(j in 1:length(nf)){
    x[,j]=x[,j]*nf[j]
  }
  return(x)
}

# centering with respect to mean mu
centering <- function (x, mu) {
  if (is.vector(x)) {
    # fill in code
    x=x-mu
  }
  else {
    for(j in 1:ncol(x)){
      x[,j]=x[,j]-mu[j]
    }
  }
  return(x)
}

# MSE (mean squared error) between predicted-y py and true-y ty
mse <- function(py,ty) {
  return(mean((py-ty)^2))
}

# compute ridge regression weight
myridge.fit <- function(X,y,lambda) {
  # fill in code -- replace the following with ridge regression formula
  w=solve(t(X)%*%X+lambda*diag(ncol(X)))%*%t(X)%*%y
  return(w)
}

# compute predicted Y with coefficients w and data matrix X
predict.linear <- function (w,X) {
  n<-dim(X)[1]
  y<-c(array(0,dim=c(n,1)))
  for (i in 1:n) {
    y[i]=sum(w*X[i,])
  }
  return(y)
}

# compute mean squared error for linear weight w on data (X,y)
mse.linear <- function(w,X,y) {
  py=predict.linear(w,X)
  return (mse(py,y))
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

#read data
header<-c("CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV")
housing<-data.matrix(read.table("housing.data",col.names=header))
trn<-housing[1:253,]
tst<-housing[254:506,]
trnx<-housing[1:253,1:13]
trny<-housing[1:253,14]
tstx<-housing[254:506,1:13]
tsty<-housing[254:506,14]

#normalization
nfact=apply(trnx,2,normfact)
trnx=normalize(trnx,nfact)
tstx=normalize(tstx,nfact)

#centering
mux=apply(trnx,2,mean)
muy=mean(trny)
trnx=centering(trnx,mux)
trny=centering(trny,muy)
muxtst=apply(tstx,2,mean)
muytst=mean(tsty)
tstx=centering(tstx,muxtst)
tsty=centering(tsty,muytst)

#cross validation partition
set.seed(233)
cv=partition.cv(trn,0.7)
cvtrn=cv$train
cvvali=cv$validation

#ridge regression
ridgeEval <- function(trn,vali,lambda,tst,best) {
  tstx=tst[,1:13]
  tsty=tst[,14]
  trnx=trn[,1:13]
  trny=trn[,14]
  valix=vali[,1:13]
  valiy=vali[,14]
  valierr=lambda
  print(paste("====================ridge regression===================="))
  # loop through regualrization parameters
  for (i in 1:length(lambda)) {
    # compute coefficient using ridge regression
    ww<-myridge.fit(trnx,trny,lambda[i])
    # compute  error
    valierr[i]<-mse.linear(ww,valix,valiy)
    print(paste("validation.error=",valierr[i]," lambda=",lambda[i]))
  }
  ww<-myridge.fit(tstx,tsty,best)
  # compute  error
  err<-mse.linear(ww,valix,valiy)
  print(paste("test.error=",err," lambda=",best))
  
}
#regularization parameter lambda
lambda=c(1e-4,1e-3,1e-2,0.1,1,10,1e2, 1e3,1e4,1e5,1e6,1e7,1e8)
#ridge regression
ridgeEval(cvtrn,cvvali,lambda,tst,0.1)


#neural network regression
threshold=c(0.01,0.05,0.1)
hidden=c(1,2,3,4)
NNperformanceEval<-function(trn,vali,threshold,hidden,tst,bestT,bestH){
  valix=vali[,1:13]
  valiy=vali[,14]
  tstx=tst[,1:13]
  tsty=tst[,14]
  n <- names(data.frame(trn))
  f <- as.formula(paste("MEDV ~", paste(n[!n %in% "MEDV"], collapse = " + ")))
    print(paste("********Neural Networks-regression*********"))
    ns=length(threshold)
    hs=length(hidden)
    for (i in 1:ns) {
      for (j in 1:hs){
        nn<-neuralnet(formula = f,data = trn,hidden=hidden[j],threshold=threshold[i])
        predy<-compute(nn,valix)$net.result
        Cerror<-mse(predy,valiy)
        print(paste("validation.error=",Cerror," threshold=",threshold[i]," hidden=",hidden[j]))
      
      }
    nn<-neuralnet(formula = f,data = tst,hidden=bestH,threshold=bestT)
    predy<-compute(nn,tstx)$net.result
    Cerror<-mse(predy,tsty)
    print(paste("test.error=",Cerror," threshold=",bestT," hidden=",bestH))
    }
}
NNperformanceEval(cvtrn,cvvali,threshold,hidden,tst,0.01,2)


#GBM
#parameter
shrinkage<-c(0.01)
depth=c(1)
n.minobsinnode=c(20,10)
GBMperformanceEval<-function(trn,vali,shrinkage,depth,tst,bestS,BestD){
  valix=vali[,1:13]
  valiy=vali[,14]
  tstx=tst[,1:13]
  tsty=tst[,14]
    print(paste("********GBM-regression*********"))
    n <- names(data.frame(trn))
    f <- as.formula(paste("MEDV ~", paste(n[!n %in% "MEDV"], collapse = " + ")))
    ns=length(shrinkage)
    nd=length(depth)
    for (i in 1:ns) {
      for (j in 1:nd){
        gbm1 <-gbm(formula = f,
                   distribution = 'gaussian',
                   data=data.frame(trn), 
                   n.trees = 10000,
                   interaction.depth = depth[j],
                   shrinkage = shrinkage[i],
                   bag.fraction = 0.5,
                   cv.folds = 10/3)
        nTrees<-gbm.perf(gbm1)
        predy<-predict.gbm(gbm1,newdata=data.frame(valix),type='response',n.trees=nTrees)
        Cerror<-mse((predy),valiy)
        print(paste("validation.error=",Cerror," shrinkage=",shrinkage[i]," interaction.depth = ",depth[j]," Iteration = ",nTrees))
      }
    }
    gbm1 <-gbm(formula = f,
               distribution = 'gaussian',
               data=data.frame(tst), 
               n.trees = 10000,
               interaction.depth = BestD,
               shrinkage = bestS,
               bag.fraction = 0.5,
               cv.folds = 10/3)
    nTrees<-gbm.perf(gbm1)
    predy<-predict.gbm(gbm1,newdata=data.frame(tstx),type='response',n.trees=nTrees)
    Cerror<-mse((predy),tsty)
    print(paste("test.error=",Cerror," shrinkage=",bestS," interaction.depth = ",BestD," Iteration = ",nTrees))
}
GBMperformanceEval(cvtrn,cvvali,shrinkage,depth,tst,0.01,1)


#Random Forest
ntree=c(1000,2000,3000,5000,6000)
mtry=c()
RFperformanceEval<-function(trn,vali,ntree,tst,bestN){
  valix=vali[,1:13]
  valiy=vali[,14]
    print(paste("********RF-Regression*********"))
    n <- names(data.frame(trn))
    f <- as.formula(paste("MEDV ~", paste(n[!n %in% "MEDV"], collapse = " + ")))
    ns=length(ntree)
    for (i in 1:ns) {
      RF<-randomForest(formula = f,data=trn,importance=TRUE,proximity=TRUE,ntree=ntree[i])
      predy<-predict(RF,newdata=valix,type='response')
      Cerror<-mse(predy,valiy)
      print(paste("validation.error=",Cerror," Ntree = ",ntree[i]))
      
    }
    RF<-randomForest(formula = f,data=tst,importance=TRUE,proximity=TRUE,ntree=bestN)
    predy<-predict(RF,newdata=tstx,type='response')
    Cerror<-mse(predy,tsty)
    print(paste("test.error=",Cerror," Ntree = ",bestN))
}
RFperformanceEval(cvtrn,cvvali,ntree,tst,1000)


