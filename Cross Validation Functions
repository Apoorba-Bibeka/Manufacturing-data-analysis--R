#**********************************************************************************
#***************************Cross Validation Functions for LDA,QDA and KNN*********
#**********************************************************************************

#Fuction for K fold cross validating LDA


cross_val_lda<-function(xvars,yvar,dat,K=5){
  library(MASS)
  #Get the samples from data set
  #For example:
  #s[1,],s[2,]....... are test data set
  #-s[1],-s[2]....... are training data set
  
  #Define the size of the k_th fold. Round down so that this function works for 
  #diff data set
  chunk_k<-floor(nrow(dat)/K)
  set.seed(1)
  s<-array(data=0,c(K,chunk_k))
  nrow_dat<-seq(1,nrow(dat))
  for(i in 1:K){
    s[i,]<-sample(nrow_dat,size=chunk_k,replace=FALSE)
    nrow_dat<-setdiff(nrow_dat,s)
  }
  
  #Use function arguments to create input for lda. 
  my.formula <- paste(yvar, '~', paste( xvars, collapse=' + ' ) )
  my.formula <- as.formula(my.formula)
  
  #Get the cross validation error 
  lda_mis<-vector(length=K)
  for(i in 1:K){
    fit<-lda( my.formula, data=dat[-s[i,],])
    lda_pr<-predict(fit,dat[s[i,],])
    t=table(lda_pr$class,dat[s[i,],yvar])
    lda_mis[i]<-(t[2]+t[3])/sum(t)
  }
  cv.error<-mean(lda_mis)
  cv.error
}



#Fuction for K fold cross validating QDA


cross_val_qda<-function(xvars,yvar,dat,K=5){
  library(MASS)
  #Get the samples from data set
  #For example:
  #s[1,],s[2,]....... are test data set
  #-s[1],-s[2]....... are training data set
  
  #Define the size of the k_th fold. Round down so that this function works for 
  #diff data set
  chunk_k<-floor(nrow(dat)/K)
  set.seed(1)
  s<-array(data=0,c(K,chunk_k))
  nrow_dat<-seq(1,nrow(dat))
  for(i in 1:K){
    s[i,]<-sample(nrow_dat,size=chunk_k,replace=FALSE)
    nrow_dat<-setdiff(nrow_dat,s)
  }
  
  #Use function arguments to create input for lda. 
  my.formula <- paste(yvar, '~', paste( xvars, collapse=' + ' ) )
  my.formula <- as.formula(my.formula)
  
  #Get the cross validation error 
  qda_mis<-vector(length=K)
  for(i in 1:K){
    fit<-qda( my.formula, data=dat[-s[i,],])
    qda_pr<-predict(fit,dat[s[i,],])
    t=table(qda_pr$class,dat[s[i,],yvar])
    qda_mis[i]<-(t[2]+t[3])/sum(t)
  }
  cv.error<-mean(qda_mis)
  cv.error
}


#Fuction for K fold cross validating KNN

cross_val_knn<-function(xvars,yvar,dat,K=5,kn){

  library(class)
  #Get the samples from data set
  #For example:
  #s[1,],s[2,]....... are test data set
  #-s[1],-s[2]....... are training data set
  
  #Define the size of the k_th fold. Round down so that this function works for 
  #diff data set
  chunk_k<-floor(nrow(dat)/K)
  set.seed(1)
  s<-array(data=0,c(K,chunk_k))
  nrow_dat<-seq(1,nrow(dat))
  for(i in 1:K){
    s[i,]<-sample(nrow_dat,size=chunk_k,replace=FALSE)
    nrow_dat<-setdiff(nrow_dat,s)
  }

  #Get the cross validation error 
  knn_mis<-vector(length=K)
  for(i in 1:K){
    set.seed(1)
    set.seed(1)
    knn.pred=knn(data.frame(dat[-s[i,],xvars]),data.frame(dat[s[i,],xvars]),dat[-s[i,],yvar],k=kn)
    t=table(knn.pred,dat[s[i,],yvar])
    knn_mis[i]<-(t[2]+t[3])/sum(t)
  }
  cv.error<-mean(knn_mis)
  cv.error
}



#Fuction for K fold cross validating SVM linear
library(e1071)

cross_val_svm_l<-function(xvars,yvar,dat,K=5,ker,c){
  #Get the samples from data set
  #For example:
  #s[1,],s[2,]....... are test data set
  #-s[1],-s[2]....... are training data set
  
  #Define the size of the k_th fold. Round down so that this function works for 
  #diff data set
  chunk_k<-floor(nrow(dat)/K)
  set.seed(1)
  s<-array(data=0,c(K,chunk_k))
  nrow_dat<-seq(1,nrow(dat))
  for(i in 1:K){
    s[i,]<-sample(nrow_dat,size=chunk_k,replace=FALSE)
    nrow_dat<-setdiff(nrow_dat,s)
  }
  
  #Use function arguments to create input for lda. 
  my.formula <- paste(yvar, '~', paste( xvars, collapse=' + ' ) )
  my.formula <- as.formula(my.formula)
  
  #Get the cross validation error 
  svm_mis<-vector(length=K)
  for(i in 1:K){
    fit<-svm( my.formula, data=dat[-s[i,],],kernel=ker,cost=c)
    svm_pr<-predict(fit,dat[s[i,],])
    t=table(svm_pr,dat[s[i,],yvar])
    svm_mis[i]<-(t[2]+t[3])/sum(t)
  }
  cv.error<-mean(svm_mis)
  cv.error
}



#Fuction for K fold cross validating SVM linear
library(e1071)

cross_val_svm_l<-function(xvars,yvar,dat,K=5,c){
  #Get the samples from data set
  #For example:
  #s[1,],s[2,]....... are test data set
  #-s[1],-s[2]....... are training data set
  
  #Define the size of the k_th fold. Round down so that this function works for 
  #diff data set
  chunk_k<-floor(nrow(dat)/K)
  set.seed(1)
  s<-array(data=0,c(K,chunk_k))
  nrow_dat<-seq(1,nrow(dat))
  for(i in 1:K){
    s[i,]<-sample(nrow_dat,size=chunk_k,replace=FALSE)
    nrow_dat<-setdiff(nrow_dat,s)
  }
  
  #Use function arguments to create input for lda. 
  my.formula <- paste(yvar, '~', paste( xvars, collapse=' + ' ) )
  my.formula <- as.formula(my.formula)
  
  #Get the cross validation error 
  svm_mis<-vector(length=K)
  for(i in 1:K){
    fit<-svm( my.formula, data=dat[-s[i,],],kernel="linear",probability=TRUE,cost=c)
    svm_pr<-predict(fit,dat[s[i,],])
    t=table(svm_pr,dat[s[i,],yvar])
    svm_mis[i]<-(t[2]+t[3])/sum(t)
  }
  cv.error<-mean(svm_mis)
  cv.error
}



#Fuction for K fold cross validating SVM Radial

cross_val_svm_r<-function(xvars,yvar,dat,K=5,gam,c){
  #Get the samples from data set
  #For example:
  #s[1,],s[2,]....... are test data set
  #-s[1],-s[2]....... are training data set
  
  #Define the size of the k_th fold. Round down so that this function works for 
  #diff data set
  chunk_k<-floor(nrow(dat)/K)
  set.seed(1)
  s<-array(data=0,c(K,chunk_k))
  nrow_dat<-seq(1,nrow(dat))
  for(i in 1:K){
    s[i,]<-sample(nrow_dat,size=chunk_k,replace=FALSE)
    nrow_dat<-setdiff(nrow_dat,s)
  }
  
  #Use function arguments to create input for lda. 
  my.formula <- paste(yvar, '~', paste( xvars, collapse=' + ' ) )
  my.formula <- as.formula(my.formula)
  
  #Get the cross validation error 
  svm_mis<-vector(length=K)
  for(i in 1:K){
    fit<-svm( my.formula, data=dat[-s[i,],],kernel="radial",probability=TRUE,cost=c,gamma=gam)
    svm_pr<-predict(fit,dat[s[i,],])
    t=table(svm_pr,dat[s[i,],yvar])
    svm_mis[i]<-(t[2]+t[3])/sum(t)
  }
  cv.error<-mean(svm_mis)
  cv.error
}


#Fuction for K fold cross validating SVM Poly

cross_val_svm_p<-function(xvars,yvar,dat,K=5,deg,c){
  #Get the samples from data set
  #For example:
  #s[1,],s[2,]....... are test data set
  #-s[1],-s[2]....... are training data set
  
  #Define the size of the k_th fold. Round down so that this function works for 
  #diff data set
  chunk_k<-floor(nrow(dat)/K)
  set.seed(1)
  s<-array(data=0,c(K,chunk_k))
  nrow_dat<-seq(1,nrow(dat))
  for(i in 1:K){
    s[i,]<-sample(nrow_dat,size=chunk_k,replace=FALSE)
    nrow_dat<-setdiff(nrow_dat,s)
  }
  
  #Use function arguments to create input for lda. 
  my.formula <- paste(yvar, '~', paste( xvars, collapse=' + ' ) )
  my.formula <- as.formula(my.formula)
  
  #Get the cross validation error 
  svm_mis<-vector(length=K)
  for(i in 1:K){
    fit<-svm( my.formula, data=dat[-s[i,],],kernel="polynomial",probability=TRUE,cost=c,degree =deg)
    svm_pr<-predict(fit,dat[s[i,],])
    t=table(svm_pr,dat[s[i,],yvar])
    svm_mis[i]<-(t[2]+t[3])/sum(t)
  }
  cv.error<-mean(svm_mis)
  cv.error
}
