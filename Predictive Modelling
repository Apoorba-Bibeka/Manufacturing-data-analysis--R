#Load the data 
```{r}
#**************************************************************************
#************************* Load the data **********************************
#**************************************************************************
rm(list=ls())
setwd("/Users/Apoorb/Dropbox/PHD_course_work/ISEN-613/Exam/exam_2")
data<-read.csv("data.csv",header=TRUE)
sum(is.na(data))
#The above result show that there are no na's in the data 
str(data)
summary(data)
data$Padwear<-as.factor(data$Padwear)
#Structure and Summary of the data tell us that all the predictors
#and response are numeric, hence there are no missing values. 

#Following are the boxplots for the data
par(mfrow=c(2,2))
for(i in 2:ncol(data)){
  boxplot(data[,i],data[,1],xlab=names(data)[1],ylab=names(data)[i])
}

#Dividing the data into training and test data:
set.seed(1)
train<-sample(nrow(data),292)
data.train<-data[train,]
data.test<-data[-train,]
test_y<-data.test$Padwear

```

# 1 Following the check-list given in your lecture, apply alternative classification methods for pad wear detection (show your work, upload your codes).

## Loading all the library 
```{r, message=FALSE, warning=FALSE}
#**************************************************************************
#************************* Load the libraries *****************************
#**************************************************************************
#cross_val_function.R contains library for LDA, QDA and KNN and functions for cross validation
source("/Users/Apoorb/Dropbox/PHD_course_work/ISEN-613/Exam/exam_2/cross_val_function.R")
#ROC curve lib
require(ROCR)
#Random Forest and bagging 
require(randomForest)
#Support Vector Machine 
require(e1071)
#CV for glm
require(boot)
#Tree
require(tree)
```


## 1.a Logistic Regression

## I tried different logistic models with different combination of predictors. The choice of predictor to be included in the model was based on boxplots, pruned tree and from the importance of differnt variable found from random forest. AIC and cross validation error with K=5 were the two critera on which the best model was choosen. 
```{r}
#**************************************************************************
#************************* Logistic Regression ****************************
#**************************************************************************

#Trying different logistic models and checking their cross validation error for K=5
#**********************************************************************************
set.seed(1)
#Model1
logis_m_1<-glm(Padwear~.,data=data.train,family="binomial")  #AIC=347.3
cv.error.1=cv.glm(data.train,logis_m_1,K=5)$delta[1]

#Model2
logis_m_2<-glm(Padwear~pv.x+clearance.y+abs.mean.x,data=data.train,family="binomial") #All the terms became insignificant    #AIC=402.9
cv.error.2=cv.glm(data.train,logis_m_2,K=5)$delta[1]
#Model3
logis_m_3<-glm(Padwear~std.y+pv.y+impulse.f.x,data=data.train,family="binomial")#AIC =360.7
cv.error.3=cv.glm(data.train,logis_m_3,K=5)$delta[1]
#Model4
logis_m_4<-glm(Padwear~std.y+pv.y+impulse.f.x+pv.x,data=data.train,family="binomial")#AIC =343.4
cv.error.4=cv.glm(data.train,logis_m_4,K=5)$delta[1]
#Model5
logis_m_5<-glm(Padwear~std.y+pv.x+RMS.y,data=data.train,family="binomial")#AIC =340
cv.error.5=cv.glm(data.train,logis_m_5,K=5)$delta[1] #Lowest cv error (0.202)
#Model6
logis_m_6<-glm(Padwear~std.y*pv.x*RMS.y,data=data.train,family="binomial")#AIC =343.3
cv.error.6=cv.glm(data.train,logis_m_6,K=5)$delta[1]  

#Model 7
logis_m_7<-glm(Padwear~std.y+pv.x+RMS.y+kurtosis.y,data=data.train,family="binomial")#AIC =335.92
cv.error.7=cv.glm(data.train,logis_m_7,K=5)$delta[1] 

#Model 8
logis_m_8<-glm(Padwear~std.y+pv.x+RMS.y+kurtosis.y+skewness.y,data=data.train,family="binomial")#AIC =334.36
cv.error.8=cv.glm(data.train,logis_m_8,K=5)$delta[1] 

#Model 9
logis_m_9<-glm(Padwear~std.y+pv.x+RMS.y+skewness.y,data=data.train,family="binomial")#AIC =332.4
cv.error.9=cv.glm(data.train,logis_m_9,K=5)$delta[1] 

#Model 10 
#Stepwise selection - backward
logis_m_10<-step(logis_m_1,trace=0)  #AIC=329.4
cv.error.10=cv.glm(data.train,logis_m_10,K=5)$delta[1] 


#Stepwise selection -forward
noth<-glm(Padwear~1,data = data.train,family = "binomial")
forwards = step(noth,
scope=list(lower=formula(noth),upper=formula(logis_m_10)),trace = 0, direction="forward")
#Same result as backward selection process 

#Model 11
#Remove non significant terms from model 10
logis_m_11<-glm(Padwear~std.y+std.x+crest.f.x+kurtosis.y+impulse.f.y,data=data.train,family = "binomial")  #AIC =329.98 and all the predictors are significant
cv.error.11=cv.glm(data.train,logis_m_11,K=5)$delta[1] 

for(i in 1:11){
  a<-eval(as.symbol(paste("cv.error.",i,sep="")))
  a<-round(a,4)
  b<-paste("Model",i,sep=" ")
  if(i==1) print(c("Model","CV Error"))
  print(c(b,a))
}

logis_m_f<-logis_m_11
cv.err_logis=cv.glm(data.train,logis_m_f,K=5)$delta[1]

#After choosing the best model calculate the misclassification rate, sensitivity and specificity 
#**************************************************************************
logis_pred<-predict(logis_m_f,newdata=data.test,type="response")
logis_class<-rep(0,nrow(data.test))
logis_class<-ifelse(logis_pred<0.5,0,1)
logis_class<-as.factor(logis_class)
conf<-table(logis_class,test_y)
spec_logis<-conf[1]/(conf[1]+conf[2])
sen_logis<-conf[4]/(conf[3]+conf[4])
mis_clas_logis<-mean(test_y!=logis_class)

```

## Logistic Regression : Based on the AIC and cross validation error, Model 9 was found to be the best model in logistic regression. The model is as follows:  
## logit(p(x)=1)= -2.47 - 5.654 std.y + 2.962 std.x - 0.76 crest.f.x + 0.6032 kurtosis.y - 0.881 impulse.f.y 

## Final Logistic Model Summary are as follows:
```{r}
summary(logis_m_f)
#Cross Validation Error
cv.err_logis
#Misclassification error rate
mis_clas_logis
#Sensitivity 
sen_logis
#Specificity
spec_logis
```


## 1.b LDA 
```{r, warning=FALSE}
#**************************************************************************
#************************* LDA ********************************************
#**************************************************************************

#Trying Diff model for LDA 
#**************************************************************************
#Model 1 CV error
cross_val_lda(xvars=names(data)[2:21],yvar="Padwear",dat=data.train,K=5)
#Model 2 CV error
cross_val_lda(xvars=c("std.y","pv.x","RMS.y"),yvar="Padwear",dat=data.train,K=5)
#Model 3 CV
cross_val_lda(xvars=c("std.y","pv.x","RMS.y","skewness.y"),yvar="Padwear",dat=data.train,K=5)
#Model 4 CV 
cross_val_lda(xvars=c("std.y","pv.x","RMS.y","kurtosis.y","skewness.y"),yvar="Padwear",dat=data.train,K=5)


#Final Cross valiation error
cv.err_lda<-cross_val_lda(xvars=c("std.y","pv.x","RMS.y","skewness.y"),yvar="Padwear",dat=data.train,K=5)


lda_f<-lda(Padwear~std.y+pv.x+RMS.y+skewness.y,data=data.train)
lda_res<-predict(lda_f,newdata=data.test)
lda_pred<-lda_res$posterior[,2]

#After choosing the best model calculate the misclassification rate, sensitivity and specificity 
#**************************************************************************
conf<-table(lda_res$class,test_y)
spec_lda<-conf[1]/(conf[1]+conf[2])
sen_lda<-conf[4]/(conf[3]+conf[4])
mis_clas_lda<-mean(test_y!=lda_res$class)
```

## Best LDA model is :
##  D(x)=752.376 std.y + 0.910 pv.x -760.64 RMS.y - 1.76 skewness.y
  
## Cross validation error with K=5 was used for model selection. 
  
## Final Lda Model Summary are as follows:
```{r}
lda_f
#Cross Validation Error 
cv.err_lda
#Misclassification error rate
mis_clas_lda
#Sensitivity
sen_lda
#Specificity
spec_lda
```


## 1.c QDA 
```{r}
#**************************************************************************
#************************* QDA ********************************************
#**************************************************************************
#Model 1 CV error
cross_val_qda(xvars=names(data)[2:21],yvar="Padwear",dat=data.train,K=5)
#Model 2 CV error
cross_val_qda(xvars=c("std.y","pv.x","RMS.y"),yvar="Padwear",dat=data.train,K=5)
#Model 3 CV
cross_val_qda(xvars=c("std.y","pv.x","RMS.y","skewness.y"),yvar="Padwear",dat=data.train,K=5)
#Model 4 CV 
cross_val_qda(xvars=c("std.y","pv.x","RMS.y","kurtosis.y","skewness.y"),yvar="Padwear",dat=data.train,K=5)



#Final Model QDA
cv.err_qda<-cross_val_qda(xvars=c("std.y","pv.x","RMS.y"),yvar="Padwear",dat=data.train,K=5)


qda_f<-qda(Padwear~std.y+pv.x+RMS.y,data=data.train)
qda_res<-predict(qda_f,newdata=data.test)
qda_pred<-qda_res$posterior[,2]
#After choosing the best model calculate the misclassification rate, sensitivity and specificity 
#**************************************************************************
conf<-table(qda_res$class,test_y)
spec_qda<-conf[1]/(conf[1]+conf[2])
sen_qda<-conf[4]/(conf[3]+conf[4])
mis_clas_qda<-mean(test_y!=qda_res$class)
```

## Best QDA model is contains the terms std.y, pv.x and RMS.y.  Cross validation error with K=5 was used for model selection. 
  
## Final Qda Model Summary are as follows:
```{r}
qda_f
#Cross Validation error
cv.err_qda
#Misclassification error rate
mis_clas_qda
#Sensitivity
sen_qda
#Specificity
spec_qda
```

## 1.d KNN 
```{r}
#*************************************************************************
#************************* KNN *******************************************
#*************************************************************************
set.seed(1)
for(i in 3:10){
   #Model 1
  a<-cross_val_knn(xvars=names(data)[2:21],yvar="Padwear",dat=data.train,K=5,kn=i)
  #Model 2
  b<-cross_val_knn(xvars=c("pv.x","RMS.y"),yvar="Padwear",dat=data.train,K=5,kn=i)
  #Model 3
   c<-cross_val_knn(xvars=c("std.y","pv.x","RMS.y"),yvar="Padwear",dat=data.train,K=5,kn=i)
   #Model 4
   d<-cross_val_knn(xvars=c("std.y","pv.y","abs.mean.y","pv.x","RMS.y"),yvar="Padwear",dat=data.train,K=5,kn=i)
   
      #Model 5
   e<-cross_val_knn(xvars=c("std.y","pv.x","kurtosis.y","RMS.y"),yvar="Padwear",dat=data.train,K=5,kn=i)
   
   
      bu1<-c(a,b,c,d,e)
      bu2<-paste("Model",seq(1:length(bu1)))
      if(i==3) print(c("K",bu2))
      print(c(i,bu1))
}

#Best Model with k=6
cv.err_knn<-cross_val_knn(xvars=c("pv.x","RMS.y"),yvar="Padwear",dat=data.train,K=5,kn=6)
knn_mod=knn(data.train[,c("pv.x","RMS.y")],data.test[,c("pv.x","RMS.y")],data.train[,1],k=6,prob=TRUE)
prob<-attr(knn_mod,which="prob")
knn_pred<-ifelse(knn_mod=="1",prob,1.00-prob)

#After choosing the best model calculate the misclassification rate, sensitivity and specificity 
#*************************************************************************
conf<-table(knn_mod,test_y)
spec_knn<-conf[1]/(conf[1]+conf[2])
sen_knn<-conf[4]/(conf[3]+conf[4])
mis_clas_knn<-mean(test_y!=knn_mod)

```
## Best KNN model was found for K=6 and predictors pv.x and RMS.y. Cross validation error with K=5 was used for model selection. 

## Model summary are as follows:
```{r}
#Misclassification error rate
mis_clas_knn
#Specificity
spec_knn
#Sensitivity
sen_knn
```



## 1.e CART with Pruning 
```{r}
#*************************************************************************
#************************* Tree Pruning **********************************
#*************************************************************************
 tree_pad =tree(Padwear~.-Padwear,data.train)
set.seed(1)
cv.tree_pad=cv.tree(tree_pad,FUN=prune.misclass)
#
# visualize the results
#
par(mfrow=c(2,1))
plot(cv.tree_pad$size,cv.tree_pad$dev,type="b")
plot(cv.tree_pad$k,cv.tree_pad$dev,type="b")
par(mfrow=c(1,1))
#
# prune.misclass() based on cv results
#
prune.tree_pad= prune.misclass(tree_pad,best =9)
#plot(prune.tree_pad)
#text(prune.tree_pad,pretty=0)
#
#test pruned tree
#
tree.class=predict(prune.tree_pad, data.test,type="class")
tree_pred=predict(prune.tree_pad, data.test,type="vector")[,2]

#After choosing the best model calculate the misclassification rate, sensitivity and specificity 
#*************************************************************************
conf<-table(tree.class,test_y)
spec_tree<-conf[1]/(conf[1]+conf[2])
sen_tree<-conf[4]/(conf[3]+conf[4])
mis_clas_tree<-mean(test_y!=tree.class)
```

## After pruning and using cross validation best tree was found to have 9 terminal nodes 

## Summary of best tree are:
```{r}
plot(prune.tree_pad)
text(prune.tree_pad,pretty=0)
#Misclassification error
mis_clas_tree
#Sensitivity
sen_tree
#Specificity
spec_tree
```


## 1.f Random Forest  
```{r}
#*************************************************************************
#************************* Random Forest *********************************
#*************************************************************************
set.seed(1)
ran_forest_mod<-randomForest(Padwear~.-Padwear,data=data.train,importance =TRUE)
oob_ranForest<-.3014
#OOB estimate of erro rate is 30.14% 
sort(ran_forest_mod$importance[,3],decreasing = TRUE)
par(mfrow=c(2,1))
barplot(ran_forest_mod$importance[,3],main="Decrease in Accuracy")
barplot(ran_forest_mod$importance[,4],main="Decrease in Gini index")
par(mfrow=c(1,1))
ranForest_pred<-predict(ran_forest_mod,newdata=data.test,type="prob")[,2]
ranForest_class<-predict(ran_forest_mod,newdata=data.test,type="class")

#After choosing the best model calculate the misclassification rate, sensitivity and specificity 
#*************************************************************************
conf<-table(ranForest_class,test_y)
spec_ranForest<-conf[1]/(conf[1]+conf[2])
sen_ranForest<-conf[4]/(conf[3]+conf[4])
mis_clas_ranForest<-mean(test_y!=ranForest_class)
```

## RMS.y and std.y the two top most important predictors w.r.t of decrease in error rate

## Summary of random forest
```{r}
ran_forest_mod
#OOB error rate =30.14 %
#Misclasification error rate
mis_clas_ranForest
#Sensitivity
sen_ranForest
#Specificity
spec_ranForest
```


## 1.g Bagging  
```{r}
#*************************************************************************
#************************* Bagging ***************************************
#*************************************************************************
set.seed(1)
bag_mod<-randomForest(Padwear~.-Padwear,data=data.train,mtry=20,importance =TRUE)
oob_bag<-.2945
#OOB estimate of erro rate is 29.45% 
sort(bag_mod$importance[,3],decreasing = TRUE)

bag_pred<-predict(bag_mod,newdata=data.test,type="prob")[,2]
bag_class<-predict(bag_mod,newdata=data.test,type="class")

#After choosing the best model calculate the misclassification rate, sensitivity and specificity 
#*************************************************************************
conf<-table(bag_class,test_y)
spec_bag<-conf[1]/(conf[1]+conf[2])
sen_bag<-conf[4]/(conf[3]+conf[4])
mis_clas_bag<-mean(test_y!=bag_class)
```

## RMS.y and std.y the two top most important predictors w.r.t of decrease in error rate

## Summary of Bagging 

```{r}
bag_mod
#OOB error rate= 29.45%
#Misclassification error rate
mis_clas_bag
#Sensitivity
sen_bag
#Specificity
spec_bag
```


## 1.h Support Vector Machine  -Linear
```{r}
#*************************************************************************
#************************* SVM Linear ************************************
#*************************************************************************
set.seed(1)
tune.out=tune(svm,Padwear~.-Padwear,data=data.train,kernel="linear",probability=TRUE,ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10,100)))
#summary(tune.out)
bestmod=tune.out$best.model
best_c<-tune.out$best.parameters[1]
cv.err_svm_lin<-cross_val_svm_l(xvars=names(data.train)[2:21],yvar="Padwear",dat=data.train,c=best_c)

p_buf<-predict(bestmod,data.test,probability = TRUE,decision.values = TRUE)
svm_lin_pred<-attributes(p_buf)$probabilities[,2]
#The following code was giving errorneous results
#svm_lin_pred=attributes(predict(bestmod,data.test,decision.values=TRUE))$decision.values

#After choosing the best model calculate the misclassification rate, sensitivity and specificity 
#*************************************************************************
predict_svm<-predict(bestmod,data.test)
conf<-table(predict_svm,test_y)
spec_svm_linear<-conf[1]/(conf[1]+conf[2])
sen_svm_linear<-conf[4]/(conf[3]+conf[4])
mis_clas_svm_linear<-mean(test_y!=predict_svm)
```

## Best SVM with linear kernel had cost= 0.01

## Summary SVM linear
```{r}
#Misclassification error rate
mis_clas_svm_linear
#Sensitivity
sen_svm_linear
#Specificity
spec_svm_linear
```


## 1.i Support Vector Machine  -Radial
```{r}
#*************************************************************************
#************************* SVM Radial ************************************
#*************************************************************************
tune.out=tune(svm,Padwear~.-Padwear,data=data.train,kernel="radial",probability=TRUE,ranges=list(cost=c(0.1,1,10,100,1000),gamma=c(0.5,1,2,3,4)))
#summary(tune.out)
bestmod=tune.out$best.model
best_c<-tune.out$best.parameters[1]
gam<-tune.out$best.parameters[2]

cv.err_svm_rad<-cross_val_svm_r(xvars=names(data.train)[2:21],yvar="Padwear",dat=data.train,c=best_c,gam=0.5)
#summary(bestmod)

p_buf<-predict(bestmod,data.test,probability = TRUE,decision.values = TRUE)
svm_rad_pred<-attributes(p_buf)$probabilities[,2]
#The following code was giving errorneous results
#svm_rad_pred=attributes(predict(bestmod,data.test,decision.values=TRUE))$decision.values
predict_svm<-predict(bestmod,data.test)

#After choosing the best model calculate the misclassification rate, sensitivity and specificity 
#*************************************************************************
conf<-table(predict_svm,test_y)
spec_svm_radial<-conf[1]/(conf[1]+conf[2])
sen_svm_radial<-conf[4]/(conf[3]+conf[4])
mis_clas_svm_radial<-mean(test_y!=predict_svm)
```

## Best SVM with Radial kernel had cost= 1 , gamma= 0.5

## Summary SVM Radial
```{r}
#Misclassification error rate
mis_clas_svm_radial
#Sensitivity
sen_svm_radial
#Specificity
spec_svm_radial
```

## 1.j Support Vector Machine  -Polynomial
```{r}
#*************************************************************************
#************************* SVM Polynomial ********************************
#*************************************************************************
tune.out=tune(svm,Padwear~.-Padwear,data=data.train,kernel="polynomial",probability=TRUE,ranges=list(cost=c(0.1,1,10,100,1000),degree=c(2,3)))
bestmod=tune.out$best.model

best_c<-tune.out$best.parameters[1]
best_deg<-tune.out$best.parameters[2]
cv.err_svm_poly<-cross_val_svm_p(xvars=names(data.train)[2:21],yvar="Padwear",dat=data.train,c=best_c,deg=best_deg)
#summary(bestmod)

p_buf<-predict(bestmod,data.test,probability = TRUE,decision.values = TRUE)
svm_poly_pred<-attributes(p_buf)$probabilities[,2]
#The following code was giving errorneous results
#svm_poly_pred=attributes(predict(bestmod,data.test,decision.values=TRUE))$decision.values
predict_svm<-predict(bestmod,data.test)

#After choosing the best model calculate the misclassification rate, sensitivity and specificity 
#************************************************************************
conf<-table(predict_svm,test_y)
spec_svm_poly<-conf[1]/(conf[1]+conf[2])
sen_svm_poly<-conf[4]/(conf[3]+conf[4])
mis_clas_svm_poly<-mean(test_y!=predict_svm)
```

## Best SVM with polynomial kernel had cost= 1 and degree= 3

## Summary SVM Polynomial
```{r}
#Misclassification error rate
mis_clas_svm_poly
#Sensitivity
sen_svm_poly
#Specificity
spec_svm_poly
```

#ROC Curves 
```{r}
#*************************************************************************
#************************* ROC Curve  ************************************
#*************************************************************************
library(ROCR)
rocplot=function(pred, truth, ...){
  predob = prediction(pred, truth)
  perf = performance(predob, "tpr", "fpr")
  plot(perf,...)}
```

# Q2.cont Based on detailed graphical and quantitative analysis, including cross validation studies,compare the performance of various classification methods.

```{r}
#************************************************************************
#************************* ROC Curve  ***********************************
#************************************************************************
rocplot(logis_pred,test_y,main="Logistic Regression")
auc <- performance(prediction(logis_pred,test_y),"auc")
auc_logis <- unlist(slot(auc, "y.values"))

rocplot(lda_pred,test_y,main="LDA")
auc <- performance(prediction(lda_pred,test_y),"auc")
auc_lda <- unlist(slot(auc, "y.values"))

rocplot(qda_pred,test_y,main="QDA")
auc <- performance(prediction(qda_pred,test_y),"auc")
auc_qda <- unlist(slot(auc, "y.values"))

rocplot(knn_pred,test_y,main="KNN")
auc <- performance(prediction(knn_pred,test_y),"auc")
auc_knn <- unlist(slot(auc, "y.values"))

rocplot(tree_pred,test_y,main="Pruned Tree")
auc <- performance(prediction(tree_pred,test_y),"auc")
auc_tree <- unlist(slot(auc, "y.values"))

rocplot(ranForest_pred,test_y,main="Random Forest")
auc <- performance(prediction(ranForest_pred,test_y),"auc")
auc_ranForest <- unlist(slot(auc, "y.values"))

rocplot(bag_pred,test_y,main="Bagging")
auc <- performance(prediction(bag_pred,test_y),"auc")
auc_bag <- unlist(slot(auc, "y.values"))

rocplot(svm_lin_pred,test_y,main="SVM Linear")
auc <- performance(prediction(svm_lin_pred,test_y),"auc")
auc_svm_lin <- unlist(slot(auc, "y.values"))

rocplot(svm_rad_pred,test_y, main="SVM Radial")
auc <- performance(prediction(svm_rad_pred,test_y),"auc")
auc_svm_rad <- unlist(slot(auc, "y.values"))

rocplot(svm_poly_pred,test_y,main="SVM Polynomial")
auc <- performance(prediction(svm_poly_pred,test_y),"auc")
auc_svm_poly <- unlist(slot(auc, "y.values"))
```

```{r, include=FALSE}
#*************************************************************************
#************************* Results ***************************************
#*************************************************************************
models<-c("Logistic","LDA","QDA","KNN","Prune Tree","Random Forest","Bagging","SVM Linear","SVM Radial","SVM Polynonial")
specificity<-rep(-1,10)
sensitivity<-rep(-1,10)
misclass<-rep(-1,10)
AUC<-rep(-1,10)
cv_error<-rep(NA,10)
results<-data.frame(cbind(models,specificity,sensitivity,misclass,cv_error,AUC))
```


```{r, warning=FALSE,include=FALSE}
results$AUC<-c(auc_logis,auc_lda,auc_qda,auc_knn,auc_tree,auc_ranForest,auc_bag,auc_svm_lin,auc_svm_rad,auc_svm_poly)
###############
results$misclass<-c(mis_clas_logis,mis_clas_lda,mis_clas_qda,mis_clas_knn,mis_clas_tree,mis_clas_ranForest,mis_clas_bag,mis_clas_svm_linear,mis_clas_svm_radial,mis_clas_svm_poly)

###############
results$cv_error<-c(cv.err_logis,cv.err_lda,cv.err_qda,cv.err_knn,999,oob_ranForest,oob_bag,cv.err_svm_lin,cv.err_svm_rad,cv.err_svm_poly)

###############
results$specificity<-c(spec_logis,spec_lda,spec_qda,spec_knn,spec_tree,spec_ranForest,spec_bag,spec_svm_linear,spec_svm_radial,spec_svm_poly)

###############
results$sensitivity<-c(sen_logis,sen_lda,sen_qda,sen_knn,sen_tree,sen_ranForest,sen_bag,sen_svm_linear,sen_svm_radial,sen_svm_poly)


r<-results

library(data.table)
results<-data.table(r)
results<-results[order(cv_error,(1-AUC),misclass,(1-sensitivity),(1-specificity))]
results<-results[,.(models,AUC,cv_error,misclass,sensitivity,specificity)]
results[cv_error==999,cv_error:=NA]
results[,AUC:=round(AUC*100,digits=2)]
results[,misclass:=round(misclass*100,digits=2)]
results[,sensitivity:=round(sensitivity*100,digits=2)]
results[,specificity:=round(specificity*100,digits=2)]
results[,cv_error:=round(cv_error*100,digits=2)]
setnames(results,"misclass","test_error")
```

# Q2.cont Based on detailed graphical and quantitative analysis, including cross validation studies,compare the performance of various classification methods.

Data frame containing the results for best model for each classification technique
```{r}
results
```
  
Based on the results the best logistic regression model gives the lowest cross validation error compared to all other classification techniques. Also, the area under the curve is second highest value (71.35%) for logisitic regression which is just slightly lower than the highest value (KNN AUC=74.85 %). However, when the model was tested on a validation set, it gave a higher test error.

Bagging was the second-best model w.r.t to OOB error. The AUC for bagging was high also.

Random Forest was ranked third based on the cross-validation error. It has similar results as Bagging.

KNN has a similar CV error as random forest. It has the highest AUC among all the models. 

QDA has a comparable AUC as four model discussed above. It provided a very high sensitivity of 80.00%.

LDA's performace was similar to QDA.

SVM linear outpreformed SVM with raidal or polynomial kernel w.r.t CV error and AUC.

Prunned tree preformed worst. It had an AUC and validation set error of almost 50%.


# Q3. Based on your analysis suggest and justify the model that the industry must use.

Based on my analysis I would suggest using the following logistic regression model:

logit(p(x)=1)= -2.47 - 5.654 std.y + 2.962 std.x - 0.76 crest.f.x + 0.6032 kurtosis.y - 0.881 impulse.f.y 

This model had the lowest AIC among the other logistic regression model I tried and all the predictors were statistically significant at 95% CL for this model. Moreover, it had a high AUC, lowest CV error and reasonable sensititvity and specificity at 0.5 threshold.

# Q4. Interpret the results of classification.

Based on the logistic model as std.y or/and impulse.f.y increaes the probability of a padwear decreases. The probability of padwear increases as std.x, crest.f.x or/and kurtosis.y increases. std.y and std.x have big impact on the probability of padwear. 

Based on classification tree following interpretations can be made:
  1. std.y is the most important factor in finding out padwear. The same conclusion can be drawn by looking at the importance table in random forrest. 
  2. std.y greater than 1.13 will most likely lead to no padwear.
  3. There is a big chance of padwear if std.y < 1.13 and pv.y < 1.47.
  4. Values of std.x, skewness.y, kurtosis.x, std.y and kurtosis.y would be helpful in finding padwear if std.y<1.13 and pv.y >1.47. 


#Q5. Justify the choice of the parameters employed for the best model.

I choose the parameters for the logistic regression based on the boxplots, importance results from random forest, prunned tree for CART,  stepwise selection method and by looking at the AIC values. 

std.y, std.x and kurtosis.y were found to have high importance in random forest. 
Also, boxplots of std.x, std.y , crest.f.x, kurtosis.y and impulse.f.y show that they have very differnt average value when there is padwear. 

Moreover, all the predictors with statistically signficant for my best logistic model at 95% CL. 

 
****

# Appendix 1: Functions for K Fold Cross Validation 


```{r}
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

```

