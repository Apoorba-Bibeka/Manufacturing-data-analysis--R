```{r setup, include=FALSE}
#Housekeeping
rm(list=ls())
library(data.table)
library(car)
knitr::opts_chunk$set(echo = TRUE)
getwd()
setwd("/Users/Apoorb/Dropbox/PHD_course_work/ISEN-613/Exam")
data<-fread("Training.csv")
```
  
#   Modelling Process   
###I first divided the data into two parts: train and test data. I added 80% of the data in train data set and 20% of the data in test data set. 
###Following code shows how I divided the data. 
```{r tidy=TRUE}
train_data<-data[1:(.8*nrow(data))]
test_data<-data[(.8*nrow(data)+1):nrow(data)]
test_y<-test_data[,F1]
test_data<-subset(test_data,select=1:14)    
```
###After dividing the data into train and test data, I checked the relationship between all the variables by plotting all the variables with each other.  
```{r echo=FALSE, tidy=TRUE}
plot(train_data)    
```
  
###From the plots, I found that the predictor variables F1Lx where x:{1.2...7} have a linear relationship with F1. However these predictor varibles looked to be highly correlated.   
###Predictor varibles F2Lx where x:{1,2.....7} did not show any noticable trend w.r.t F1 so I tried to tranform these predictors. After logrithmically transforming these varibles I noticed somewhat linear trend so I used log transformed F2Lx variables.  


###After seeing the relationship between the predictors and response, I decided to carry out forward selection to decide which regression model to use. I started with F1L1 predictor, as it seemed to be highly correlated with F1.   

###The following section shows the forward selection method:
```{r}
lm.fit1<-lm(F1~F1L1+F1L2,data=train_data)  #Reject-vif is high. Variables are correlated.
lm.fit1<-lm(F1~F1L1+F1L2,data=train_data)  #Reject-vif is high. Variables are correlated.
lm.fit1<-lm(F1~F1L1+F1L2+F1L3,data=train_data) #Bad model - High colinearity 
lm.fit1<-lm(F1~F1L1+F1L4,data=train_data) #very good low VIF ~2.6. Keep F1L1 And F1L4
lm.fit1<-lm(F1~F1L1+F1L4+F1L5,data=train_data) #Remove F1L5- Not significant 
lm.fit1<-lm(F1~F1L1+F1L4+F1L6,data=train_data) #High collinearity 
#Remove F1L6
lm.fit1<-lm(F1~F1L1+F1L4+F1L7,data=train_data) #Remove F1L4 -Not significant 


lm.fit1<-lm(F1~F1L1+F1L3,data=train_data)  #Might be good (VIF~4)
lm.fit1<-lm(F1~F1L1+F1L7,data=train_data)
#From the above two model, models with F1L7 has higher R2 so I choose that model. 

#The following models have insignificant coefficients for predictors other than F1L1 and F1L7:
lm.fit1<-lm(F1~F1L1+F1L7+F1L2,data=train_data)
lm.fit1<-lm(F1~F1L1+F1L7+F1L3,data=train_data)
lm.fit1<-lm(F1~F1L1+F1L7+F1L4,data=train_data)
lm.fit1<-lm(F1~F1L1+F1L7+F1L5,data=train_data)
lm.fit1<-lm(F1~F1L1+F1L7+F1L6,data=train_data)
lm.fit1<-lm(F1~F1L1+F1L7+log(F2L1),data=train_data)
lm.fit1<-lm(F1~F1L1+F1L7+log(F2L2),data=train_data)
lm.fit1<-lm(F1~F1L1+F1L7+log(F2L3),data=train_data)
lm.fit1<-lm(F1~F1L1+F1L7+log(F2L4),data=train_data)
lm.fit1<-lm(F1~F1L1+F1L7+log(F2L5),data=train_data)
lm.fit1<-lm(F1~F1L1+F1L7+log(F2L6),data=train_data)
lm.fit1<-lm(F1~F1L1+F1L7+log(F2L7),data=train_data)


```


###The following section shows the best linear  model I found from forward selection. 
```{r tidy=TRUE}
lm.fit<-lm(F1~F1L1+F1L7,data=train_data)
```
###Next, I tried a interaction model for the above predictors.  
```{r}
lm.fit1<-lm(F1~F1L1*F1L7,data=train_data)
summary(lm.fit1)
```
  
###The interaction model impoved upon the linear model. I then conducted ANOVA analysis to check if the coefficient of interaction term is significant. 

```{r tidy=TRUE, echo=FALSE}
anova(lm.fit,lm.fit1)
```
   
###The ANOVA test showed that the interaction term is significant.  

###After arriving at the best possible model in terms of model fitness, I next carried out different diagnostic tests. Following are the results of different diagnostic tests.    
```{r echo=FALSE}
resid<-rstandard(lm.fit1)
no<-seq(1,length(resid),1)
da_resid<-data.table(cbind(no,resid))
da_resid[abs(resid)>3,]

par(mfrow=c(2,2))
plot(lm.fit1)
par(mfrow=c(2,1))
plot(resid(lm.fit1),type='l')
plot(resid(lm.fit1))
#It plots the model parameters with (component+residual) and shows if we #have a significant non-linear effect 

#crPlots(lm.fit1)
#Checking the collinearity
vif_mod<-vif(lm.fit1)     
```

### The vif is    
```{r echo=FALSE}
vif_mod    
```


###According to the residuals point 41 and 100 have high levrage, so I removed these points.
###According to the residuals, points 34 and and 171 are outlier, so I removed these points also.
###After removing the above points I refitted the model 
```{r tidy=TRUE}
train_data1<-train_data[-c(34,41,100,171),]
lm.fit1<-lm(F1~F1L1*F1L7,data=train_data1)
summary(lm.fit1)
```
###On removing the outliers and high leverage points the interaction term and F1L7 became insignificant so I selected the model without interaction over the model with interaction.

###The following model was selected as the final model

```{r tidy=TRUE}
lm.fit1<-lm(F1~F1L1+F1L7,data=train_data)
summary(lm.fit1)   
```

###The diagnostics for above model are as follows: 

```{r echo=FALSE}
resid<-rstandard(lm.fit1)
no<-seq(1,length(resid),1)
da_resid<-data.table(cbind(no,resid))
da_resid[abs(resid)>3,]

par(mfrow=c(2,2))
plot(lm.fit1)
par(mfrow=c(2,1))
plot(resid(lm.fit1),type='l')
plot(resid(lm.fit1))
#It plots the model parameters with (component+residual) and shows if we #have a significant non-linear effect 

crPlots(lm.fit1)
#Checking the collinearity
vif_mod<-vif(lm.fit1)   
```
### The vif is   
```{r echo=FALSE}
vif_mod   
```

###From the diagnostic test it can be seen that the data has a constant mean of zero. Also, the variance looks to be constant for the residuals. However, there is clear trend in the residuals indicating that the data might be correlated.  From the Q-Q plots it can be seen that the normality assumption is also violated. 

###From car plot, it looks like that the predictor varible do not have a significant non linear effect. 

###VIF for both the predictors is less than 4 thus indicating that there is no evidence of collinearily between the two predictors. 

###Next, I check the prediction accuracy using the test data.
```{r}
pred_y<-predict(lm.fit1,newdata=test_data)
plot(pred_y,test_y)
abline(lm(test_y~pred_y),col="blue")
#Get the R2 for the predicted value 
ss_t<-sum((test_y-mean(test_y))^2)
ss_r<-sum((test_y-pred_y)^2)
r2<-1-(ss_r/ss_t)
r2
```

###From the above results we see that the R^2^ for the predicted and actual value for test data is 78.81%.
  
*********   

#Modeling Process  
###backward selection modeling process. 

```{r}
lm.fit1<-lm(F1~F1L3+F1L4+F1L5+F1L6+F1L7,data=train_data) #Remove variables with high p-values
lm.fit<-lm(F1~F1L3+F1L7,data=train_data)  #The best model with all significant terms 
lm.fit1<-lm(F1~F1L3+F1L7+I(F1L7^2),data=train_data) #Try a hybrid model 
lm.fit1<-lm(F1~F1L3*F1L7,data=train_data) #Try interaction model
#The interaction model has higer R2 than the hybrid model so choose the interaction model

 #Check if interaction model improves upon the additive model. 
anova(lm.fit,lm.fit1) 
#From F-test, we see that the interaction model improves upon the hybrid model so choose the interaction model. 

#Remove leverage points and outliers and refit the model 
train_data1<-train_data[-c(34,35,101,105),]
lm.fit1<-lm(F1~F1L3*F1L7,data=train_data1)
summary(lm.fit1)
```

###I choose the model with F1L3 and F1L7 and their interaction term. 

###Next I carried out the diagnostics on the model. 

```{r echo=FALSE}
resid<-rstandard(lm.fit1)
no<-seq(1,length(resid),1)
da_resid<-data.table(cbind(no,resid))
da_resid[abs(resid)>3,]

par(mfrow=c(2,2))
plot(lm.fit1)
par(mfrow=c(2,1))
plot(resid(lm.fit1),type='l')
plot(resid(lm.fit1))

#Checking the collinearity
vif_mod<-vif(lm.fit1)   
```
### The vif is    
```{r echo=FALSE}
vif_mod    
```

###From the diagnostic test it can be seen that the data does not have a constant mean of zero. Also, the variance of the residuals increaes with the fitted values.There is clear trend in the residuals indicating that the data might be correlated.  From the Q-Q plots it can be seen that the normality assumption is also violated. 

###Car plots for checking non linearity are not available for models with interaction. 

###VIF for all the predictors is more than 4 thus indicating that there is collinearily between the predictors. However, since we have an interaction term in the model, in makes sense for the collinearity to be high. 


###Next, I check the prediction accuracy using the test data.
```{r}
pred_y<-predict(lm.fit1,newdata=test_data)
plot(pred_y,test_y)
abline(lm(test_y~pred_y),col="blue")
#Get the R2 for the predicted value 
ss_t<-sum((test_y-mean(test_y))^2)
ss_r<-sum((test_y-pred_y)^2)
r2<-1-(ss_r/ss_t)
r2
```

###From the above results we see that the R^2^ for the predicted and actual value for test data is 28.18%.
  
*********      
  
  
#
##Modeling Process  
###backward selection modeling process. I tried various hybrid models as the car plots was showing non linear relationship between predictors. I finally choose the model which was parsimonious, had high R^2^2 and had better diagnostic results as compared to the previous models.

```{r}
lm.fit1<-lm(F1~(F2L3)+(F2L4)+(F2L5)+(F2L6)+(F2L7),data=train_data)
lm.fit1<-lm(F1~(F2L3)+F2L4+F2L7,data=train_data)
lm.fit1<-lm(F1~F2L3+I(F2L3^2)+F2L4+I(F2L4^2)+F2L7,data=train_data)
lm.fit1<-lm(F1~log(F2L3)+log(F2L4)+log(F2L7),data=train_data)
lm.fit1<-lm(F1~log(F2L3)+log(F2L4),data=train_data)
summary(lm.fit1)
```

###I choose the model with log(F2L3) and log(F2L4). Next I carried out the diagnostics on the model. 

```{r echo=FALSE}
resid<-rstandard(lm.fit1)
no<-seq(1,length(resid),1)
da_resid<-data.table(cbind(no,resid))
da_resid[abs(resid)>3,]

par(mfrow=c(2,2))
plot(lm.fit1)
par(mfrow=c(2,1))
plot(resid(lm.fit1),type='l')
plot(resid(lm.fit1))

crPlots(lm.fit1)
#Checking the collinearity
vif_mod<-vif(lm.fit1)    
```

### The vif is   
```{r echo=FALSE}
vif_mod    
```

###From the diagnostic test it can be seen that the data does not have a constant mean of zero. Also, the variance of the residuals increaes with the fitted values.There is clear trend in the residuals indicating that the data might be correlated.  From the Q-Q plots it can be seen that the points at the tail do not follow a normal distribution thus the normality assumption is also violated.

###From the car plots it can be observed that the true relationship might be non linear. 

###VIF for all the predictors is less than 4 thus there is no evidence for collinearity. 

###Next, I check the prediction accuracy using the test data.
```{r}
pred_y<-predict(lm.fit1,newdata=test_data)
plot(pred_y,test_y)
abline(lm(test_y~pred_y),col="blue")
#Get the R2 for the predicted value 
ss_t<-sum((test_y-mean(test_y))^2)
ss_r<-sum((test_y-pred_y)^2)
r2<-1-(ss_r/ss_t)
r2
```

###Getting erroneous R^2^ for the above test data. 


*********     

###Based on the R^2^ values of the fitted models and for the test data and residual diagnostics, I choose the model from Q1 as the final model. Model from Q1 met the constant mean=0 and constant variance assumption, it had few outliers. There were few points with high leverage.  Also the R^2^ for training data was 92.14% and for test data was 78.81% which is high as compared to be models in Q2. Q1 model outperformed the Q2 models both in terms of R^2^ and diagnostics.

###The final model is :
```{r}
lm.fit1<-lm(F1~F1L1+F1L7,data=train_data)
summary(lm.fit1)   
```
###According to the above model, a 1 unit increase in F1L1 results is 0.998 unit increase in F1 and 1 unit increase in F1L7 results in 0.0273 unit decrease in F1. 

*****





