# Project 2 Edwisor - Cab Fare Prediction

# Instruction to run code
# You need to set the working directory path in the following code
# And the working directory that you have set must have train_cab and test data sets

# Lets clear the environment
rm(list=ls())
# set working directory
setwd('F:\\data Scientist\\project 2 Cab Fare Prediction  R')
getwd()
 
# lets load the libraries 
# lets save the libraries in x variable
x= c("ggplot2","corrgram","DataCombine","scales","psych","gplots",
     "Metrics" ,"inTrees","DMwR","car", "caret","rlang","usdm", "Information", "randomForest", 
     "unbalanced", "C50", "dummies", "e1071", "MASS", "ROSE", "rpart", "gbm",'xgboost','stats')
# lets load the libraries
#install.packages("psych")
lapply(x, require, character.only=TRUE)
rm(x)


# lets load the data 
# the data given to us is train_cab and test data
train_cab= read.csv("train_cab.csv", header=T, na.strings = c(' ','', 'NA'))
test_cab= read.csv('test.csv', )

#### features in given data are as follows in problem statement
# pickup_datetime - timestamp value indicating when the cab ride started.
# pickup_longitude - float for longitude coordinate of where the cab ride started.
# pickup_latitude - float for latitude coordinate of where the cab ride started.
# dropoff_longitude - float for longitude coordinate of where the cab ride ended.
# dropoff_latitude - float for latitude coordinate of where the cab ride ended.
# passenger_count - an integer indicating the number of passengers in the cab


# lets observe the given data 
str(train_cab)
str(test_cab)
# we can see the shape of the train_cab data there are 16067 rows and 7 variables including fare_amount as a target varibale
# test data has 9914 rows and 6 variables


# summary of train_cab data
summary(train_cab)

##### We can see summary of train data 
##### Findings of summary
# pickup_datetime variable is timestamp variable which in object data type
# Also we know that longitude ranges from (-180 to +180) and latitude from (-90 to +90)
# pickup_latitude max value is 401.08 which is above 90 so we need to look into this issue.
# passenger_count variable has minimum value 1 and max 5345 which is not possible
# fare_amount is price variable it should be numeric also it contains negative values
# there are missing values

# summary of train_cab data
summary(test_cab)

# Findings for test_cab data
# all varibales are within range 
# just we need to convert pickup_datetime variable to timestamp variable

# lets see top 5 observation of train and test data
head(train_cab)
head(test_cab)

# lets first convert variables in both data to proper shapes 
# lets convert pickup_datetime to date time variable form both data

str(train_cab$pickup_datetime)

# now lets convert fare_amount to numeric
train_cab$fare_amount = as.numeric(as.character(train_cab$fare_amount))
str(train_cab$fare_amount)

# lets round off the passenger_count variable values from both data sets as it is person number
train_cab$passenger_count=round(train_cab$passenger_count)
test_cab$passenger_count=round(test_cab$passenger_count)

# check the data types again
str(train_cab)
str(test_cab)

# now all the variables are in proper data types lets move further

######## EXPLORATORY DATA ANALYSIS
# We observed that variables are not within the standard range 
# so lets do some data cleaning operations on the data

### passenger_count varibale

summary(train_cab$passenger_count)
# we can see minimum value is 0 and maximum is 5345 which is not possible so we will remove those observations
# We will assume maximum 6 passengers in cab so outside form 1 to 6 we will remove all observations
# lets check count of values below 1
nrow(train_cab[which(train_cab$passenger_count<1),])
# count is 58

# lets check count of values above 6 
nrow(train_cab[which(train_cab$passenger_count>6),])
# count is 20

# We can not impute these values of we will remove these 78 observations 
train_cab = train_cab[-which(train_cab$passenger_count > 6),]
train_cab = train_cab[-which(train_cab$passenger_count < 1 ),]

# unique values in train data for passenger_count
unique(train_cab$passenger_count)

# unique values in test data for passenger_count
unique(test_cab$passenger_count)

# So we have cleaned the passenger_count variable form train and test data lets move further

## fare_amount variable

summary(train_cab$fare_amount)
# we can see minimum value is negative and maximum is 54343 so we will remove these observations with negative values

# lets sort in descending order
sort(-train_cab$fare_amount)
# we can see values above 453 are above range or not practical 

# lets check count of values below 1
nrow(train_cab[which(train_cab$fare_amount<1),])
# count is 5

# lets check count of values above 6 
nrow(train_cab[which(train_cab$fare_amount>453),])
# count is 2
# So we will remove the  7 observations in fare_amount below 1 and above 453
train_cab = train_cab[-which(train_cab$fare_amount > 453),]
train_cab = train_cab[-which(train_cab$fare_amount < 1 ),]
summary(train_cab$fare_amount)

# now we can see the values are in the range.

#### Lets do EDA on latitude and longitude variables from train and test data
# we already seen that only pickup_latitude variable is having value above range


# pickup_latitude varibale

summary(train_cab$pickup_latitude)
# we can see maximum value is 401.88 which is above range 
nrow(train_cab[which(train_cab$pickup_latitude>90),])

# we can see there is only one observation above 90 so we will remove it
train_cab = train_cab[-which(train_cab$pickup_latitude > 90),]
summary(train_cab)

# Now we will check all 4 latitude and longitude variables for presence of "0" values.
nrow(train_cab[which(train_cab$pickup_longitude == 0 ),])
nrow(train_cab[which(train_cab$pickup_latitude == 0 ),])
nrow(train_cab[which(train_cab$dropoff_longitude == 0 ),])
nrow(train_cab[which(train_cab$pickup_latitude == 0 ),])

# we can see there are observations with "0" values 311,311,312 and 311 
# so we will remove all these observations from each variable which are having "0" values
train_cab = train_cab[-which(train_cab$pickup_longitude == 0),]
train_cab = train_cab[-which(train_cab$dropoff_longitude == 0),]

# Lets check for test data
summary(test_cab)
# we can see there is no variable outside the range for test data so we will proceed further

#### Now we have cleaned both the data sets lets check it for missing values

#### MISSING VALUE ANALYSIS

# In this step we will find the variables with missing values.
# If missing values are present then we will impute them. 

# create dataframe with missing value count
missing_val = data.frame(apply(train_cab,2,function(x){sum(is.na(x))}))

#convert row names into columns
missing_val$Columns= row.names(missing_val)
row.names(missing_val) = NULL

# rename the missing value count variable name as missing percentage
names(missing_val)[1]= "Missing_Percentage"

# calculate missing values percentage
missing_val$Missing_Percentage= (missing_val$Missing_Percentage/nrow(train_cab))*100 

# lets arrange the Missing_Percentage column in descending order
missing_val=missing_val[order(missing_val$Missing_Percentage),]

# rearrange the columns
missing_val= missing_val[,c(2,1)]

# lets save the data frame of missing values in our working directory
write.csv(missing_val, "Missing_percentage.csv", row.names=F)

# lets plot the bar graph of missing values w.r.t variable
ggplot(data=missing_val[6:7, ], aes(x=reorder(Columns, -Missing_Percentage), y=Missing_Percentage))+ 
  geom_bar(stat='identity', fill="DarkSlateBlue") + 
  xlab("variables")+ ylab("Missing_percentage")+  ggtitle("Missing values of train_cab")+theme_bw()

# lets create copy of the data set to use it for checking which method is good to impute missing values.
train_cab_missingdata=train_cab
#train_cab=train_cab_missingdata

# Procedure to Impute missing values is as follows
# 1) Select any random observation from data and equal it with "NA"
# 2) Now impute that value by using mean, mode, median, KNN 
# 3) Compare the value imputed by above methods with actual value.
# 4) select the method which will give more accurate result
# 5) Now choose that method and find all missing values in that variable.
# 6) Repeat above steps t0 impute all missing values.

## Mean and median are used for numeric variales
## Mode and KNN are used for Categorical variables

# Lets impute missing values in passenger_count variable
# we will check only with KNN 
# we will not use mode because we have most of the observations having passenger count value '1'

#----------------------------------------------------------------------------------------------
# passenger_count-
# Actual `value-1 
# KNN- 1 

# lets select any random observation
train_cab$passenger_count[900] 
train_cab$passenger_count[900]= NA

# lets impute with KNN
#train_cab= knnImputation(train_cab, k=5)
train_cab$passenger_count[900]
#--------------------------------------------------------------------------------------------------
# so we have finilized the KNN as imputation method for passenger_count now lets check for fare_amount variable.


# Lets impute missing values in fare_amount variable 
# first lets check which method is giving good accuracy from mean, median, KNN method

#-------------------------------------------------------------------------------------------------
# lets select any random observation
train_cab$fare_amount[900]
train_cab$fare_amount[900]= NA

##### fare_amount
# Actual value=49.8
# mean= 11.3667   
# median= 8.5    
# KNN= 48.0639

# lets impute with mean
train_cab$fare_amount[is.na(train_cab$fare_amount)]=mean(train_cab$fare_amount, na.rm=T)
train_cab$fare_amount[900]
train_cab$fare_amount[900]= NA


# lets impute with medain
train_cab$fare_amount[is.na(train_cab$fare_amount)]=median(train_cab$fare_amount, na.rm=T)
train_cab$fare_amount[900]
train_cab$fare_amount[900]= NA


# lets make passenger_count to numeric 
#train_cab$passenger_count=as.numeric(train_cab$passenger_count)

# lets impute with KNN
#train_cab= knnImputation(train_cab, k=5)
train_cab$fare_amount[900]
#-----------------------------------------------------------
### We can see KNN imputation method is giving great accuracy to impute missing values
# So we will freeze this method for calculating missing values. 

# lets impute all missing values with KNN
str(train_cab)
train_cab= knnImputation(train_cab, k=5)

# lets recheck the missing values
sum(is.na((train_cab)))

# lets check test data for missing values present or not
sum(is.na((test_cab)))

# lets round off the passenger_count variable values from both data sets
train_cab$passenger_count=round(train_cab$passenger_count)
test_cab$passenger_count=round(test_cab$passenger_count)



######## FEATURE ENGINEERING 

# If we see our variables we have date_time variable which contains information of day,month,year and time at that point. 
# But These all are enclosed in one format.
# so we can not use that as it is and our ML model didnt regognise this variable 
# so we will split that variable 
# create new variables like date, month, year, weekday, hour and minute.

# Similarly if we see other four variables pickup_longitude,pickup_latitude,dropoff_longitude, dropoff_latitude
# these are nothing but the coordinates of passenger where from he is picked up and dropped by cab. 
# But we cannot use these variables also for our ML model. 
# As we are having passenger pickup and drop coordinates.
# we can create new variable from this that is 'DISTANCE 'distance' variable. 
# It will give us the distance travelled by cab during each ride.
# Distance variable will be more important to decide Cab fare amount.

######## Feature Engineering on train data

# lets first do feature engineering on pickup_datetime varible from train data
str(train_cab$pickup_datetime)

#lets first convert variable from factor to date time variable 
# lets create raw_date variable to extract and create date, month and year variables from given date.
train_cab$raw_date= as.Date(train_cab$pickup_datetime)

# lets check it again for missing values
sum(is.na(train_cab$raw_date))

# we can see there are NA value in raw_date variable
# lets remove that rows having NA values
train_cab=train_cab[complete.cases(train_cab[,8]),]
# lets check again for missing values
sum(is.na(train_cab$raw_date))
sum(is.na(train_cab))

# so now we dont have missing values 
# lets go further for creating new features from raw_date and pickup_datetime varibale
head(train_cab)

# Now lets extract date, month, year from this raw_date time variable
# lets extract date from pickup_datetime
train_cab$date= as.integer(format(train_cab$raw_date,'%d' ))

# lets extract month from pickup_datetime
train_cab$month= as.integer(format(train_cab$raw_date,'%m' ))

# lets extract year from pickup_datetime
train_cab$year= as.integer(format(train_cab$raw_date,'%Y' ))


# lets create hour and weekday variables from given pickup_datetime variable
# lets extract hour from pickup_datetime
train_cab$hour = substr(as.factor(train_cab$pickup_datetime),12,13)

# lets extract weekday from pickup_datetime
train_cab$weekday= as.factor(format(train_cab$raw_date, '%u'))


# lets see the unique values in each new variables that we have created
head(train_cab)
unique(train_cab$date)
unique(train_cab$month)
unique(train_cab$year)
unique(train_cab$hour)
unique(train_cab$weekday)
# Here in weekday we will consider value 1 as monday and so on. 
# lets check no of unique values in each new created variables
table(train_cab$date)
table(train_cab$month)
table(train_cab$year)
table(train_cab$hour)
table(train_cab$weekday)


###### Feature Engineering on test data

# lets first do feature engineering on pickup_datetime varible from test data
str(test_cab$pickup_datetime)

# lets see first 5 observations of data
head(test_cab)

#lets first convert variable from factor to date time variable 
# lets create raw_date variable to extract and create date, month and year variables from given date time variable.
test_cab$raw_date= as.Date(test_cab$pickup_datetime)

# lets check it again for missing values
sum(is.na(test_cab$raw_date))

# no missing values lets proceed further

# Now lets extract date, month, year from this raw_date variable
# lets extract date from pickup_datetime
test_cab$date= as.integer(format(test_cab$raw_date,'%d' ))

# lets extract month from pickup_datetime
test_cab$month= as.integer(format(test_cab$raw_date,'%m' ))

# lets extract year from pickup_datetime
test_cab$year= as.integer(format(test_cab$raw_date,'%Y' ))

# lets create hour and weekday variables from given pickup_datetime variable

# lets extract hour from pickup_datetime
test_cab$hour = substr(as.factor(test_cab$pickup_datetime),12,13)


# lets extract weekday from pickup_datetime
test_cab$weekday= as.factor(format(test_cab$raw_date, '%u'))

head(test_cab)

##### Now we have finished feature engineering of pickup_datetime variable for bot train and test data
# So now we dont require pickup_datetime variable as we have created new feature form it
# So we will remove it 
# Also we have created raw_date variable, we will remove that also
# we will remove above 2 variables from train and test data

train_cab=subset(train_cab, select= -c(pickup_datetime,raw_date))
test_cab=subset(test_cab, select= -c(pickup_datetime,raw_date ))
head(train_cab)

####### Lets do feature engineering of latitude and longitude variables
# we already discussed above that we can create distance variable from latitude and longitude variables
# So lets create distance variable in both train and test data
# we will use haversine formula to calculate distance between two points of lat. and long.
# Haversine formula calculates great circle distance on sphere with given lat and long. 


# lets create function to convert decimal degrees to radians
Radians= function(Deg){
  (Deg*pi)/180
}

### lets write Haversine formulla to calculate distance 
# lets create function for Haversine formula and store in Haversine object
Haversine= function(pi_lon,pi_lat, dr_lon, dr_lat){
  
  ### where pi_lon=pickup_longitude,    pi_lat= pickup_latitude, 
  ###       dr_lon= dropoff_longitude,  dr_lat=dropoff_latitude
  
  # now lets convert pickup_latitude and dropoff_latitude to radians and save in pi_lon_rad and dr_lon_rad to objectpi_lon_rad=Radians(pi_lon) 
  dr_lon_rad=Radians(dr_lon)
  
  # lets substract them and save in sub_lon
  sub_lon= Radians(dr_lon-pi_lon)  
  
  # now lets convert pickup_latitude and dropoff_latitude to radians and save in pi_lat_rad and dr_lat_rad to object
  pi_lat_rad=Radians(pi_lat) 
  dr_lat_rad=Radians(dr_lat)
  
  # lets substract them and save in sub_lat
  sub_lat= Radians(dr_lat-pi_lat)
  
  # lets write the formula
  H=sin(sub_lat/2)*sin(sub_lat/2)+ cos(pi_lat_rad)*cos(dr_lat_rad)* sin(sub_lon/2)* sin(sub_lon/2)
  A=2*atan2(sqrt(H), sqrt(1-H))
  R=6371e3
  R*A/1000
}

## lets apply the Haversine formula on our train data

train_cab$distance= Haversine(train_cab$pickup_longitude, train_cab$pickup_latitude,
                               train_cab$dropoff_longitude, train_cab$dropoff_latitude)

# lets apply on test data
test_cab$distance= Haversine(test_cab$pickup_longitude, test_cab$pickup_latitude,
                              test_cab$dropoff_longitude, test_cab$dropoff_latitude)

# lest see top 5 observations of train data
head(train_cab)

# lets see structure of distance variable from train and test data
str(train_cab$distance)
str(test_cab$distance)

# lets see the summary of distance variable 
summary(train_cab$distance)
## we can see minimum value is 0.00 and maximum is 5420.989 which is not practical

## lets sort the distance variable values in descending order
train_cab[order(-train_cab$distance),]

## we can see the value after 129.95 are increased highly so we will remove values above 130 
## so we need to remove these observations

#### lets check no of values above '130' and equal to '0' in train data
nrow(train_cab[which(train_cab$distance == 0 ),])
nrow(train_cab[which(train_cab$distance >130 ),])
### there are total 157 observations so we will remove them.

# so we will remove all these observations which are having "0" values and above'130'
train_cab = train_cab[-which(train_cab$distance == 0),]
train_cab = train_cab[-which(train_cab$distance >130),]
str(train_cab)

## lets convert variables to factor
factor_data= colnames(train_cab[,c('date','month', 'year', 'weekday', 'hour', 'passenger_count' )])

for (i in factor_data){
  train_cab[,i]=as.factor(train_cab[,i])
}

# lets check the missing values
sum(is.na(train_cab))

##### so now we have completed the feature engineering of longitude ant latitude variables 
# we can now remove pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude  variables
# Because we have created distance variable from those variables
# lets remove from train data adn test data
train_cab=subset(train_cab, select= -c(pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude ))
test_cab=subset(test_cab, select= -c(pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude ))

# top 5 observations
head(train_cab)
head(test_cab)

# now we have 8 variables in train data and 7 variables in test data after feature engineering

## lets convert the factor variable to numeric for data visualization.
factor_data= colnames(train_cab[,c('date','month', 'year', 'weekday', 'hour', 'passenger_count' )])

for (i in factor_data){
  train_cab[,i]=as.numeric(train_cab[,i])
}

########## DATA VISUALIZATION
## In this section we will visualize our train data with respect to fare_amount variable

##### 'passenger_count' variable
## Lets plot histogram of passenger_count variable
hist(train_cab$passenger_count)
# Finding-  single travelling passengers are most frequent travellers.

## Lets plot scatter plot of passenger_count variable and fare_amount
ggplot(train_cab,aes(passenger_count,fare_amount)) + 
  geom_point(alpha=0.5,color="DarkSlateBlue") +
  labs(title = "Scatter Plot b/w passenger_count and fare_amount", x = "passenger_count", y = "fare_amount")+
  scale_color_gradientn(colors=c('light green')) +
  theme_bw()
# Finding- We can see highest fare is from single passenger.

##### 'weekday' variable
### Lets plot scatter plot of weekday variable adn fare_amount
ggplot(train_cab,aes(weekday,fare_amount)) + 
  geom_point(alpha=0.5,color="DarkSlateBlue") +
  labs(title = "Scatter Plot b/w weekday and fare_amount", x = "weekday", y = "fare_amount")+
  scale_color_gradientn(colors=c('blue')) +
  theme_bw()
# Finding- Monday to friday cab fare is high

##### 'hour' variable
### Lets plot scatter plot of hour variable and fare_amount
ggplot(train_cab,aes(hour,fare_amount)) + 
  geom_point(alpha=0.5,color="DarkSlateBlue") +
  labs(title = "Scatter Plot b/w hour and fare_amount", x = "hour", y = "fare_amount", xlab("hour"), ylab("fare_amount"))+
  scale_color_gradientn(colors=c('blue')) +
  theme_bw()
# Finding- After 8 pm cab fare charge is higher

#### 'distance' variable
### Lets plot scatter plot of distance variable and fare_amount
ggplot(train_cab,aes(distance,fare_amount)) +   geom_point(alpha=0.5,color="DarkSlateBlue") + 
  labs(title = "Scatter Plot b/w distance and fare_amount",x = "distance",y = "fare_amount")+
  scale_color_gradientn(colors=c('blue')) +  theme_bw()
# Finding- There is a linear relatinship between distance and fare_amount
# so distance variable is very important w.r.t fare_amount


############# FEATURE SELECTION

# lets convert our categorical variables to factor agian for feature selection techniques.
factor_data= colnames(train_cab[,c('date','month', 'year', 'weekday', 'hour', 'passenger_count' )])

for (i in factor_data){
  train_cab[,i]=as.factor(train_cab[,i])
}

###### Correlation plot 
## By this plot we can check the correlation between variables

### lets select the numeric vaiables from train data
numeric_index = sapply(train_cab,is.numeric)
numeric_data = train_cab[,numeric_index]

# Correlation plot to check correlation of variables
corrgram(train_cab[,numeric_index],order=F, upper.panel=panel.pie , main="correlation plot")
#### we can see in plot distance and fare_amount are very highly correlated witrh each other

##### ANOVA test

### This test is perfomed to check weather the means of categories of independant variables are equal or not
### It is decided based on P value and it is hypothesis based test.
## if p value is less than 0.05 then we reject the null hypothesis saying that means are not equal
### if p value is more than 0.054 we say accept the null hypothesis saying that means are equal.

### lets store all categorical variables in one variable.
factor_index=sapply(train_cab, is.factor)
factor=train_cab[,factor_index]

for(i in 1:6){
  print(names(factor)[i])
  cnt= train_cab[,1]
  anova=aov(cnt~factor[,i], data=factor)
  print(summary(anova))
}

### we can see weekday and date variable p value is more than 0.05 
# so we will remove weekday variable from train and test data

train_cab=subset(train_cab, select=-c(weekday))
head(train_cab)

test_cab=subset(test_cab, select=-c(weekday))

# lets convert facor variable to numeric for VIF calculation
factor_data= colnames(train_cab[,c('date','month', 'year', 'hour', 'passenger_count' )])

for (i in factor_data){
  train_cab[,i]=as.numeric(train_cab[,i])
}


#### Multicollinearity test of all independant variables

### it gives whether there is dependancy between each independant variable with other is present or not
#### lets load the library
library(usdm)
# we will check the Multicollinearity by VIF that is variance inflation factor
## if VIF is between 1 to 5 there is no Multicollinearity
# if VIF is more than 5 there is high Multicollinearity
# lets calculate VIF for all independant variables
vif(train_cab[,-1])

# lets print the results of VIF 
vifcor(train_cab[,-1], th = 0.9)

### we can see VIF of all variables is between 1 to 5 
## SO we can say that there is now multicollinearity between independant variables.



### FEATURE SCALING
### lets plot the histogram to see the dtaa distribution of distance and fare_amount variable 
hist(train_cab$fare_amount)
hist(train_cab$distance)

## We know that Normalizatoin and Standardization are the two methods for doing scaling
## But our data has outliers and normalization is sensititve to outliers 
## so we will do log transform of fare_amount and distance variable.
train_cab$fare_amount=log1p(train_cab$fare_amount)
test_cab$distance=log1p(test_cab$distance)
train_cab$distance=log1p(train_cab$distance)

## Lets plot histograms again to see the data distribution.
hist(train_cab$distance)
hist(train_cab$fare_amount)

# lets create copy of scalled data.
train_cab_scaled= train_cab

#### Lets split train data into Train and Test data to build ML models on it 
set.seed(1200)
Train.index = sample(1:nrow(train_cab), 0.85 * nrow(train_cab))
Train = train_cab[ Train.index,]
Test  = train_cab[-Train.index,]

# lets remove the unwanted data from R environment
rmExcept(c('train_cab', 'test_cab', 'Train', 'Test', 'train_cab_scaled', 'train_cab_missingdata'))

##### ERROR METRICS
### we have different error metrics to analize the ML model
## We will use rsquare,MAPE,Accuracy, rmse and mae for our models
# rsquare- It will tell us how much variation of target variable is explained by independant variables
# MAPE- it is a percentage error between real and predicted values of target variable
# Accuracy- It is the accuracy of model in percentage,  which is (100- MAPE)
# rmse- It is Standard deviation of residuals i.e. prediction errors. It should be low and between 0.2 to 0.5 is good
# mae- It is comparision of predited versus observed value of target variable. It gives the error and it should be less.

## So we will calculate all these error metrics for each model
## We require rsquare value high and MAPE value as much less as possible
## SO we will select the model whose performance is like high rsquarew and low MAPE value.


#### Now lets build the regression models on Train and Test data as our target variable is numeric variable.

########## LINEAR REGRESSION MODEL

# Lets build linear regression model
lr_model=lm(fare_amount~., data=Train)

#summary of regression model
summary(lr_model)

# prediction on test data
predictions_lr=predict(lr_model, Test[,2:7])
Predictions_LR_train = predict(lr_model,Train)

# Function to calculate r square error metric
rsquare=function(y,y1){
  cor(y,y1)^2
}

# Function to calculate error metric mape which is Mean Absolute Percentage Error. 

mape= function(y, yhat){
  mean(abs((y-yhat)/y))*100
}

# calculate error metrics to evaluate the model for train data
rsquare(Train[,1],Predictions_LR_train)
mape(Train[,1],Predictions_LR_train)
rmse(Train[,1],Predictions_LR_train)
mae(Train[,1],Predictions_LR_train)

##Predictive performance of model using error metrics 
# rsquare= 0.7569
# mape(error rate)=7.53
# Accuracy =92.47
# rmse=0.2694
# mae=0.1729

# calculate error metrics to evaluate the model for test data
rsquare(Test[,1],predictions_lr)
mape(Test[,1],predictions_lr)
rmse(Test[,1],predictions_lr)
mae(Test[,1],predictions_lr)

##Predictive performance of model using error metrics 
# rsquare= 0.7493
# mape(error rate)=7.85
# Accuracy =92.15
# rmse=0.2730
# mae= 0.180

## We can see the model is doing well on train data as compare to test data.
## But there is no much difference between error metrics of train and test data
## So we will calculate error metrics on test data only for further models

######### DECISION TREE MODEL OF REGRESSION

#Decision tree for regression
fit=rpart(fare_amount~., data= Train, method="anova")

# predict for new test data
predictions_dt= predict(fit, Test[,-1])

# compare real and predicted values of target variable

comparision_dt=data.frame("Real"=Test[,1], "Predicted"= predictions_dt)

#lets print error metrics
rsquare(Test[,1],predictions_dt)
mape(Test[,1], predictions_dt)
rmse(Test[,1],predictions_dt)
mae(Test[,1],predictions_dt)


##Predictive performance of model using error metrics  on test data
# rsquare= 0.7392
# mape(error rate)=8.58
# Accuracy =91.42
# rmse=0.2784
# mae= 0.1963

#### lets do parameter tunning of decision tree model using random search cv
## lets first set the model with default parameters 
control = trainControl(method="repeatedcv", number=5, repeats=1, search='random')
maxdepth = c(1:30)
params = expand.grid(.maxdepth=maxdepth)

# Lets build a model using above parameters on train data 
DT_model = caret::train(fare_amount~., data=Train, method="rpart2",trControl=control,tuneGrid= params)
print(DT_model)

#lets look best parameters
best_parameters = DT_model$bestTune
print(best_parameters)
# maxdepth= 7

# Now lets build Decision tree model again using best parameters that we got above
DT_tunned = rpart(fare_amount~.,Train, method = 'anova',maxdepth=7)
print(DT_tunned)

#lets predict for test data
predictions_DT_tunned = predict(DT_tunned,Test)

# calculate error metrics to evaluate the model
rsquare(Test[,1],predictions_DT_tunned)
mape(Test[,1], predictions_DT_tunned)
rmse(Test[,1],predictions_DT_tunned)
mae(Test[,1],predictions_DT_tunned)

##Predictive performance of Decision tree tunned model using error metrics on test data 
# rsquare= 0.7392
# mape(error rate)=8.58
# Accuracy =91.42
# rmse=0.2784
# mae= 0.1963

####### RANDOM FOREST MODEL OF REGRESSION

# creating the model 
RF_model= randomForest(fare_amount~., Train, importance=TRUE, ntree= 80)
RF_model
# convert rf object to trees
treeList= RF2List(RF_model)         

# extract the rules from the model
rules=extractRules(treeList, Train[,-1])
rules[1:2,]

# make rule readable

readablerules=presentRules(rules,colnames(Train))

# get rule metrics 

ruleMetric=getRuleMetric(rules, Train[,-1], Train$fare_amount)

# predict the test data using RF model
predictions_rf=predict(RF_model,Test[,-1])

# Calculate error metrics to evaluate the performance of model
rsquare(Test[,1],predictions_rf)
mape(Test[,1],predictions_rf)
rmse(Test[,1],predictions_rf)
mae(Test[,1],predictions_rf)

##Predictive performance of model using error metrics 
# rsquare= 0.7872
# mape(error rate)= 7.55
# Accuracy =92.45
# rmse=0.2521
# mae= 0.1716

#### Now lets improve the accuracy using XGBoost ensemble model
## lets store our train and test data in matrix form
train_data_matrix = as.matrix(sapply(Train[-1],as.numeric))
test_data_data_matrix = as.matrix(sapply(Test[-1],as.numeric))

# lets build the XGBoost model on train data
xgboost_model = xgboost(data = train_data_matrix,label = Train$fare_amount,nrounds = 15,verbose = FALSE)

# lets see the summary of our model
summary(xgboost_model)

# lets apply the model on test data to predict.
xgb_predictions = predict(xgboost_model,test_data_data_matrix)

# Lets create data frame of real and predicted values of target variable by xgboost model
comparison_xgb=data.frame("Real"=Test[,1], "Predicted"= xgb_predictions)

## lets see the relation between real and predicted values by plot. 
plot(Test$fare_amount, xgb_predictions, xlab= 'Real_values', ylab= 'predicted_values', main='xgb_model fare_amount values')

## we can see there is linear relationship between real and predicted values

# plotting the line graph for real and predicted values of target variable
plot(Test$fare_amount, type="l", lty=4, col="violet")
lines(xgb_predictions, col="red")

# lets plot variable importance plot w.r.t fare_amount variable.
# lets store imporantat variables gain and frequency in 'imp' object
imp= xgb.importance(feature_names = colnames(Test[,2:7]), model =xgboost_model )
imp
# lets plot the variable importance plot.
xgb.plot.importance(importance_matrix = imp[1:7], xlab= 'Gain', ylab= "Variables")

# Calculate error metrics to evaluate the performance of model
rsquare(Test[,1],xgb_predictions)
mape(Test[,1],xgb_predictions)
rmse(Test[,1],xgb_predictions)
mae(Test[,1],xgb_predictions)


##Predictive performance of model using error metrics 
# rsquare= 0.8010
# mape(error rate)= 7.1479
# Accuracy =92.85
# rmse=0.2436
# mae= 0.1632

### CONCLUSIUONS 

# We have built (linear regression, Decision tree regression,
#          Decision tree tunned model, Random Forest regression and XGBoost Ensemble Regression model.)
# We have used rsquare, MAPE, rmse and mae error metrics to evaluate the performance of each model 
# ALso we have calculated accuracy of each model
# Now From all the models we have built we seen that Random forest and XGBoost models are good

## But XGBoost model is having rquare value above '80' and MAPE '7.14'
## we got the accuracy '92.86'% which is best amongst all the models we built
## So on the basis of these error metrics we will finalize 'XGBoost model as final model


## We will predict fare_amount for given test data using XGBoost model
test_cab_new = as.matrix(sapply(test_cab,as.numeric))


# lets predict fare_amount for given test data with problem statement.
xgb_predict_test = predict(xgboost_model,test_cab_new)


## lets store predicted fare amount in given test data 
test_cab$predicted_fare= with(test_cab,xgb_predict_test)
head(test_cab)

## Now lets save the given test data with added predicted fare variable  as "test_predicted_R" in our working dir. 
write.csv(test_cab,"test_predicted_R.csv",row.names = FALSE)




 



