#Project Name Cab Fare Prediction 

#Problem Statement -

#You are a cab rental start-up company. You have successfully run the pilot project and now want to launch your cab service across the country. You have collected the historical data from your pilot project and now have a requirement to apply analytics forfare prediction. You need to design a system that predicts the fare amount for a cab ride in the city.

# Instruction to run code
# You need to set the working directory path in the following code
# And the working directory that you have set must have train_cab and test data sets


# loading libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from geopy.distance import geodesic
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge,Lasso
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics


# Lets Set working directory
os.chdir("F:\\data Scientist\\Project 2 Edwisor")
os.getcwd()

#For the given project we have provided the train and test data sets seperately. so now we will load the data sets and observe it. 


# load given train data 
train_cab= pd.read_csv('train_cab.csv', na_values={"pickup_datetime":"43"})
 

#EXPLORATORY DATA ANALYSIS


#In this section we will observe/visualize the data and do some data cleaning operations on both train and test data sets 

# Lets see first five rows of train data
train_cab.head() 

# lets see the shape of train data
train_cab.shape

# Statistical measures of data
train_cab.describe()

#-We can see the train_cab data statistical measures in the above table.
#-we know that latitude ranges from -90 to +90 and longitude ranges from -180 to +180.
#-But in above train_cab data table pickup_latitude variable maximum value is above 90 so we need to remove those values. All other values of latitide and longitude are within range.
#-Also passenger_count variable contains minimum value as'0.0' and maximum value as '5345.0'and its data type is float. So we also need to solve this issue.

# data types of train data variables
train_cab.dtypes

#-We can see that fare_amount is in 'object' data type so it need to be converted to 'numeric'also  pickup_datetime vatiable is a 'timestamp varibale' so it is also need to be converted to 'date_time'

#Convert fare_amount varible from "object" data type to "numeric"
train_cab["fare_amount"] = pd.to_numeric(train_cab["fare_amount"],errors = "coerce")

# Now convert pickup_datetime variable to date_time data type in train data
train_cab['pickup_datetime']=pd.to_datetime(train_cab['pickup_datetime'],errors='coerce')

# load given 
test_cab=pd.read_csv('test.csv')
# To see first five rows of test data
test_cab.head()

# lets see the shape of test data
test_cab.shape

# Statistical measures of data
test_cab.describe()

# data types of test data variables
test_cab.dtypes

##In test data also we need to convert the pickup_datetime varibale to 'date_time' as it is related to time and date.

# Now convert pickup_datetime variable to date_time data type in train data
test_cab['pickup_datetime']=pd.to_datetime(test_cab['pickup_datetime'],errors='coerce')

#Now we will see some visualizations to understand the train_cab data in better way
#we will plot histogram of fare_amount to see the distribution of data

train_cab['passenger_count'].value_counts()

train_cab['fare_amount'].sort_values(ascending=False)

#-By domain knowledge of this project we can say that fare_amount should not be '0' or negative. 

#-we can see above the values beyond 453.00 are very high which is not practicaly possible. so we need to remove the observations having values more than '453'

# lets check that is there any values below '1' in fare_amount and 'negative'
print('values below 1=''={}'.format(sum(train_cab['fare_amount']<1)))
print('values above 453=''={}'.format(sum(train_cab['fare_amount']>453)))

train_cab[train_cab['fare_amount']<1]


# lets drop all '7' observations which are below  '1' and above '453'
train_cab = train_cab.drop(train_cab[train_cab['fare_amount']<1].index, axis=0)
train_cab = train_cab.drop(train_cab[train_cab['fare_amount']>453].index, axis=0)

# lets plot box plot of passenger_count variable
plt.figure(figsize=(20,5)) 
plt.xlim(0,100)
sns.boxplot(x=train_cab['passenger_count'],data=train_cab,orient='h')
plt.title('Boxplot of passenger_count')

#-We will assume maximum 6 passengers could travel in one cab
#-From the above plot we can see that passenger_count in train_cab data has some values more than '6' which is not possible practically.

train_cab["passenger_count"].describe()

#we can see above passenger_count is having minimum value as '0' and maximum as '5345' which is not practical. There should be at least one passenger and maximum six. so we will remove the observations who have passenger_count more than '6' and less than '1'.

# Remove passenger_count above '6' and below '1'  
train_cab = train_cab.drop(train_cab[train_cab['passenger_count']>6].index, axis=0)
train_cab = train_cab.drop(train_cab[train_cab['passenger_count']<1].index, axis=0)

# recheck
sum(train_cab['passenger_count']<1)

train_cab["passenger_count"].unique()

#-As we can see the unique values of passenger_count varibale. 
#-It contains 'NA' values that is missing values so we will deal with it in missing value analysis.
#-It is having 1.3 as a unique value, But it is not practical as passenger can not be 1.3 
#-so we will also remove the obersvations with value 1.3.     

# Removing the observations in passenger_count with 1.3 value.
train_cab = train_cab.drop(train_cab[train_cab['passenger_count']==1.3].index, axis=0)

#-Now passenger_count will have unique values 1.0,2.0,3.0,4.0,5.0,6.0 and 'NA'
#-It contains float values.
#-We will convert the variable in proper data type and category after the missing value analyis 

#check the above values for test_cab data
test_cab["passenger_count"].unique()

#-we can see that test_cab data does not contain outliers like train data.So we have completed the processing of passenger_count varibale of both train and test data.

#-Now in next step we will look into longitude and latitude variables
#-We already observed in train_cab data that pickup_latitude is above 90. 
#-so we will reomove those observations who are outside the limit because we can not impute them.

train_cab[train_cab['pickup_latitude']>90]

train_cab = train_cab.drop((train_cab[train_cab['pickup_latitude']>90]).index, axis=0)

# Now let us check the all 4 latitude and longitude variables for presence of '0' value
location_var= ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']
for i in location_var:
    print('Zero value count in',i,'={}'.format(sum(train_cab[i]==0)))


#-We can see all the four varibales contains '0' values so we will remove all those observations which contains '0' value from that respective varibales.

# Remove observations with '0' value
for i in location_var:
    train_cab = train_cab.drop((train_cab[train_cab[i]==0]).index, axis=0)

#Missing value analysis-


#In this step we will find the variables with missing values. If missing values are present then we will impute them. 

# create dataframe with missing values

missing_val= pd.DataFrame(train_cab.isnull().sum())
missing_val

# reset the inde of rows
missing_val = missing_val.reset_index()
missing_val

#we can see we have missing values in three variables fare_amount,pickup_datetime,passenger_count so we need to impute them.

# lets rename the variables
missing_val=missing_val.rename(columns= {"index":"Variables", 0:"Missing_Percentage"})
missing_val

# calculate the percentage of missing values
missing_val["Missing_Percentage"]= (missing_val["Missing_Percentage"]/len(train_cab))*100
missing_val

# lets sort the Missing_percentage in descending order
missing_val=missing_val.sort_values("Missing_Percentage", ascending=False).reset_index(drop=True)
missing_val

# Procedure to Impute missing values is as follows
# 1) Select any random observation from data having its value and equal it with "NA"
# 2) Now impute that value by using mean, mode, median 
# 3) Compare the value imputed by above methods with actual value.
# 4) select the method which will give more accurate result
# 5) Now choose that method and find all missing values in that variable.
# 6) Repeat above steps t0 impute all missing values.


# Now we will calculate missing values for fare_amount variable.
# we will use mean, median  
# Mode is not useful as it is of numeric data type value

# lets selecet any random observation
train_cab['fare_amount'].loc[7230]


# fare_amount location 7230 value
# Actual value-6.9
# Mean-11.36
# Median-8.5

# Now make this value equal to "NA"
train_cab['fare_amount'].loc[7230]=np.nan
train_cab['fare_amount'].loc[7230]


# lets impute with mean
train_cab['fare_amount']=train_cab['fare_amount'].fillna(train_cab['fare_amount'].mean())
train_cab['fare_amount'].loc[7230]

# now again make that value as equal to 'NA'
train_cab['fare_amount'].loc[7230]=np.nan
train_cab['fare_amount'].loc[7230]

# lets impute with median
train_cab['fare_amount']=train_cab['fare_amount'].fillna(train_cab['fare_amount'].median())
train_cab['fare_amount'].loc[7230]

#we can see median is giving better result than mean so we will calculate all missing values using median

train_cab['fare_amount']= train_cab['fare_amount'].fillna(train_cab['fare_amount'].median())

# Now lets calculate missing values for passenger count variable
# we will use mode method for imputing missing values

# lets selecet any random observation
train_cab['passenger_count'].loc[2]

# passenger_count location '2' 
# Actual value-2.0
# mode= 1.0
# mean= 1.64
# median= 1.64

# Now make this value equal to "NA"
train_cab['passenger_count'].loc[2]=np.nan
train_cab['passenger_count'].loc[2]

# lets impute with mode
train_cab['passenger_count']= train_cab['passenger_count'].fillna(train_cab['passenger_count'].mode()[0])

train_cab['passenger_count'].loc[2]

# now again make that value as equal to 'NA'
train_cab['passenger_count'].loc[2]=np.nan

# lets impute with mean
train_cab['passenger_count']=train_cab['passenger_count'].fillna(train_cab['passenger_count'].mean())
train_cab['passenger_count'].loc[2]

# now again make that value as equal to 'NA'
train_cab['passenger_count'].loc[2]=np.nan

# lets impute with median
train_cab['passenger_count']=train_cab['passenger_count'].fillna(train_cab['passenger_count'].median())
train_cab['passenger_count'].loc[2]

# we can see mean and median are giving good results than mode. so we will freeze mean method.
# Also the value is of float data type but passenger_count can not be 1.64 
# So we will convert passenger_count to int data type and make round off of its values.

# lets impute all missing value with mean 
train_cab['passenger_count']=train_cab['passenger_count'].fillna(train_cab['passenger_count'].mean())

# let us check the details of passenger_count variable
train_cab['passenger_count'].unique()


# lets round off the values 
train_cab['passenger_count'].round()

train_cab['passenger_count'].unique()

# we know that there is missing value in pickup_datetime variable.
# which is only one missing value, so we will remove that observation instead of imputing it 
train_cab = train_cab.drop(train_cab[train_cab['pickup_datetime'].isnull()].index, axis=0)

missing_val= pd.DataFrame(train_cab.isnull().sum())
missing_val

test_cab.isnull().sum()

#-SO WE HAVE IMPUTED ALL THE MISSING VALUES. OUR BOTH DATA ARE FREE FROM MISSING VALUES. SO NOW WE WILL PROCEED FURTHER.

#-Now next step after missing value analysis is actually outlier analysis.
#-But before going ot outlier analysis we will do some FEATURE ENGINEERING

###FEATURE ENGINEERING

#-If we see our variables we have date_time variable which contains information of day,month,year and time at that point. But these all are enclosed in one format there so we can not use that as it is and our ML model didnt regognise this variable so we will split that variable and create new variables like date,month,year, weekday, hour and minute.

#-Similarly if we see other four variables pickup_longitude,pickup_latitude,dropoff_longitude, dropoff_latitude these are nothing but the coordinates of passenger where from he is picked up and dropped by cab. 
#-But we cannot use these variables also for our ML model. As we are having passenger pickup and drop coordinates we can create new variable from this that is DISTANCE variable. It will give us the distance travelled by cab during each ride.
#-Distance variable will be more important to decide Cab fare amount.
#-So we will create Distance variable


####FEATURE ENGINEERING FOR TRAIN DATA

# First we will convert pickup_datetime variable from train_cab data to date_time format
train_cab['pickup_datetime']=pd.to_datetime(train_cab['pickup_datetime'], format='%Y-%m-%d %H:%M:%S UTC')


# Now we will seperate pickup_datetime variable 
# create new variables like date, month, year, weekday, hour, minute
train_cab['date']= train_cab['pickup_datetime'].dt.day
train_cab['month']= train_cab['pickup_datetime'].dt.month
train_cab['year']= train_cab['pickup_datetime'].dt.year
train_cab['weekday']= train_cab['pickup_datetime'].dt.dayofweek
train_cab['hour']= train_cab['pickup_datetime'].dt.hour
train_cab['minute']= train_cab['pickup_datetime'].dt.minute

# lets see the top 5 observations of train_cab data to see newly extracted variables
train_cab.head()

train_cab.dtypes

# Now we will create Distance variable 
# There are two formulas to calculate distance Haversine formula and Vincenty formulla
# Haversine calculate great circle distance between latitude/longitude points assuming a spherical earth
# Vincenty calculate geodesic distance between latitude/longitude points on ellipsoidal model of earth
# But we know earth is not complete sphere so haversine formula will not give accurate results 
#  so we will use vincenty to create distance variable in km

# lets create distance variable 
train_cab['distance']=train_cab.apply(lambda y: geodesic((y['pickup_latitude'],y['pickup_longitude']), (y['dropoff_latitude'],   y['dropoff_longitude'])).km, axis=1)

# lets see the top 5 observations
train_cab.head()

# lets see the details of distance variable
train_cab['distance'].describe()

#-we can see that distance is having minimum value as 0.0 and maximum as 5434.77 but it is not practical.

# now lets arrange the distance values in descending order to see itin detail
train_cab['distance'].sort_values(ascending=False)

#-we can se values after 129.37 are increased very drasticaly so we can call it as a outlier. So we will remove those observations above 130 and having '0' values

# lets see the observaions above
sum(train_cab['distance']==0),sum(train_cab['distance']>130)

# lets remove observations with '0' and more than '130' values from train data
train_cab=train_cab.drop(train_cab[train_cab['distance']==0].index,axis=0)
train_cab=train_cab.drop(train_cab[train_cab['distance']>130].index,axis=0)

###FEATURE ENGINEERING FOR TEST DATA
#we will just repeat all the operations to test_cab data

# pickup_datetime variable from test_cab data to date_time format
test_cab['pickup_datetime']=pd.to_datetime(test_cab['pickup_datetime'], format='%Y-%m-%d %H:%M:%S UTC')


# Now we will seperate pickup_datetime variable 
# create new variables like date, month, year, weekday, hour, minute in test_cab data
test_cab['date']= test_cab['pickup_datetime'].dt.day
test_cab['month']= test_cab['pickup_datetime'].dt.month
test_cab['year']= test_cab['pickup_datetime'].dt.year
test_cab['weekday']= test_cab['pickup_datetime'].dt.dayofweek
test_cab['hour']= test_cab['pickup_datetime'].dt.hour
test_cab['minute']= test_cab['pickup_datetime'].dt.minute


test_cab.head()

# lets create distance variable in test data 
test_cab['distance']=test_cab.apply(lambda y: geodesic((y['pickup_latitude'],y['pickup_longitude']), (y['dropoff_latitude'],   y['dropoff_longitude'])).km, axis=1)

test_cab.head()

test_cab['distance'].describe()

# lets see the observaions above
sum(test_cab['distance']==0),sum(test_cab['distance']>130)

# lets remove observations with '0' and more than '130' values from test data
test_cab=test_cab.drop(test_cab[test_cab['distance']==0].index,axis=0)
test_cab=test_cab.drop(test_cab[test_cab['distance']>130].index,axis=0)


# lets see shape of train_cab and test_cab data
train_cab.shape, test_cab.shape

# lets make copy of train and test data sets
train_cab_splitted= train_cab.copy()
test_cab_splitted= test_cab.copy()

#Removing the variables used for feature engineering
#-We have applied feature engineering techniques to both train_cab and test_cab data sets.
#-Now we will remove the variables which we have used to create new variables.
#-We have used pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude so we will remove all these from both train and test data
#-Also we will remove minute varible as it is not looking so important w.r.t. to target variable. 

# lets store variables to drop in one variable

drop_variables= ['pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'minute']

# lets drop above variables from train_cab data
train_cab= train_cab.drop(drop_variables,axis=1)
test_cab= test_cab.drop(drop_variables,axis=1)



train_cab.head()

train_cab.dtypes

# we can see in data the new variables that we created are of 'float' data type 
# so we will convert all to 'int'
train_cab['passenger_count']= train_cab['passenger_count'].astype('int64')
train_cab['date']= train_cab['date'].astype('int64')
train_cab['month']= train_cab['month'].astype('int64')
train_cab['year']= train_cab['year'].astype('int64')
train_cab['weekday']= train_cab['weekday'].astype('int64')
train_cab['hour']= train_cab['hour'].astype('int64')


#we can see in data the new variables that we created are of 'flaot' data type 
#so we will convert all variables to 'int' data type
test_cab['passenger_count']= test_cab['passenger_count'].astype('int64')
test_cab['date']= test_cab['date'].astype('int64')
test_cab['month']= test_cab['month'].astype('int64')
test_cab['year']= test_cab['year'].astype('int64')
test_cab['weekday']= test_cab['weekday'].astype('int64')
test_cab['hour']= test_cab['hour'].astype('int64')


###DATA VISUALIZATIONS
#- In this section we will visualize our data to understand it in better way.

#lets see how many passengers travelling in single ride
plt.hist(train_cab['passenger_count'],color='green')

#-Single passengers are using cab service highly.

# lets see the relationship between passenger count and fare amount.
plt.figure(figsize=(10,5))
plt.scatter(x="passenger_count",y="fare_amount", data=train_cab,color='blue')
plt.xlabel('No. of passengers')
plt.ylabel('Fare_amount')
plt.show()


#-Cab fare is more for passenger_count of 1 and 2 as compared to other.

# lets see the relationship between date and fare amount.
plt.figure(figsize=(10,5))
plt.scatter(x="date",y="fare_amount", data=train_cab,color='blue')
plt.xlabel('date')
plt.ylabel('Fare_amount')
plt.show()

#-We can see there is no much effect of 'date' on 'fare_amount' and we are also considering 'weekday' variable which is related to 'date' 
#-So we can drop date variable by observing its dependancy with other varibles in chi square test

#Relationship between hour and Fare_amount
plt.figure(figsize=(10,5))
plt.scatter(x="hour",y="fare_amount", data=train_cab, color='blue')
plt.xlabel('hour')
plt.ylabel('fare_amount')
plt.show()

#-from the above plot we can say that generaly cab fare is higher after 8 pm

# lets see no of cabs w.r.t. hour in day
plt.figure(figsize=(15,7))
train_cab.groupby(train_cab["hour"])['hour'].count().plot(kind="bar")
plt.show()

#-we can see in above plot highest no of cabs are from 7 pm to 12 pm
#-Also no of cabs are less upto 5 am

# lets see the realation between fare_amount and distance variable
plt.figure(figsize=(10,5))
plt.scatter(x="distance",y="fare_amount", data=train_cab,color='blue')
plt.xlabel('distance')
plt.ylabel('fare_amount')
plt.show()

#-we can see there is a linear relatioship of distance with the fare_amount

#lest see the number of cab rides w.r.t. weekday
plt.figure(figsize=(15,7))
sns.countplot(x="weekday", data=train_cab)

#-From the above plot we can say that weekday dosent have much impact on number of cab rides 

###FEATURE SELECTION


#-In this step we will see the corelations between target variable and independant variables
#-Also we will perform different tests to check the relation between the depedant and indepedant variables 
#-Using the results of the test finally we decide which variables should be selected and which should be dropped

train_cab.dtypes

# Even if we see the data types of all variables as 'float' and 'int'
# we know that the variables fare_amount and distance are only numeric variables
# Other varables expect those are having some unique values only.
# for e.g. passenger_count has unique values 1,2,3,4,5,6.
# They are actually categorical variables 
# So we will treat these variables as categorical variables but thier data type remains 'int'

## Correlation analysis-  it is used to check the Correlation between the variables.
# This can be done mostly on numeirc variables
# lets save our numeric variables in one variable
numeric= ['fare_amount','distance']
          
train_cab_corr= train_cab.loc[:,numeric]

# setting the  height and width of plot
f, ax=plt.subplots(figsize=(13,9))

# correaltion matrix
cor_matrix=train_cab_corr.corr()

# plotting correlation plot
sns.heatmap(cor_matrix,mask=np.zeros_like(cor_matrix, dtype=np.bool), 
            cmap=sns.diverging_palette(250,12,as_cmap=True),square=True, ax=ax)

#-We can see that their is very high corelation between distance and fare_amount

####Chi-Square test-


#-We have already performed anova test and we got thre result that all our categorical variables are not dependant on each other.
#-But still we will perform the Chi-square test to validate our results.
#-This test is performed to check the dependancies between cateorical variables.
#-Null hypothesis - variables are not dependant (P<0.05-Reject)
#-Alternate hypothesis- variables are dependant (P<0.05-Accept)


# Lets perform chi square test of independance
from scipy.stats import chi2_contingency

factor_data=train_cab[['passenger_count', 'date', 'weekday', 'month', 'year', 'hour']]
for i in factor_data:
    for j in factor_data:
        if(i!=j):
            chi2,p,dof, ex=chi2_contingency(pd.crosstab(train_cab[i],train_cab[j]))
            while(p<0.05):
                print(i,j,p)
                          
                break
        

#-we can 'date' is correlated with 'weekday' and most of the variables.  
#-Also in data visualizations we observed that 'date' is not having much impact on 'fare_amount'
#-So we will remove 'date' variable


##Multicollinearity test using VIF(variance inflation factor)-

 
#-VIF detects correlation between predictor variables i.e. relationship between them.
#-If two predictor variables are correlated then we can say there is presence of Multicollinearity
#-Multicollinearity affetcs the regression models so it should not present in our variables
#-So for this we do this test using VIF
#-If VIF is between 1 to 5 then we say that there is no Multicollinearity
#-If VIF>5 then there is a multicollinearity and we need to remove it or reconsider the variables. 


# lets create dataframe of predictor variables
outcome, predictors = dmatrices('fare_amount ~ distance+passenger_count+date+weekday+month+year+hour',train_cab, return_type='dataframe')
# Lets calculate VIF for each independant variables form train_cab data
VIF = pd.DataFrame()
VIF["VIF"] = [variance_inflation_factor(predictors.values, i) for i in range(predictors.shape[1])]
VIF["Predictors"] = predictors.columns
VIF

#-We can see VIF for all the predictors is within the required range i.e. from 1-5
#-So we can say that multicollinearity is not present in our independant variables

# SO AFTER PERFORMING VARIOUS TESTS ON OUR DATA FOR FEATURE SELECTION WE HAVE FOLLOWING OBSERVATIONS
# There is no multicollinearity in our data
# We will remove 'date' variable from both train and test data.
# Select all other variables for our ML models

# lets create a copy of our data selected for Machine learning 
train_cab_selected= train_cab.copy()
test_cab_selected= test_cab.copy()

# lets drop 'date' variable from both the data sets
train_cab= train_cab.drop('date', axis=1)
test_cab= test_cab.drop('date', axis=1)


###FEATURE SCALING-


#-In this step actualy we need to do either Normalization or stadardization on numeric variables.
#-The scaling method is decided by observing histogram of variables.

# lets plot the histogram to see data distribution of fare_amount variable from train_cab data 
sns.distplot(train_cab['fare_amount'],bins='auto',color='green')
plt.title("Distribution for fare_amount variable ")
plt.ylabel("Density")
plt.show()


# lets plot histogram for distance variable from train data.
sns.distplot(train_cab['distance'],bins='auto',color='green')
plt.title("Distribution for distance variable ")
plt.ylabel("Density")
plt.show()

# lets plot the histogram to see data distribution distance variable from test_cab data 

sns.distplot(test_cab['distance'],bins='auto',color='green')
plt.title("Distribution for distance variable from test data")
plt.ylabel("Density")
plt.show()

#-We can see the histogram for all the variables is left skewed.
#-That means it contains the values which will impact more on ML model.
#-As we know that Normalization is sensitive for these data type of data so we can not use normalizatoin or standardization method of scaling
#-So we will apply log tranform to both the variables to remove the effect pf skewness.

# lets apply log tranform on numeric variables from train and test data
train_cab['fare_amount'] = np.log1p(train_cab['fare_amount'])
train_cab['distance'] = np.log1p(train_cab['distance'])
test_cab['distance'] = np.log1p(test_cab['distance'])

# lets plot the histogram to see data distribution of fare_amount variable from train_cab data after log transform
sns.distplot(train_cab['fare_amount'],bins='auto',color='green')
plt.title("Distribution for fare_amount variable after log transform ")
plt.ylabel("Density")
plt.show()

train_cab.dtypes

#-So now we have done all the data preprocessing operations on given train and test data. Our data is clean and ready to be used for ML models. So lets proceed further

# TRAIN TEST SPLITTING OF DATA. 
# first store all predictor variables in 'x' and target variable in 'y' from train_cab data
x=train_cab.drop(['fare_amount'],axis=1)  # predictors
y=train_cab['fare_amount']                 # target variable

# lets split our train_cab data into train and test data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=1)

# lets check teh shape of all the data sets we have created
x_train.shape, x_test.shape, y_train.shape, y_test.shape


##MACHINE LEARNING MODEL BUILDING

#Before procedding to model building lets decide the error metrics
#The error metrics that we will use are RMSLE, RMSE, R-square, Adjusted R-square, MAPE.
#We will use RMSLE metric because it uses log transform and gives best results for the data 
#we will also use Adjusted R-square value as it is more optimized that R-square to see the performance of model
#We will create the functions to calculate above selected error metrics for simplicity
#we will also define a function to calculate the predicions of models for train and test data

# Function to calculate RMSLE
def rmsle(yt,yp):    # yt- y_train and yp- y_predicted
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in yt]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in yp]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))
# Function to calculate other error metrics
def error_metrics(yt, yp):
    print('r square  ', metrics.r2_score(yt,yp))
    print('Adjusted r square:{}'.format(1 - (1-metrics.r2_score(yt, yp))*(len(yt)-1)/(len(yt)-x_train.shape[1]-1)))
    print('MAPE:{}'.format(np.mean(np.abs((yt - yp) / yt))*100))
    print('MSE:', metrics.mean_squared_error(yt, yp))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(yt, yp))) 

# Function to calculate and print the predictions of models for train and test data with its error metrics
def output_scores(model):
    print('###############   Error Metrics of Train data   #################')
    print()
    # Applying the model on train data to predict target variable
    y_predicted = model.predict(x_train)
    error_metrics(y_train,y_predicted)
    print('RMSLE:',rmsle(y_train,y_predicted))
    print()
    print('###############   Error Metrics of Test data  #################')
    print()
    # Applying the model on train data to predict target variable
    y_predicted = model.predict(x_test)
    error_metrics(y_test,y_predicted)
    print('RMSLE:',rmsle(y_test,y_predicted))

##MULTIPLE LINEAR REGRESSION

# lets build the model on train data
model = sm.OLS(y_train, x_train).fit()

# predict the test data
y_predict = model.predict(x_test) 
 
# lets print model summary
print_model = model.summary()
print(print_model)

# error metrics on train and test data
output_scores(model)

#Model results for test data-
#-Adjusted r square for test data-0.7384
#-MAPE is 8.13 % for test data


#LASSO REGRESSION MODEL

#Lets build the lasso model 
lasso_model = Lasso(alpha=0.00021209508879201905, normalize=False, max_iter = 500)

#Lets fit the lasso model on train data
lasso_model.fit(x_train,y_train)

#Lets print the coefficients for predictors
coefficients = lasso_model.coef_
print('coefficients of predictors:{}'.format(coefficients))

# Error metrics of train and test data
output_scores(lasso_model)


#Lasso Model results for test data-
#-Adjusted r-square for test data-0.7605
#-MAPE is 7.66 %

##RIDGE REGRESSION MODEL

# Lets build Ridge regression model
ridge_reg = Ridge(alpha=0.0005,max_iter = 500)

# Apply model on train data
ridge_reg.fit(x_train,y_train)

# print the coefficients of predictors
ridge_reg_coef = ridge_reg.coef_
print('coefficients of predictors:{}'.format(ridge_reg_coef))

 # Error metrics of train and test data
output_scores(ridge_reg)

#Model results-
#-Adjusted r-square for test data 0.7605
#-MAPE is 7.66 %
#-we can see reults of Lasso and ridge regression are same
#-Now we will build decision tree model.

####DECISION TREE REGRESSION MODEL

#-This model creates the decision tree like flow chart and gives the rules to predict the target variable.


# lets build the model and apply it on train data
fit_dt=DecisionTreeRegressor(max_depth= 2).fit(x_train, y_train)
# lets print the relative importance score for predictors
tree_features = fit_dt.feature_importances_
print('feature importance score of predictors:{}'.format(tree_features))

# Error metrics of train and test data
output_scores(fit_dt)

#Model resultsfor test data-
#-Adjusted r-square is 0.7039
#-MAPE is 9.70 %
#-#Results are not good as compared to linear regression models
#-#So we need to improve accuracy of model.

##RANDOM FOREST REGRESSION MODEL

#-This is the improved version of decision tree model it uses many trees in one model to improve the accuracy.
#-It feeds error of one tree to another tree to improve the accuracy.

# Lets build the Random forest model 
rf_model = RandomForestRegressor(n_estimators = 70, random_state=0)
# Aply model on train data
rf_model.fit(x_train, y_train)

# lets print the relative importance score for predictors
Forest_features = rf_model.feature_importances_
print('feature importance score of predictors:{}'.format(Forest_features))

# Error metrics of train and test data
output_scores(rf_model)


#Model results for test data-
#Adjusted r-square is 0.79
#-MAPE is 7.66 %
#-Results are good as compared to linear regression models and Decision tree model

###XGBoost MODEL

##-This model is nothing but the ensemble technique which provides optimized results
##-It is implementation of gradient boosted decision trees designed for speed and performance

# Lets build XGboost model and apply on train data
xgb_model = GradientBoostingRegressor(n_estimators= 70, max_depth= 2)

xgb_model.fit(x_train, y_train)


# Error metrics of train and test data
output_scores(xgb_model)

#-Results of XGBoost model are very good as compared to all other models
#-But still we will do hyper parameter tuning of Random Forest model and XGBoost model to improve the results
#-The RandomizedSearchCV function will also do cross validation of model to get best score
#-cross validation means the train data is splited into subsets for our case its is 5 
#-So from those 5 data sets one will be test and other will be train
#-The model will take all these subset data one by one as test and calculate 5 scores
#-the best score will be average of these 5 scores

###PARAMETER TUNING OF RF AND XGBOOST MODELS TO IMPROVE RESULTS

# Build random forest model for tuning
RF = RandomForestRegressor(random_state = 42)
from pprint import pprint
# Lets see the parameters of our current RF model
print('Random Forest current parameters')
pprint(RF.get_params())

##Lets create Random hyperparameter grid and apply Randomized Search CV on Random Forest Model
# lets build model again as RRF
RRF = RandomForestRegressor(random_state = 0)

# Create the random grid
random_grid_RRF = {'n_estimators': range(80,120,10), 'max_depth': range(4,12,2)}

# appply Random Search CV on model with cross validation score 5
RRF_cv = RandomizedSearchCV(RRF, random_grid_RRF, cv=5)

# lets apply model on train data
RRF_cv.fit(x_train, y_train)


# Print the tuned parameters and score
print("Tuned Random Forest Parameters: {}".format(RRF_cv.best_params_))
print("Best score {}".format(RRF_cv.best_score_))

# Build xgboost model for tuning
xgb_t = GradientBoostingRegressor(random_state = 42)
from pprint import pprint
# Lets see the parameters of our current xgboost model
print('XGBoost currrent Parameters \n')
pprint(xgb_t.get_params())

##Lets create Random hyperparameter grid and apply Randomized  Search CV on XGBoost model
# lets build model again as XBG
XGB = GradientBoostingRegressor(random_state = 0)

# Create the random grid
random_grid_XGB = {'n_estimators': range(90,120,10),'max_depth': range(1,10,1)}
                   

# Apply Random Search CV on model with cross validation score 5
XGB_cv = RandomizedSearchCV(XGB, random_grid_XGB, cv=5)

# lets apply model on train data
XGB_cv.fit(x_train, y_train)


# Print the tuned parameters and score
print("Tuned XGBoost Parameters: {}".format(XGB_cv.best_params_))
print("Best score {}".format(XGB_cv.best_score_))


###FINAL CONCLUSIONS-

#-We have built Multiple linear Regression models, Decision tree, Random forest and XGBoost models. 
#-But the results i.e. r-square, Adjuested r-square and MAPE metrics for test data of Random forest and XGBoost model was good
#-So we have done parameter tuning of these models to still see if we improve results.
#-After parameter tunning we can see that XGBoost model is giving highest best score that is 0.80
#-So we will finalize XGBoost model and apply this model with tuned parameters again on train_cab data.
#-Then finaly we will use that model to predict cab fare in our given test data that is our objective of this project.


#####FINAL REGRESSION MODEL 

# lets apply the tunned parameters and build our final XGBoost model on train_cab data.
XGB_Final = GradientBoostingRegressor( n_estimators= 110, max_depth= 3)

# Apply the model on train data.
XGB_Final.fit(x_train,y_train)

# lets create and print the important features 
XGB_Final_Features = XGB_Final.feature_importances_
print(XGB_Final_Features)

# Sorting important features in descending order
indices = np.argsort(XGB_Final_Features)[::1]

# Rearrange feature names so they match the sorted feature importances
Sorted_names = [test_cab.columns[i] for i in indices]

# create and set plot size 
fig = plt.figure(figsize=(20,10))
plt.title("Feature Importance")

# Add horizontal bars
plt.barh(range(pd.DataFrame(x_train).shape[1]),XGB_Final_Features[indices],align = 'center')
plt.yticks(range(pd.DataFrame(x_train).shape[1]), Sorted_names)
plt.savefig('Final XGBoost Model Important Features plot')
plt.show()


# "XGB_Final" train_cab and test_cab data scores
output_scores(XGB_Final)

#XGB_Final Model results on test data-
#-Adjusted r-square is 0.81  ## which is best from the all models that we have applied on data. and above 80
#-r square for is also 0.81   
#-MAPE is 7.1 % 
#-Accuracy  is 92.9
#-RMSLE: 0.069 is also less.

####PREDICTION OF CAB FARE FOR GIVEN TEST DATA

# Now we will predic cab fare for given test data using our XGB_Final model
#Apply XGB_Final model on test data
Cab_fare_test = XGB_Final.predict(test_cab)

# lets see the predicted  array
Cab_fare_test

# lets add Predicted values in our Given test data
test_cab['predicted_fare_amount'] = Cab_fare_test

test_cab.head()

# Save output of our project to csv file test_predicted in our working directory
# That is with predicted cab fare amount variable in given test data. 
test_cab.to_csv('test_predicted_python.csv')
