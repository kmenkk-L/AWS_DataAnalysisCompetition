
#AWS_Data Analysis Competition
#Group 4
########################################################## Data Preparation  #############################################################################

# Import original csv data file to R
data <- read.csv("/Users/qiwenliang/Desktop/AWS Programming/data.csv")
data <- data.frame(data)

### Deal with Missing value

# Examine NA value in the dataset
data[!complete.cases(data),] 
#we can see all NA value are appears in variable TotalCharges
which(is.na(data$TotalCharges)) 
# As missing value is MCAR - there’s no relationship between the missing data points and other missing values in the data set 
# Percentage of missing value over the data set = 11/7043 = 0.156%. Thus, we will delete the rows with NA value. 
# MCAR: the propensity for a data point to be missing is completely random.

data <- na.omit(data)

### Transform data
# convert all the "Yes" from data set into "1", and "No" into "0"
data$gender = ifelse(data$gender == "Yes", 1, 0)
data$Partner= ifelse(data$Partner == "Yes", 1, 0)
data$Dependents =ifelse(data$Dependents == "Yes", 1, 0)
data$PhoneService =ifelse(data$PhoneService == "Yes", 1, 0)
data$MultipleLines =ifelse(data$MultipleLines == "Yes", 1, 0)
data$OnlineSecurity =ifelse(data$OnlineSecurity == "Yes", 1, 0)
data$OnlineBackup =ifelse(data$OnlineBackup == "Yes", 1, 0)
data$DeviceProtection= ifelse(data$DeviceProtection == "Yes", 1, 0)
data$TechSupport =ifelse(data$TechSupport == "Yes", 1, 0)
data$StreamingTV =ifelse(data$StreamingTV == "Yes", 1, 0)
data$StreamingMovies =ifelse(data$StreamingMovies == "Yes", 1, 0)
data$PaperlessBilling =ifelse(data$PaperlessBilling == "Yes", 1, 0)
data$Churn =ifelse(data$Churn == "Yes", 1, 0)


# convert each level in factor variables into dummy variables with "1" and "0"
# If customers signed up for the particular service, would equal to "1", else, equal to "0"
model.matrix(~ InternetService -1 , data)
model.matrix(~ Contract -1 , data)
model.matrix(~ PaymentMethod -1 , data)

# Merge dummy variables to original data set
data <- data.frame(data[,!colnames(data)%in%"InternetService"],
                         model.matrix(~InternetService -1,data))
data <- data.frame(data[,!colnames(data)%in%"Contract"],
                   model.matrix(~Contract -1,data))
data <- data.frame(data[,!colnames(data)%in%"PaymentMethod"],
                   model.matrix(~PaymentMethod -1,data))


# Delete columns: customer ID, InternetService, PaymentMethod, Contract
data <- data[-c(1,21,28,24)]

# Rename variables in the data set
library(dplyr) 
data <- data %>%
  rename(
    Bank.transfer.automatic = PaymentMethodBank.transfer..automatic.,
    Credit.card.automatic = PaymentMethodCredit.card..automatic.,
    Electronic.check = PaymentMethodElectronic.check,
    Month.to.month = ContractMonth.to.month,
    One.year = ContractOne.year,
    DSL = InternetServiceDSL,
    Fiber.optic = InternetServiceFiber.optic,
  )


# Convert all variables into categorical variables

library(dplyr) # for data manipulation (dplyr)
cat.names = data %>% select_if(is.numeric) %>% colnames()
data[,cat.names] = data.frame(apply(data[cat.names], 2, as.factor))

#convert variables tenure, totalcharges,monthlycharges into numerical variables
data$tenure <- as.numeric(data$tenure)
data$TotalCharges <- as.numeric(data$TotalCharges)
data$MonthlyCharges <- as.numeric(data$MonthlyCharges)
str(data) #check data set


# The final data contains 7032 obs and 24 columns. 
# The dependent binary variable is of factor type contains Churn(customers who left within last month) as "1", else, as "0"

############################################################# Data Sampling ################################################################################

prop.table(table(data$Churn)) 
# Number of customers who did not leave within last month is about 73.42%, while Churn=Yes is 26.58%
# We need to oversampling the rare event, which is Churn = "1"


######## Data Partition, create training data set and test data set #########

# We choose training data set : test data set = 70% : 30%
# Use the training data set to build and validate the model
# Treat the test data set as the unseen new data

set.seed(123)
ind <-sample(2,nrow(data),replace=TRUE,prob=c(0.7,0.3))
train <- data[ind==1,]
test <- data[ind==2,]

# Examine response variables in training data set
table(train$Churn) 
# Only 3643 obs belongs to class "0", and 1300 obs belongs to class "1"

prop.table(table(train$Churn)) 
# Proportion of response variables for class "0" vs class "1" = 73.7% vs 26.3%


## Apply RandomForest method into training data set to select classifier

library(randomForest)
rftrain <- randomForest(Churn~.,data=train)

## Predictive model evaluation with test data
library(caret)
library(e1071)
# Using rftrain model we created, to test the test data
confusionMatrix(predict(rftrain,test),test$Churn,positive = "1") 
# indicate positive is 1, which means customers who left within last month is positive

### Information from Confusion Matrix applied in training data set & explanation why we need to do oversampling

# Reference is the actual value; Prediction is predicted by the model
# Correct prediction: 1383 customers did not leave(from data) and model give the same prediction;
# Correct prediction: 257 customers left within last month (actual value), and model give same prediction; 
# Incorrect prediction: 137 customers did not leave(from data) and model give the wrong prediction;
# Incorrect prediction: 257 customers left within last month (actual value), and model give wrong prediction; 
# Overall Accuracy = 78.51%, 95% Confidence Interval indicate the accuracy value is likely to lie between 76.68% and 80.25%
# No information rate: largest proportion of the observed class (class "0") = (1377+143)/2089(test data)
# means if we do not develop any model and simply classified every customers who did not Churn,  will be around 72.76%
# 72.76% < model accuracy 78.51%, it seems the model prediction is better, but let's check sensitivity and specificity first
# Sensitivity indicate how often we are able to correctly predict class "1" = 45.17%
# Specificity indicate how often we are able to correctly predict class "0" = 90.99%
# The reason of big difference between sensitivity and specificity is that the prediction model is dominated by class "0", excellent in predicting class "0" but not for predicting class "1"
# Therefore, the overall predicting accuracy of model 78.51% is misleading, which can be proved by the huge fluctuation between sensitivity and specificity
# In order to deal with the data imbalance problem, we need to do oversampling 


########## Use oversampling method to deal with low sensitivity and data imbalance problem  #########

library(ROSE) # randomly over sampling examples
over <- ovun.sample(Churn ~., data=train, method="over",N = 7286)$data 
# N = 7286 = 3643*2 (numbers of response variables in class"0" in training data set, use code"table(train$Churn)" to examine), make class "1" & class "0" become equal sample in training data set
# Sample from class "1" repeat randomly, until equal sample
table(over$Churn) 

# Run Randomforest again in the oversampled data
library(randomForest)
rfover <- randomForest(Churn~.,data=over)
importance (rfover)

confusionMatrix(predict(rfover,test),test$Churn,positive = "1")
# Applied test data set to predict Random forest model after oversampling
# Overall model accuracy drop from 78.51% to 76.3%, sensitivity:specificity = 66.26% : 80.07%
# Improve sensitivity significant by doing oversampling, even though overall accuracy decrease, the model still very good


######################################################## Model Building: Logistic Regression ########################################################################

# Recall that we can view the importance of each variables from Random Forest 
# Mean Decrease Gini is a measure of variable importance for estimating a target variable across all of the trees that make up the forest.
# A higher Mean Decrease in Gini indicates higher variable importance.
# Thus, we choose six variables with highest Mean Decrease Gini(>100): tenure, Fiber.optic, Month.to.month, MothlyCharges, TotalCharges, Electronic.check
# As our response variable takes on the value "0" and "1" only and we think X-variables are related to the response variable.
# Thus, we use logistic regression to build the model

m1<- glm(Churn ~ TotalCharges + MonthlyCharges + tenure + Fiber.optic + Month.to.month + Electronic.check, family = binomial(), data=data)
summary(m1)
library(car)
vif(m1) 
# From summary of model m1, variable MonthlyCharges is insignificant with p-value = 0.458 > 0.05
# Thus, we drop this insignificant variable and build model m2

m2<- glm(Churn ~ TotalCharges +  tenure + Fiber.optic + Month.to.month + Electronic.check, family = binomial(), data=data)
summary(m2)
vif(m2)  
# From summary of model m2, all variables are highly significant 
# VIF(Variance Inflation Factor) of all the variables are similar at valur 1, which means the predictor is not correlated with other variables
# There is no multicollinearity among the variables


# Create interaction term as new variable 
cat.names = data %>% select_if(is.factor) %>% colnames()
data[,cat.names] = data.frame(apply(data[cat.names], 2, as.numeric))

TotChaTen <- data$TotalCharges * data$tenure
TotChaFib <- data$TotalCharges * data$Fiber.optic
TotChaMtm <- data$TotalCharges * data$Month.to.month
TotChaEle <- data$TotalCharges * data$Electronic.check
TenFib <- data$tenure * data$Fiber.optic
TenMtm <- data$tenure * data$Month.to.month
TenEle <- data$tenure * data$Electronic.check
FibMtm <- data$Fiber.optic * data$Month.to.month
FibEle <- data$Fiber.optic * data$Electronic.check

# Put all the interaction term into model m2
f1 <- glm(formula = Churn ~ TotalCharges + tenure + Fiber.optic + Month.to.month + Electronic.check + 
            TotChaTen + TotChaFib + TotChaMtm + TotChaEle +
            TenFib + TenMtm + TenEle +
            FibMtm + FibEle, family = binomial(), data = data)
summary(f1)
# The interaction term of TotalCharges and Electronic.check has highest p-value = 0.7136, which is highly insignificant
# Drop variable TotChaEle from the model


f2 <- glm(formula = Churn ~ TotalCharges + tenure + Fiber.optic + Month.to.month + Electronic.check + 
            TotChaTen + TotChaFib + TotChaMtm + 
            TenFib + TenMtm + TenEle +
            FibMtm + FibEle, family = binomial(), data = data)
summary(f2)
# The interaction term of Tenure and Electronic.check has highest p-value = 0.7369, which is highly insignificant
# Drop variable TenEle from the model


f3 <- glm(formula = Churn ~ TotalCharges + tenure + Fiber.optic + Month.to.month + Electronic.check + 
            TotChaTen + TotChaFib + TotChaMtm + 
            TenFib + TenMtm + 
            FibMtm + FibEle, family = binomial(), data = data)
summary(f3)
# The interaction term of TotalCharges and Month.to.month has highest p-value = 0.5182, which is highly insignificant
# Drop variable TotChaMtm from the model


f4 <- glm(formula = Churn ~ TotalCharges + tenure + Fiber.optic + Month.to.month + Electronic.check + 
            TotChaTen + TotChaFib + 
            TenFib + TenMtm + 
            FibMtm + FibEle, family = binomial(), data = data)
summary(f4)
# The interaction term of TotalCharges and tenure has highest p-value = 0.3979, which is highly insignificant
# Drop variable TotChaTen from the model


f5 <- glm(formula = Churn ~ TotalCharges + tenure + Fiber.optic + Month.to.month + Electronic.check + 
            TotChaFib +  
            TenFib + TenMtm + 
            FibMtm + FibEle, family = binomial(), data = data)
summary(f5)
# The interaction term of Fiber.optic and Electronic.check has highest p-value = 0.3553, which is highly insignificant
# Drop variable FibEle from the model


f6 <- glm(formula = Churn ~ TotalCharges + tenure + Fiber.optic + Month.to.month + Electronic.check + 
            TotChaFib +  
            TenFib + TenMtm + 
            FibMtm , family = binomial(), data = data)
summary(f6)
# The interaction term of tenure and Month.to.month has highest p-value = 0.2694, which is highly insignificant
# Drop variable TenMtm from the model


f7 <- glm(formula = Churn ~ TotalCharges + tenure + Fiber.optic + Month.to.month + Electronic.check + 
            TotChaFib +  
            TenFib + 
            FibMtm , family = binomial(), data = data)
summary(f7)
# All interaction term are significant, but variable TotalCharges is 
# Interaction term TotChaFib is a cross-over interaction term


### Final Model
final <- f7
summary(final)
# Variables with positive coefficient means that higher values will indicate high prob. to Churn


####################################################  Evaluation: Model Accuracy ##################################################################################

### Confusion Matrix

# Since the prediction of a logistic regression model is a probability, in order to use it as a classifier, we’ll have to choose a cutoff value（threshold value）
# The best threshold point to be used in GLM models is the point which maximize the specificity and the sensitivity

predict.train <- predict(final,type="response")  #Prediction we made from the regression model
summary(predict.train)
tapply(predict.train, data$Churn, mean) 

# Compute the avg prediction for each outcomes
# We find that for all of the correct prediction of Churn = "yes", we predict an average probability of about 0.45. 
# And for all of the correct prediction Churn = "No", we predict an average probability of about 0.20.
# This is good because it looks like we’re predicting a higher probability of the actual value of Churn = "Yes" cases.

# Determine threshold value t
# If the probability of customers who left within last month is greater than this threshold value t, we predict as Churn = "Yes". 
# But if the probability of customers who didn't leave within last month is less than the threshold value t, then we predict as Churn = "No"


library(ROCR)
ROCRpred <- prediction(predict.train,data$Churn)
# The first argument is the predictions we made with our regression model, which we called predict.train.
# The second argument is the true outcomes of our data points


### Performance function

# Defines what we’d like to plot on the x and y-axes of our ROC curve.
ROCRperf = performance(ROCRpred, "tpr", "fpr")  
# tpr=true positive rate, fpr=false positive rate

plot(ROCRperf, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))
abline(a=0,b=1, col="#8AB63F")
# The higher the threshold, the higher the specificity and the lower the sensitivity. 
# The lower the threshold, the higher the sensitivity and lower the specificity.
# We choose threshold value = 0.4 as it maximizes the true positive rate(>60%) while keeping the false positive rate really low(<20%)
# The green line represents a completely uninformative test, corresponding to AUC=0.5
# ROC curve pulled close to the upper left corner indicates a better performing test. 

predict.test <- predict (final,type="response", newdata=data)
table(data$Churn,predict.test >= 0.4)
Accuracy <- (4226+1163)/7032
Accuracy # = 76.64%
# There are total 7032obs in the original data set
# Out of which 5163 of them are actually didn't leave the company within last month
# And 1869 of them are actually left within last month

### Baseline model accuracy
table(data$Churn)
BLaccuracy <- 5163/7032 
BLaccuracy
# Baseline model has an accuracy 73.42%

# The model can accurately identify customers who left last month accuracy being equal to 76.64% which is greater than our baseline model(73.42%)

### AUC value
# AUC stand for Area Under the ROC Curve, which measure the entire two-dimensional area underneath the entire ROC curve from (0,0) to (1,1).
# AUC is classification-threshold-invariant, which measures the quality of the model's predictions irrespective of what classification threshold is chosen.
# Summarize the ROC curve just by taking the area between the curve and the x-axis

auc.perf = performance(ROCRpred,measure = "auc")
auc.perf@y.values # AUC = 0.819
# When AUC lies between 0.8 and 0.9, the model is considered excellent prediction ability.

