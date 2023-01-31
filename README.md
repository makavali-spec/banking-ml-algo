# banking-ml-algo
The data is related with direct marketing campaigns of a Portuguese banking institution, based on phone calls (Moro, Cortez, and Rita 2014).
The goal of the campaigns were to get the clients to subscribe to a term deposit. There are 20 input variables and 1 binary output variable (y) that indicates whether the
 client subscribed to a term deposit with values ‘yes’,‘no’. The input variables can be divided into four categories:

bank client data
data related to last contact of current campaign
social and economic context attributes
other attributes.

Bank client data contains variables containing information about the client. 
It includes variables indicating age, job, marital status, education, whether they have credit in default, whether they have a housing loan, whether they have a personal loan.

Data related to the last contact of the current campaign contain variables indicating the mode of communication, month of last communication, day of week when the last contact was 
made and the last call duration.

Social and economic context attributes contain variables with the quarterly employment variation rate, monthly consumer price index, monthly consumer confidence index, 
number of employees and the euribor 3 month rate.

Other attributes include number of previous contacts with the client during the current campaign, number of days since the last contact for the previous campaign, 
number of contacts performed before the current campaign for the client and the outcome of the previous marketing campaign.

The goal of the project is to classify with high accuracy whether the campaign will be successful or not given a set of input variables.


Proposed Plan:

In this project use the above data parameters to predict the outcome of the marketing campaign for the customer.

Modules: Matplotlib and Seaborn for basic visualization and exploratory data analysis. 
Make use of pandas packages to wrangle the data. Some data wrangling techniques that we will be using are imputation of missing/ NA data values, 
and converting categorical variables to numeric variables using one hot encoding.

For classification:

Logistic Regression
Random Forests
K-Nearest Neighbours
Support Vector Machines
Neural Networks.

The preliminary challenges that you are likely to face will be in data wrangling and feature selection. 
Since there are 20 variables to fit the models and predict the outcome of the survey, it would be a challenge to select only those features which have a significant impact on the response variable. We plan to carry out feature engineering to create new features based on pre-existing ones.




