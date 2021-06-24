import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
df=pd.read_excel(r'C:\Users\tanya\Downloads\house-prices-advanced-regression-techniques\training.xlsx')
df1=pd.read_excel(r'C:\Users\tanya\Downloads\house-prices-advanced-regression-techniques\testing.xlsx')
#df1=pd.read_excel(r'C:\Users\tanya\Downloads\house-prices-advanced-regression-techniques\testing.xlsx')
categorical_data=['MSSubClass','MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
numerical_data=['SalePrice','LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']
numerical_data1=['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

import statistics
mean=statistics.mean(df['SalePrice'])
sd=statistics.stdev(df['SalePrice'])


def normal(df,i):
    df[i]=(df[i] - df[i].min()) / (df[i].max() - df[i].min())    

for i in numerical_data:
    normal(df,i)

def normal1(df1,i):
    df1[i]=(df1[i] - df1[i].min()) / (df1[i].max() - df1[i].min())    

for i in numerical_data1:
    normal(df1,i)

df.drop(['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'],axis=1)
df=pd.get_dummies(data=df,columns=categorical_data,drop_first=True)
#print(df.iloc[0])
print('*******************')
df1.drop(['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'],axis=1)
df1=pd.get_dummies(data=df1,columns=categorical_data,drop_first=True)

#Outlier_treatment
def outlier_treatment(x,df):
    Q1=df[x].quantile(0.25)
    Q3=df[x].quantile(0.75)
    IQR=Q3-Q1
    Lower_Whisker = Q1-1.5*IQR
    Upper_Whisker = Q3+1.5*IQR
    df[x]=df[df[x]<Upper_Whisker]
    df[x]=df[df[x]>Lower_Whisker]
 
#for i in numerical_data:
 #   outlier_treatment(i,df)
    

year=list(df['YearBuilt'])

for i in range(len(year)):
    year[i]=2021-year[i]

year1=list(df['YearRemodAdd'])

for i in range(len(year1)):
    year1[i]=2021-year1[i]
df['year']=year
df['year1']=year1


#Perform Corelation Test
year=list(df1['YearBuilt'])

for i in range(len(year)):
    year[i]=2021-year[i]

year1=list(df1['YearRemodAdd'])

for i in range(len(year1)):
    year1[i]=2021-year1[i]
df1['year']=year
df1['year1']=year1





remove1=['Utilities_NoSeWa','Condition2_RRAe','Condition2_RRAn','Condition2_RRNn','HouseStyle_2.5Fin','RoofMatl_CompShg','RoofMatl_Membran','RoofMatl_Metal','RoofMatl_Roll','Exterior1st_ImStucc','Exterior1st_Stone','Exterior2nd_Other','Heating_GasA','Heating_OthW','Electrical_Mix','GarageCond_Fa','PoolQC_Fa','MiscFeature_TenC']
remove2=['MSSubClass_150']
df=df.drop(columns=remove1,axis=1)
df1=df1.drop(columns=remove2,axis=1)


sales=['SalePrice']
#Applying Machine Learning Algorithms
y=df['SalePrice']
df=df.drop(columns=sales,axis=1)
df=df.fillna(0)
X=df

df1=df1.fillna(0)
X1=df1



import math
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
ml_model=LinearRegression()
print(ml_model)
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
model=ml_model.fit(X,y)
#print('Trainig score:{}',format(model.score(X_train,y_train)))
predictions=model.predict(X1)

print(len(predictions))
r=[]
for i in predictions:
    x=(i*sd)+mean
    #print(x)
    r.append(x)

print(r)








#r2_score=metrics.r2_score(y_test,predictions)
#print('r2 score is ',r2_score)
#mae=metrics.mean_absolute_error(y_test,predictions)
#print('Mae:',mae)
#rmse=math.sqrt(mae)
#print('rmse:',rmse)
#sns.distplot(y_test-predictions)









