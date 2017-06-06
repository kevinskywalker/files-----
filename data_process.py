# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 18:00:46 2017

@author: liujiacheng1
"""

import pandas as pd
import numpy as np
import matplotlib as plt

from sklearn.cross_validation import train_test_split

feature_list = [			"tag_ljc3",	"LotArea",	"OverallQual",		"YearBuilt",	"YearRemodAdd",	"MasVnrArea",	"BsmtFinSF1",			"TotalBsmtSF",	"1stFlrSF",	"2ndFlrSF",		"GrLivArea",			"FullBath",	"HalfBath",			"TotRmsAbvGrd",	"Fireplaces",		"GarageCars",	"GarageArea",	"WoodDeckSF",	"OpenPorchSF",													"MSZoning_RL",	"MSZoning_RM",									"LotShape_Reg",																															"Neighborhood_NoRidge",	"Neighborhood_NridgHt",																																				"HouseStyle_2Story",				"RoofStyle_Gable",		"RoofStyle_Hip",										"RoofMatl_WdShngl",														"Exterior1st_VinylSd",																	"Exterior2nd_VinylSd",						"MasVnrType_None",	"MasVnrType_Stone",	"ExterQual_Ex",		"ExterQual_Gd",	"ExterQual_TA",							"Foundation_CBlock",	"Foundation_PConc",					"BsmtQual_Ex",		"BsmtQual_Gd",	"BsmtQual_TA",								"BsmtExposure_Gd",						"BsmtFinType1_GLQ",																	"HeatingQC_Ex",				"HeatingQC_TA",										"KitchenQual_Ex",		"KitchenQual_Gd",	"KitchenQual_TA",									"FireplaceQu_0",	"FireplaceQu_Ex",		"FireplaceQu_Gd",					"GarageType_Attchd",				"GarageType_Detchd",		"GarageFinish_Fin",		"GarageFinish_Unf",						"GarageQual_TA",						"GarageCond_TA",	"PavedDrive_N",		"PavedDrive_Y",																						"SaleType_New",								"SaleCondition_Partial"]


def read_data():
    
    raw_data = pd.read_csv('train_test.csv').replace('NaN',0).replace('NA',0).fillna(0)
    
    

    raw_data = raw_data.fillna(method='ffill')
   # raw_data['unitprice'] = raw_data['SalePrice']/raw_data['LotArea']
    
    #feature generate
    raw_data['tag_ljc1'] = raw_data['GarageYrBlt'] - raw_data['YearBuilt']
    raw_data['tag_ljc2'] = raw_data['GarageYrBlt'] - raw_data['YearRemodAdd']    
    raw_data['tag_ljc3'] = raw_data['YearBuilt'] - raw_data['YrSold']

    #feature engineer
    qual_dict = {None: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
    raw_data["ExterQual"] = raw_data["ExterQual"].map(qual_dict)
    raw_data["ExterCond"] = raw_data["ExterCond"].map(qual_dict)
    raw_data["BsmtQual"] = raw_data["BsmtQual"].map(qual_dict)
    raw_data["BsmtCond"] = raw_data["BsmtCond"].map(qual_dict)
    raw_data["HeatingQC"] = raw_data["HeatingQC"].map(qual_dict)
    raw_data["KitchenQual"] = raw_data["KitchenQual"].map(qual_dict)
    raw_data["FireplaceQu"] = raw_data["FireplaceQu"].map(qual_dict)
    raw_data["GarageQual"] = raw_data["GarageQual"].map(qual_dict)
    raw_data["GarageCond"] = raw_data["GarageCond"].map(qual_dict)

    
    raw_data["BsmtExposure"] = raw_data["BsmtExposure"].map(
        {None: 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4})

    bsmt_fin_dict = {None: 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
    raw_data["BsmtFinType1"] = raw_data["BsmtFinType1"].map(bsmt_fin_dict)
    raw_data["BsmtFinType2"] = raw_data["BsmtFinType2"].map(bsmt_fin_dict)

    raw_data["Functional"] = raw_data["Functional"].map(
        {None: 0, "Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, 
         "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8})

    raw_data["GarageFinish"] = raw_data["GarageFinish"].map(
        {None: 0, "Unf": 1, "RFn": 2, "Fin": 3})

    raw_data["Fence"] = raw_data["Fence"].map(
        {None: 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4})
    
    
    
    raw_data["SaleCondition_PriceDown"] = raw_data.SaleCondition.replace(
        {'Abnorml': 1, 'Alloca': 1, 'AdjLand': 1, 'Family': 1, 'Normal': 0, 'Partial': 0})

    # House completed before sale or not
    raw_data["BoughtOffPlan"] = raw_data.SaleCondition.replace(
        {"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, "Family" : 0, "Normal" : 0, "Partial" : 1})
    


    area_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 
                 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'LowQualFinSF', 'PoolArea' ]
    raw_data["TotalArea"] = raw_data[area_cols].sum(axis=1)

    raw_data["TotalArea1st2nd"] = raw_data["1stFlrSF"] + raw_data["2ndFlrSF"]

    raw_data["Age"] = 2010 - raw_data["YearBuilt"]
    raw_data["TimeSinceSold"] = 2010 - raw_data["YrSold"]

    raw_data["SeasonSold"] = raw_data["MoSold"].map({12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 
                                                  6:2, 7:2, 8:2, 9:3, 10:3, 11:3})
    
    raw_data["YearsSinceRemodel"] = raw_data["YrSold"] - raw_data["YearRemodAdd"]
    
    # Simplifications of existing features into bad/average/good.
    raw_data["SimplOverallQual"] = raw_data.OverallQual.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2, 7 : 3, 8 : 3, 9 : 3, 10 : 3})
    raw_data["SimplOverallCond"] = raw_data.OverallCond.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2, 7 : 3, 8 : 3, 9 : 3, 10 : 3})
    raw_data["SimplPoolQC"] = raw_data.PoolQC.replace(
        {1 : 1, 2 : 1, 3 : 2, 4 : 2})
    raw_data["SimplGarageCond"] = raw_data.GarageCond.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    raw_data["SimplGarageQual"] = raw_data.GarageQual.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    raw_data["SimplFireplaceQu"] = raw_data.FireplaceQu.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    raw_data["SimplFireplaceQu"] = raw_data.FireplaceQu.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    raw_data["SimplFunctional"] = raw_data.Functional.replace(
        {1 : 1, 2 : 1, 3 : 2, 4 : 2, 5 : 3, 6 : 3, 7 : 3, 8 : 4})
    raw_data["SimplKitchenQual"] = raw_data.KitchenQual.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    raw_data["SimplHeatingQC"] = raw_data.HeatingQC.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    raw_data["SimplBsmtFinType1"] = raw_data.BsmtFinType1.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2})
    raw_data["SimplBsmtFinType2"] = raw_data.BsmtFinType2.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2})
    raw_data["SimplBsmtCond"] = raw_data.BsmtCond.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    raw_data["SimplBsmtQual"] = raw_data.BsmtQual.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    raw_data["SimplExterCond"] = raw_data.ExterCond.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    raw_data["SimplExterQual"] = raw_data.ExterQual.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
            
    # Bin by neighborhood (a little arbitrarily). Values were computed by: 
    # train_raw_data["SalePrice"].groupby(train_raw_data["Neighborhood"]).median().sort_values()
    neighborhood_map = {
        "MeadowV" : 0,  #  88000
        "IDOTRR" : 1,   # 103000
        "BrDale" : 1,   # 106000
        "OldTown" : 1,  # 119000
        "Edwards" : 1,  # 119500
        "BrkSide" : 1,  # 124300
        "Sawyer" : 1,   # 135000
        "Blueste" : 1,  # 137500
        "SWISU" : 2,    # 139500
        "NAmes" : 2,    # 140000
        "NPkVill" : 2,  # 146000
        "Mitchel" : 2,  # 153500
        "SawyerW" : 2,  # 179900
        "Gilbert" : 2,  # 181000
        "NWAmes" : 2,   # 182900
        "Blmngtn" : 2,  # 191000
        "CollgCr" : 2,  # 197200
        "ClearCr" : 3,  # 200250
        "Crawfor" : 3,  # 200624
        "Veenker" : 3,  # 218000
        "Somerst" : 3,  # 225500
        "Timber" : 3,   # 228475
        "StoneBr" : 4,  # 278000
        "NoRidge" : 4,  # 290000
        "NridgHt" : 4,  # 315000
    }
    
    raw_data["NeighborhoodBin"] = raw_data["Neighborhood"].map(neighborhood_map)
    
    hot_data = pd.get_dummies(raw_data) 
    feature_list = ["OverallQual",	"YearBuilt",	"YearRemodAdd",	"MasVnrArea",	"BsmtFinSF1",	"TotalBsmtSF",	"1stFlrSF",	"GrLivArea",	"FullBath",	"TotRmsAbvGrd",	"Fireplaces",	"GarageCars",	"GarageArea"] 
    #draw_data = hot_data.apply(plt.)
    #print(hot_data[feature_list])
    
    #hot_data.to_csv('hot_data1.csv')
   # print(hot_data)
    

    return hot_data
    
def test_data():
    
    raw_data = pd.read_csv('test (1).csv').replace('NaN',0).fillna(0)
    
    hot_data = pd.get_dummies(raw_data)
    
    
    
    #draw_data = hot_data.apply(plt.)
    
    
    
    #print(hot_data)
    

    return hot_data    
    
def house_xgboost(data,testdata):
    
    data=data.apply(pd.to_numeric, errors='ignore')
    
    import xgboost as xgb
   
    train_x,test_x,train_y,test_y = train_test_split(data[feature_list],data['SalePrice'],test_size=0.5008564576909901,random_state=False)
   
    traindata = data.iloc[:1460,:]
    testdata = data.iloc[1460:,:]
    
    
    train_x = traindata[feature_list]
    
    
    train_y = traindata['SalePrice']
    
    test_x = testdata[feature_list]
    
    test_y = testdata['SalePrice']
    
    
    
    
    
    print(data['SalePrice'])
    print(test_y)
    
    testdata = train_test_split(testdata.iloc[:,:],test_size=1)
    print(train_y)
   # print(data[0])
    data_train = xgb.DMatrix(train_x,label=train_y)
    data_test = xgb.DMatrix(test_x,label=test_y)
    print('ok2')
    param = {'max_depth':6, 'eta':0.05, 'silent':0, 'objective':'reg:linear' }
    num_round = 5000
    watch_list = [(data_train,'rmse')]
    bst = xgb.train(param, data_train, num_round,evals = watch_list)
    
    
# make prediction
    preds = bst.predict(data_test,ntree_limit = bst.best_ntree_limit)
    print('ok')
    print(preds)
    
    index_real =  pd.read_csv('a.csv').reset_index(drop=False)
    
    
    
    print(index_real['Id'])
    
    
    results = pd.DataFrame(preds)
    
    
    results['Id'] = index_real['Id']
    print(results)
    results.to_csv('t.csv',header = ['SalePrice','Id'],index=False)
    print(results)

    #print(data[1])
    
    '''
    data_train = xgb.DMatrix(x_train,label=y_train) 

    data_train = xgb.DMatrix(x_train,label=y_train)  
    data_test = xgb.DMatrix(x_test, label=y_test)  
    watch_list = [(data_test, 'eval'), (data_train, 'train')]  
    param = {'max_depth': 6, 'eta': 0.8, 'silent': 1, 'objective': 'binary:logistic'}  
    bst = xgb.train(param, data_train, num_boost_round=100, evals=watch_list)  
    y_hat = bst.predict(data_test)  
    y_hat[y_hat > 0.5] = 1  
    y_hat[~(y_hat > 0.5)] = 0  
    xgb_rate = show_accuracy(y_hat, y_test, 'XGBoost ')  
    totalSurvival(y_hat,'xgboost') 
    
    '''   
    
    
    
def house_xgboost1(data,testdata):
    
    import xgboost as xgb
   
    
    data_x = data.drop('SalePrice',axis=1).fillna(0)
    
    
    testdata = data.iloc[1460:,:]
    traindata = data.iloc[:1460,:]

    testdata_x = data_x.iloc[1460:,:]
    traindata_x = data_x.iloc[:1460,:]
    #data_train=data[feature_list]
    data_x = data.drop('SalePrice',axis=1)
    
    #train_x,test_x,train_y,test_y = train_test_split(data_train,data['SalePrice'],test_size=0.5008564576909901,random_state=False)
   
    traindata = data.iloc[:1460,:]
   
    
    
    train_x = traindata_x
    
    
    train_y = traindata['SalePrice']
    
    test_x = testdata_x
    
    test_y = testdata['SalePrice']
    
  
    print(data['SalePrice'])
    print(test_y)
    
    testdata = train_test_split(testdata.iloc[:,:],test_size=1)
    print(train_y)
   # print(data[0])
    data_train = xgb.DMatrix(train_x,label=train_y)
    data_test = xgb.DMatrix(test_x,label=test_y)
    print('ok2')
    param = {'max_depth':6, 'eta':0.05, 'silent':0, 'objective':'reg:linear' }
    num_round = 5000
    watch_list = [(data_train,'rmse')]
    bst = xgb.train(param, data_train, num_round,evals = watch_list)
    
    
# make prediction
    y_pred_xgb = bst.predict(data_test,ntree_limit = bst.best_ntree_limit)
    
    '''
    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    '''
    
    from sklearn.linear_model import Lasso

# I found this best alpha through cross-validation.
    best_alpha = 0.00099
    print('ok')
    regr = Lasso(alpha=best_alpha, max_iter=50000)
    regr.fit(train_x, train_y)

    # Run prediction on training set to get a rough idea of how well it does.
    y_pred_lasso = regr.predict(test_x)

    lasso = pd.DataFrame(y_pred_lasso).to_csv('lasso.csv')

################################################################################

# Blend the results of the two regressors and save the prediction to a CSV file.

    y_pred = (y_pred_xgb + y_pred_lasso) / 2
    #y_pred = np.exp(y_pred)
    

    '''
    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    '''
    
    
    
    print('ok')
    
    index_real =  pd.read_csv('a.csv').reset_index(drop=False)
    
    
    
    print(index_real['Id'])
    
    
    results = pd.DataFrame(y_pred)
    
    
    results['Id'] = index_real['Id']
    print(results)
    results.to_csv('final.csv',header = ['SalePrice','Id'],index=False)
    print(results)

    #print(data[1])
    
    '''
    data_train = xgb.DMatrix(x_train,label=y_train) 

    data_train = xgb.DMatrix(x_train,label=y_train)  
    data_test = xgb.DMatrix(x_test, label=y_test)  
    watch_list = [(data_test, 'eval'), (data_train, 'train')]  
    param = {'max_depth': 6, 'eta': 0.8, 'silent': 1, 'objective': 'binary:logistic'}  
    bst = xgb.train(param, data_train, num_boost_round=100, evals=watch_list)  
    y_hat = bst.predict(data_test)  
    y_hat[y_hat > 0.5] = 1  
    y_hat[~(y_hat > 0.5)] = 0  
    xgb_rate = show_accuracy(y_hat, y_test, 'XGBoost ')  
    totalSurvival(y_hat,'xgboost') 
    
    '''   
    
   
house_xgboost1(read_data(),test_data())














