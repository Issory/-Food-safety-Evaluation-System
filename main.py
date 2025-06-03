from django.shortcuts import render, redirect, HttpResponse
# from foodweb import models
import math
import prince
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from django.contrib.auth.decorators import login_required
import lightgbm as lgb
import catboost as cat
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # 均方误差
import numpy as np


def datapredict(request):
    trainfile = request.FILES.get("prefiletrain")
    textfile = request.FILES.get("prefiletest")
    train = pd.read_excel(trainfile)
    y_train = train.iloc[:, 0]
    x_train = train.iloc[:, 3:]
    text = pd.read_excel(textfile)
    y_test = text.iloc[:, 0]
    x_test = text.iloc[:, 3:]
    a1, a2, a3, a4, b1, b2, b3, b4 = 5, 5, 5, 9, 100, 100, 100, 150

    start = time.process_time()
    xgb_model = cat.CatBoostRegressor(max_depth=a1, learning_rate=0.1, n_estimators=b1)
    xgb_model.fit(x_train, y_train, verbose=False)
    x_train_leaves_xgb = pd.DataFrame(xgb_model.calc_leaf_indexes(x_train))
    x_train_leaves_xgb = x_train_leaves_xgb.add_suffix("_catleaf")
    x_test_leaves_xgb = pd.DataFrame(xgb_model.calc_leaf_indexes(x_test))
    x_test_leaves_xgb = x_test_leaves_xgb.add_suffix("_catleaf")

    gbm_model = lgb.LGBMRegressor(objective='regression', max_depth=a2, learning_rate=0.1, n_estimators=b2)  # ,
    gbm_model.fit(x_train, y_train)
    # 输入lightgbm叶子节点数据
    x_train_leaves_lig = pd.DataFrame(gbm_model.predict(x_train, pred_leaf=True))
    x_train_leaves_lig = x_train_leaves_lig.add_suffix('_ligleaf')
    x_test_leaves_lig = pd.DataFrame(gbm_model.predict(x_test, pred_leaf=True))
    x_test_leaves_lig = x_test_leaves_lig.add_suffix('_ligleaf')
    ###x_train1是原始数据和cat叶子节点数据结合x_train3是原始数据和xgb+lig叶子节点数据结合
    x_train1 = pd.concat([x_train, x_train_leaves_xgb], axis=1)
    x_train3 = pd.concat([x_train1, x_train_leaves_lig], axis=1)
    x_test1 = pd.concat([x_test, x_test_leaves_xgb], axis=1)
    x_test3 = pd.concat([x_test1, x_test_leaves_lig], axis=1)

    print("cat+lig+cat")

    xgb_model = cat.CatBoostRegressor(max_depth=9, learning_rate=0.1, n_estimators=150)
    xgb_model.fit(x_train3, y_train, verbose=False)
    pre = xgb_model.predict(x_test3)
    end = time.process_time()
    acc = xgb_model.score(x_test3, y_test)
    acc = '%.4f' % acc
    mse = mean_squared_error(pre, y_test)
    mse = '%.4f' % mse
    mae = mean_absolute_error(pre, y_test)
    mae = '%.4f' % mae
    totaltime = end - start
    totaltime = '%.4f' % totaltime
    text['预测值'] = 100-(max(pre)-pre)/(max(pre)-min(pre))*40

    food_list = text.groupby('食品大类')['预测值'].mean().reset_index()
    foodname_list = food_list['食品大类'].values.tolist()
    fooddata_list = (100 - (max(food_list['预测值']) * 100 - food_list['预测值'] * 100) / (
            max(food_list['预测值'] * 100) - min(food_list['预测值'] * 100)) * 5).values.tolist()

    city_list = text.groupby('被抽样地市')['预测值'].mean().reset_index()
    cityname_list = city_list['被抽样地市'].values.tolist()

    citydata_list = 100 - (max(city_list['预测值']) * 100 - city_list['预测值'] * 100) / (
            max(city_list['预测值'] * 100) - min(city_list['预测值'] * 100)) * 5
    city_list = (pd.concat([city_list['被抽样地市'], citydata_list], axis=1))
    citydata_list = citydata_list.values.tolist()
    text_index = text.index.values.tolist()

    text_pre = text["预测值"].values.tolist()
    text_true = 100-(max(text.iloc[:, 0])-text.iloc[:, 0])/(max(text.iloc[:, 0])-min(text.iloc[:, 0]))*40
    text_true = text_true.values.tolist()

    return render(request, "datapredict.html",
                  {"acc": acc, "mse": mse, "mae": mae, "totaltime": totaltime, "cityname_list": cityname_list,
                   "citydata_list": citydata_list, "foodname_list": foodname_list, "fooddata_list": fooddata_list,
                   "city_list": city_list, "text_index": text_index,
                   "text_pre": text_pre, "text_true": text_true})


def nodatapredict(request):
    xgb_model = cat.CatBoostRegressor()
    xgb_model.load_model('D:/minist.model')
    textfile = request.FILES.get("prefiletest")
    text = pd.read_excel(textfile)
    y_test = text.iloc[:, 0]
    x_test = text.iloc[:, 3:]
    pre = xgb_model.predict(x_test)
    acc = xgb_model.score(x_test, y_test)
    acc = '%.4f' % acc
    mse = mean_squared_error(pre, y_test)
    mse = '%.4f' % mse
    mae = mean_absolute_error(pre, y_test)
    mae = '%.4f' % mae
    totaltime = 0
    totaltime = '%.4f' % totaltime
    text['预测值'] = pre

    food_list = text.groupby('食品大类')['预测值'].mean().reset_index()
    foodname_list = food_list['食品大类'].values.tolist()
    fooddata_list = (95 - (max(food_list['预测值']) * 100 - food_list['预测值'] * 100) / (
            max(food_list['预测值'] * 100) - min(food_list['预测值'] * 100)) * 20).values.tolist()

    city_list = text.groupby('被抽样地市')['预测值'].mean().reset_index()
    cityname_list = city_list['被抽样地市'].values.tolist()

    citydata_list = 95 - (max(city_list['预测值']) * 100 - city_list['预测值'] * 100) / (
            max(city_list['预测值'] * 100) - min(city_list['预测值'] * 100)) * 20
    city_list = (pd.concat([city_list['被抽样地市'], citydata_list], axis=1))
    citydata_list = citydata_list.values.tolist()
    text_index = text.index.values.tolist()

    text_pre = text["预测值"].values.tolist()
    text_true = text.iloc[:, 0].values.tolist()

    return render(request, "datapredict.html",
                  {"acc": acc, "mse": mse, "mae": mae, "totaltime": totaltime, "cityname_list": cityname_list,
                   "citydata_list": citydata_list, "foodname_list": foodname_list, "fooddata_list": fooddata_list,
                   "city_list": city_list, "text_index": text_index,
                   "text_pre": text_pre, "text_true": text_true})



###6、最小样本量预测
def dataminsize(request):
    prefiletrainmin = request.FILES.get("prefiletrainmin")
    df = pd.read_excel(prefiletrainmin)
    EPOCHS = 5
    LABEL = '被抽样地市'
    CTSIZE = 20000


    beforenum = df.shape[0]

    train_df = df.drop([LABEL], axis=1)
    test_df = df[LABEL]
    ###使用CTGAN
    from _ctgan.synthesizer import _CTGANSynthesizer as CTGAN



    # 获取原始数据集每个类别的数量
    original_class_counts = df[LABEL].value_counts()
    ''''''
    ctgan = CTGAN(batch_size=500, patience=25)
    ctgan.fit(df, [LABEL,'食品大类']
              , EPOCHS)

    synthetic_data = ctgan.sample(CTSIZE)

    train_df = synthetic_data.drop([LABEL], axis=1)
    test_df = synthetic_data[LABEL]

    from imblearn.under_sampling import TomekLinks

    # 使用TomekLinks算法进行下采样
    tomek_links = TomekLinks(sampling_strategy='auto')

    resampled_features, resampled_labels = tomek_links.fit_resample(train_df, test_df)

    # 将下采样后的结果合并成DataFrame
    resampled_df = pd.DataFrame(data=resampled_features, columns=train_df.columns)
    resampled_df[LABEL] = resampled_labels
    afternum = resampled_df.shape[0]


    resampled_df['食品大类'] = resampled_df['食品大类'].replace(
        {8: '食用农产品', 3: '粮食加工品', 9: '食用油、油脂及其制品', 7: '豆制品', 4: '肉制品', 0: '乳制品',
         1: '方便制品', 2: '淀粉及淀粉制品', 6: '调味品', 5: '蔬菜制品', 10: '餐饮食品'})
    resampled_foodclass_counts = resampled_df['食品大类'].value_counts()
    foodname_list = resampled_foodclass_counts.index.tolist()
    fooddata_list = resampled_foodclass_counts.values.tolist()
    print(foodname_list)
    print(fooddata_list)

    ###将数据替换为地名
    resampled_df[LABEL] = resampled_df[LABEL].replace({0: 'XY', 12: 'XA', 3: 'BJ', 9: 'WN', 2: 'AK', 1: 'SL',
                                                       8: 'HZ', 7: 'YL1', 5: 'YA', 13: 'TC', 11: 'XX', 6: 'YL2',
                                                       14: 'HC', 10: 'SM', 4: 'FG'})
    # 输出下采样结果
    print(resampled_df)

    # 获取下采样后的数据集每个类别的数量
    resampled_class_counts = resampled_df['被抽样地市'].value_counts()

    cityname_list = resampled_class_counts.index.tolist()
    citydata_list = resampled_class_counts.values.tolist()


    # 输出原始数据集每个类别的数量
    print("Original Class Counts:")
    print(original_class_counts)

    # 输出下采样后的数据集每个类别的数量
    print("Resampled Class Counts:")
    print(resampled_class_counts)

    return render(request, "dataminsize.html",
                  {"beforenum": beforenum, "afternum": afternum,   "cityname_list": cityname_list,
                   "citydata_list": citydata_list, "foodname_list": foodname_list, "fooddata_list": fooddata_list,
                   "resampled_class_counts": resampled_class_counts.to_dict()})



###7、食品安全预测
def dataqualifipredict(request):
    from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score
    EPOCHS = 5
    LABEL = '合一不零'
    CTSIZE = 10000
    def gmeans(y_true, y_pred):
        gmeans = np.sqrt(balanced_accuracy_score(y_true, y_pred))
        return gmeans


    trainfile = request.FILES.get("prefiletrainyu")
    textfile = request.FILES.get("prefiletestyu")

    train = pd.read_excel(trainfile)
    train_X = train.drop([LABEL], axis=1)
    train_y = train[LABEL]
    text = pd.read_excel(textfile)
    test_x = text.drop([LABEL], axis=1)
    test_y = text[LABEL]
    #test_x=text

    from imblearn.over_sampling import SMOTE

    smote = SMOTE(random_state=0)

    # 用SMOTE模型合成样本
    train_X_resemple, train_y_resemple = smote.fit_resample(train_X, train_y)
    print('smote特征数据大小为', train_X_resemple.size)
    print('smote标签数据大小为', train_y_resemple.size)

    # 合并两个数据集
    train_X_resemple = pd.DataFrame(train_X_resemple)
    train_y_resemple = pd.DataFrame(train_y_resemple)

    data1 = pd.concat([train_X_resemple, train_y_resemple], axis=1)
    print('data1特征数据大小为', data1.size)

    data1 = pd.DataFrame(data1)

    ###使用CTGAN
    from _ctgan.synthesizer import _CTGANSynthesizer as CTGAN

    ctgan = CTGAN(batch_size=500, patience=25)

    ctgan.fit(data1, [LABEL], EPOCHS)

    synthetic_data = ctgan.sample(CTSIZE)
    print('synthetic_data特征数据大小为', synthetic_data.size)

    train_df = pd.concat([data1, synthetic_data]).reset_index(drop=True)

    ###进行对抗性过程
    train_df_dk = train_df.drop([LABEL], axis=1)
    test_df_dk = pd.DataFrame(test_x).copy()

    ###对训练集标签设置为0，测试集标签设置为1
    train_df_dk['label'] = 0
    test_df_dk['label'] = 1

    # 合并两个数据集
    data2 = pd.concat([train_df_dk, test_df_dk], axis=0)

    # 拆分特征和标签
    new_X = data2.drop(['label'], axis=1)
    new_y = data2['label']

    # 使用LGBMClassifier分类器

    from lightgbm import LGBMClassifier

    lgbm_model = LGBMClassifier()
    # 定义超参数空间
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7]
    }
    from sklearn.model_selection import GridSearchCV

    # 定义GridSearchCV对象
    grid_search = GridSearchCV(lgbm_model, param_grid, cv=5, scoring='roc_auc')

    # 在训练集上训练GridSearchCV对象
    grid_search.fit(new_X, new_y)

    # 输出最佳超参数组合和对应的AUC得分
    print("Best parameters: {}".format(grid_search.best_params_))
    print("Best cross-validation AUC score: {:.4f}".format(grid_search.best_score_))

    # 使用最佳超参数组合训练LGBMClassifier分类器
    model = LGBMClassifier(**grid_search.best_params_)
    model.fit(new_X, new_y)

    # 输出数据集2的标签概率值
    data2_prob = model.predict_proba(train_df_dk.drop(['label'], axis=1))[:, 1]

    # 对数据集2的标签概率值进行由高到低排序，筛选出前60%的顶部行
    top60_idx = data2_prob.argsort()[::-1][:int(len(train_df) * 0.9)]
    top60_rows = train_df.iloc[top60_idx]

    # train_X=top60_rows
    X_train_oversampled = top60_rows.iloc[:, 0:-1]
    y_train_oversampled = top60_rows[LABEL]

    from sklearn.ensemble import RandomForestClassifier

    RF_model = RandomForestClassifier()

    RF_model.fit(X_train_oversampled, y_train_oversampled)
    y_pred = model.predict(test_x)
    binf1 = f1_score(test_y, y_pred, average='binary')
    AUC_score = roc_auc_score(test_y, y_pred)
    Gmeans_score = np.sqrt(balanced_accuracy_score(test_y, y_pred))

    # 计算每个“食品大类”的合格率
    food_qualification_rate = text.groupby('食品大类')['合一不零'].apply(lambda x: (x == 1).mean()).reset_index()
    food_qualification_rate['食品大类'] = food_qualification_rate['食品大类'].replace(
        {8: '食用农产品', 3: '粮食加工品', 9: '食用油、油脂及其制品', 7: '豆制品', 4: '肉制品', 0: '乳制品',
         1: '方便制品', 2: '淀粉及淀粉制品', 6: '调味品', 5: '蔬菜制品', 10: '餐饮食品'})

    foodname_list = food_qualification_rate['食品大类'].values.tolist()
    fooddata_list = food_qualification_rate['合一不零'].values.tolist()

    city_zero_count = text.groupby('被抽样地市')['合一不零'].apply(lambda x: (x == 0).sum()).reset_index()
    city_zero_count['被抽样地市'] = city_zero_count['被抽样地市'].replace(
        {0: 'XY', 12: 'XA', 3: 'BJ', 9: 'WN', 2: 'AK', 1: 'SL',
         8: 'HZ', 7: 'YL', 5: 'YA', 13: 'TC', 11: 'XX', 6: 'YL',
         14: 'HC', 10: 'SM', 4: 'FG'})
    print(city_zero_count)
    city_zero_count = city_zero_count.values.tolist()
    # 计算每个“被抽样地市”的合格率
    city_qualification_rate = text.groupby('被抽样地市')['合一不零'].apply(lambda x: (x == 1).mean()).reset_index()
    city_qualification_rate['被抽样地市'] = city_qualification_rate['被抽样地市'].replace(
        {0: 'XY', 12: 'XA', 3: 'BJ', 9: 'WN', 2: 'AK', 1: 'SL',
         8: 'HZ', 7: 'YL', 5: 'YA', 13: 'TC', 11: 'XX', 6: 'YL',
         14: 'HC', 10: 'SM', 4: 'FG'})


    cityname_list = city_qualification_rate['被抽样地市'].values.tolist()
    citydata_list = city_qualification_rate['合一不零'].values.tolist()
    city_list = city_qualification_rate


    text_index = text.index.values.tolist()
    text_pre = y_pred
    text_true = text.iloc[:, 0].values.tolist()


    return render(request, "dataqualifipredict.html",
                  {"binf1": binf1, "Gmeans_score": Gmeans_score, "AUC_score": AUC_score,  "cityname_list": cityname_list,
                   "citydata_list": citydata_list, "foodname_list": foodname_list, "fooddata_list": fooddata_list,
                   "city_list": city_list, "text_index": text_index,
                   "text_pre": text_pre, "text_true": text_true,"city_zero_count":city_zero_count})

