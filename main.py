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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # �������
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
    # ����lightgbmҶ�ӽڵ�����
    x_train_leaves_lig = pd.DataFrame(gbm_model.predict(x_train, pred_leaf=True))
    x_train_leaves_lig = x_train_leaves_lig.add_suffix('_ligleaf')
    x_test_leaves_lig = pd.DataFrame(gbm_model.predict(x_test, pred_leaf=True))
    x_test_leaves_lig = x_test_leaves_lig.add_suffix('_ligleaf')
    ###x_train1��ԭʼ���ݺ�catҶ�ӽڵ����ݽ��x_train3��ԭʼ���ݺ�xgb+ligҶ�ӽڵ����ݽ��
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
    text['Ԥ��ֵ'] = 100-(max(pre)-pre)/(max(pre)-min(pre))*40

    food_list = text.groupby('ʳƷ����')['Ԥ��ֵ'].mean().reset_index()
    foodname_list = food_list['ʳƷ����'].values.tolist()
    fooddata_list = (100 - (max(food_list['Ԥ��ֵ']) * 100 - food_list['Ԥ��ֵ'] * 100) / (
            max(food_list['Ԥ��ֵ'] * 100) - min(food_list['Ԥ��ֵ'] * 100)) * 5).values.tolist()

    city_list = text.groupby('����������')['Ԥ��ֵ'].mean().reset_index()
    cityname_list = city_list['����������'].values.tolist()

    citydata_list = 100 - (max(city_list['Ԥ��ֵ']) * 100 - city_list['Ԥ��ֵ'] * 100) / (
            max(city_list['Ԥ��ֵ'] * 100) - min(city_list['Ԥ��ֵ'] * 100)) * 5
    city_list = (pd.concat([city_list['����������'], citydata_list], axis=1))
    citydata_list = citydata_list.values.tolist()
    text_index = text.index.values.tolist()

    text_pre = text["Ԥ��ֵ"].values.tolist()
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
    text['Ԥ��ֵ'] = pre

    food_list = text.groupby('ʳƷ����')['Ԥ��ֵ'].mean().reset_index()
    foodname_list = food_list['ʳƷ����'].values.tolist()
    fooddata_list = (95 - (max(food_list['Ԥ��ֵ']) * 100 - food_list['Ԥ��ֵ'] * 100) / (
            max(food_list['Ԥ��ֵ'] * 100) - min(food_list['Ԥ��ֵ'] * 100)) * 20).values.tolist()

    city_list = text.groupby('����������')['Ԥ��ֵ'].mean().reset_index()
    cityname_list = city_list['����������'].values.tolist()

    citydata_list = 95 - (max(city_list['Ԥ��ֵ']) * 100 - city_list['Ԥ��ֵ'] * 100) / (
            max(city_list['Ԥ��ֵ'] * 100) - min(city_list['Ԥ��ֵ'] * 100)) * 20
    city_list = (pd.concat([city_list['����������'], citydata_list], axis=1))
    citydata_list = citydata_list.values.tolist()
    text_index = text.index.values.tolist()

    text_pre = text["Ԥ��ֵ"].values.tolist()
    text_true = text.iloc[:, 0].values.tolist()

    return render(request, "datapredict.html",
                  {"acc": acc, "mse": mse, "mae": mae, "totaltime": totaltime, "cityname_list": cityname_list,
                   "citydata_list": citydata_list, "foodname_list": foodname_list, "fooddata_list": fooddata_list,
                   "city_list": city_list, "text_index": text_index,
                   "text_pre": text_pre, "text_true": text_true})



###6����С������Ԥ��
def dataminsize(request):
    prefiletrainmin = request.FILES.get("prefiletrainmin")
    df = pd.read_excel(prefiletrainmin)
    EPOCHS = 5
    LABEL = '����������'
    CTSIZE = 20000


    beforenum = df.shape[0]

    train_df = df.drop([LABEL], axis=1)
    test_df = df[LABEL]
    ###ʹ��CTGAN
    from _ctgan.synthesizer import _CTGANSynthesizer as CTGAN



    # ��ȡԭʼ���ݼ�ÿ����������
    original_class_counts = df[LABEL].value_counts()
    ''''''
    ctgan = CTGAN(batch_size=500, patience=25)
    ctgan.fit(df, [LABEL,'ʳƷ����']
              , EPOCHS)

    synthetic_data = ctgan.sample(CTSIZE)

    train_df = synthetic_data.drop([LABEL], axis=1)
    test_df = synthetic_data[LABEL]

    from imblearn.under_sampling import TomekLinks

    # ʹ��TomekLinks�㷨�����²���
    tomek_links = TomekLinks(sampling_strategy='auto')

    resampled_features, resampled_labels = tomek_links.fit_resample(train_df, test_df)

    # ���²�����Ľ���ϲ���DataFrame
    resampled_df = pd.DataFrame(data=resampled_features, columns=train_df.columns)
    resampled_df[LABEL] = resampled_labels
    afternum = resampled_df.shape[0]


    resampled_df['ʳƷ����'] = resampled_df['ʳƷ����'].replace(
        {8: 'ʳ��ũ��Ʒ', 3: '��ʳ�ӹ�Ʒ', 9: 'ʳ���͡���֬������Ʒ', 7: '����Ʒ', 4: '����Ʒ', 0: '����Ʒ',
         1: '������Ʒ', 2: '��ۼ������Ʒ', 6: '��ζƷ', 5: '�߲���Ʒ', 10: '����ʳƷ'})
    resampled_foodclass_counts = resampled_df['ʳƷ����'].value_counts()
    foodname_list = resampled_foodclass_counts.index.tolist()
    fooddata_list = resampled_foodclass_counts.values.tolist()
    print(foodname_list)
    print(fooddata_list)

    ###�������滻Ϊ����
    resampled_df[LABEL] = resampled_df[LABEL].replace({0: 'XY', 12: 'XA', 3: 'BJ', 9: 'WN', 2: 'AK', 1: 'SL',
                                                       8: 'HZ', 7: 'YL1', 5: 'YA', 13: 'TC', 11: 'XX', 6: 'YL2',
                                                       14: 'HC', 10: 'SM', 4: 'FG'})
    # ����²������
    print(resampled_df)

    # ��ȡ�²���������ݼ�ÿ����������
    resampled_class_counts = resampled_df['����������'].value_counts()

    cityname_list = resampled_class_counts.index.tolist()
    citydata_list = resampled_class_counts.values.tolist()


    # ���ԭʼ���ݼ�ÿ����������
    print("Original Class Counts:")
    print(original_class_counts)

    # ����²���������ݼ�ÿ����������
    print("Resampled Class Counts:")
    print(resampled_class_counts)

    return render(request, "dataminsize.html",
                  {"beforenum": beforenum, "afternum": afternum,   "cityname_list": cityname_list,
                   "citydata_list": citydata_list, "foodname_list": foodname_list, "fooddata_list": fooddata_list,
                   "resampled_class_counts": resampled_class_counts.to_dict()})



###7��ʳƷ��ȫԤ��
def dataqualifipredict(request):
    from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score
    EPOCHS = 5
    LABEL = '��һ����'
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

    # ��SMOTEģ�ͺϳ�����
    train_X_resemple, train_y_resemple = smote.fit_resample(train_X, train_y)
    print('smote�������ݴ�СΪ', train_X_resemple.size)
    print('smote��ǩ���ݴ�СΪ', train_y_resemple.size)

    # �ϲ��������ݼ�
    train_X_resemple = pd.DataFrame(train_X_resemple)
    train_y_resemple = pd.DataFrame(train_y_resemple)

    data1 = pd.concat([train_X_resemple, train_y_resemple], axis=1)
    print('data1�������ݴ�СΪ', data1.size)

    data1 = pd.DataFrame(data1)

    ###ʹ��CTGAN
    from _ctgan.synthesizer import _CTGANSynthesizer as CTGAN

    ctgan = CTGAN(batch_size=500, patience=25)

    ctgan.fit(data1, [LABEL], EPOCHS)

    synthetic_data = ctgan.sample(CTSIZE)
    print('synthetic_data�������ݴ�СΪ', synthetic_data.size)

    train_df = pd.concat([data1, synthetic_data]).reset_index(drop=True)

    ###���жԿ��Թ���
    train_df_dk = train_df.drop([LABEL], axis=1)
    test_df_dk = pd.DataFrame(test_x).copy()

    ###��ѵ������ǩ����Ϊ0�����Լ���ǩ����Ϊ1
    train_df_dk['label'] = 0
    test_df_dk['label'] = 1

    # �ϲ��������ݼ�
    data2 = pd.concat([train_df_dk, test_df_dk], axis=0)

    # ��������ͱ�ǩ
    new_X = data2.drop(['label'], axis=1)
    new_y = data2['label']

    # ʹ��LGBMClassifier������

    from lightgbm import LGBMClassifier

    lgbm_model = LGBMClassifier()
    # ���峬�����ռ�
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7]
    }
    from sklearn.model_selection import GridSearchCV

    # ����GridSearchCV����
    grid_search = GridSearchCV(lgbm_model, param_grid, cv=5, scoring='roc_auc')

    # ��ѵ������ѵ��GridSearchCV����
    grid_search.fit(new_X, new_y)

    # �����ѳ�������ϺͶ�Ӧ��AUC�÷�
    print("Best parameters: {}".format(grid_search.best_params_))
    print("Best cross-validation AUC score: {:.4f}".format(grid_search.best_score_))

    # ʹ����ѳ��������ѵ��LGBMClassifier������
    model = LGBMClassifier(**grid_search.best_params_)
    model.fit(new_X, new_y)

    # ������ݼ�2�ı�ǩ����ֵ
    data2_prob = model.predict_proba(train_df_dk.drop(['label'], axis=1))[:, 1]

    # �����ݼ�2�ı�ǩ����ֵ�����ɸߵ�������ɸѡ��ǰ60%�Ķ�����
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

    # ����ÿ����ʳƷ���ࡱ�ĺϸ���
    food_qualification_rate = text.groupby('ʳƷ����')['��һ����'].apply(lambda x: (x == 1).mean()).reset_index()
    food_qualification_rate['ʳƷ����'] = food_qualification_rate['ʳƷ����'].replace(
        {8: 'ʳ��ũ��Ʒ', 3: '��ʳ�ӹ�Ʒ', 9: 'ʳ���͡���֬������Ʒ', 7: '����Ʒ', 4: '����Ʒ', 0: '����Ʒ',
         1: '������Ʒ', 2: '��ۼ������Ʒ', 6: '��ζƷ', 5: '�߲���Ʒ', 10: '����ʳƷ'})

    foodname_list = food_qualification_rate['ʳƷ����'].values.tolist()
    fooddata_list = food_qualification_rate['��һ����'].values.tolist()

    city_zero_count = text.groupby('����������')['��һ����'].apply(lambda x: (x == 0).sum()).reset_index()
    city_zero_count['����������'] = city_zero_count['����������'].replace(
        {0: 'XY', 12: 'XA', 3: 'BJ', 9: 'WN', 2: 'AK', 1: 'SL',
         8: 'HZ', 7: 'YL', 5: 'YA', 13: 'TC', 11: 'XX', 6: 'YL',
         14: 'HC', 10: 'SM', 4: 'FG'})
    print(city_zero_count)
    city_zero_count = city_zero_count.values.tolist()
    # ����ÿ�������������С��ĺϸ���
    city_qualification_rate = text.groupby('����������')['��һ����'].apply(lambda x: (x == 1).mean()).reset_index()
    city_qualification_rate['����������'] = city_qualification_rate['����������'].replace(
        {0: 'XY', 12: 'XA', 3: 'BJ', 9: 'WN', 2: 'AK', 1: 'SL',
         8: 'HZ', 7: 'YL', 5: 'YA', 13: 'TC', 11: 'XX', 6: 'YL',
         14: 'HC', 10: 'SM', 4: 'FG'})


    cityname_list = city_qualification_rate['����������'].values.tolist()
    citydata_list = city_qualification_rate['��һ����'].values.tolist()
    city_list = city_qualification_rate


    text_index = text.index.values.tolist()
    text_pre = y_pred
    text_true = text.iloc[:, 0].values.tolist()


    return render(request, "dataqualifipredict.html",
                  {"binf1": binf1, "Gmeans_score": Gmeans_score, "AUC_score": AUC_score,  "cityname_list": cityname_list,
                   "citydata_list": citydata_list, "foodname_list": foodname_list, "fooddata_list": fooddata_list,
                   "city_list": city_list, "text_index": text_index,
                   "text_pre": text_pre, "text_true": text_true,"city_zero_count":city_zero_count})

