"""
Test rapido: HasName + AdoptionFriction sobre FE v4
Usa los mejores params conocidos (sin Optuna) para comparar kappa en ~3 min
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import os, warnings
warnings.filterwarnings('ignore')

BASE_DIR = 'c:/Users/User/Desktop/MCD/Laboratorio de Implementacion II/GitHub/UA_MDM_Labo2'
SEED = 42

train_raw = pd.read_csv(os.path.join(BASE_DIR, 'input/train/train.csv'))
sent_df   = pd.read_csv(os.path.join(BASE_DIR, 'input/train_sentiment_features.csv'))
meta_df   = pd.read_csv(os.path.join(BASE_DIR, 'input/train_metadata_features.csv'))
train_raw['desc_length'] = train_raw['Description'].fillna('').apply(len)

df = (train_raw
      .merge(sent_df[['PetID','sentiment_score','sentiment_magnitude','n_sentences']], on='PetID', how='left')
      .merge(meta_df[['PetID','avg_label_score','n_labels','crop_confidence']], on='PetID', how='left')
      .fillna(0))

train, test = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df['AdoptionSpeed'])

def target_encode(train_df, test_df, col, target='AdoptionSpeed', smoothing=10):
    global_mean = train_df[target].mean()
    stats = train_df.groupby(col)[target].agg(['mean', 'count'])
    stats['encoded'] = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
    return train_df[col].map(stats['encoded']).fillna(global_mean), test_df[col].map(stats['encoded']).fillna(global_mean)

def add_features_v4(df_):
    df_ = df_.copy()
    df_['HasPhoto']            = (df_['PhotoAmt'] > 0).astype(int)
    df_['HasVideo']            = (df_['VideoAmt'] > 0).astype(int)
    df_['IsFree']              = (df_['Fee'] == 0).astype(int)
    df_['AgeGroup']            = pd.cut(df_['Age'], bins=[-1,3,12,48,9999], labels=[0,1,2,3]).astype(int)
    df_['HealthScore']         = ((df_['Vaccinated']==1).astype(int) + (df_['Dewormed']==1).astype(int) + (df_['Sterilized']==1).astype(int))
    df_['IsPureBreed']         = (df_['Breed2'] == 0).astype(int)
    df_['PhotoPerAnimal']      = df_['PhotoAmt'] / df_['Quantity'].replace(0,1)
    df_['Age_x_PhotoAmt']      = df_['Age'] * df_['PhotoAmt']
    df_['IsPureBreed_x_Age']   = df_['IsPureBreed'] * df_['AgeGroup']
    df_['HealthScore_x_Photo'] = df_['HealthScore'] * df_['HasPhoto']
    df_['IsYoungAndFree']      = ((df_['AgeGroup'] <= 1) & (df_['IsFree'] == 1)).astype(int)
    df_['IsHealthyAndPhoto']   = ((df_['HealthScore'] == 3) & (df_['HasPhoto'] == 1)).astype(int)
    df_['FeePerAnimal']        = df_['Fee'] / df_['Quantity'].replace(0,1)
    return df_

def nlp_feats(df_):
    desc = df_['Description'].apply(lambda x: '' if (x == 0 or str(x).strip() == '') else str(x))
    df_['word_count']      = desc.apply(lambda x: len(x.split()))
    df_['unique_words']    = desc.apply(lambda x: len(set(x.lower().split())))
    df_['avg_word_len']    = desc.apply(lambda x: round(sum(len(w) for w in x.split()) / max(len(x.split()),1), 2))
    df_['uppercase_ratio'] = desc.apply(lambda x: round(sum(c.isupper() for c in x) / max(len(x),1), 4))
    df_['has_exclamation'] = desc.apply(lambda x: int('!' in x))
    return df_

# --- FE v5: agrega HasName y AdoptionFriction ---
def add_features_v5(df_):
    df_ = add_features_v4(df_)
    # HasName: el animal tiene nombre asignado (> 2 chars)
    df_['HasName'] = df_['Name'].fillna('').apply(lambda x: 1 if len(str(x)) > 2 else 0)
    # AdoptionFriction: edad * salud_cruda / (nombre + raza_pura + salud_preventiva + 1)
    df_['AdoptionFriction'] = (df_['Age'] * df_['Health']) / (df_['HasName'] + df_['IsPureBreed'] + df_['HealthScore'] + 1)
    return df_

train = train.copy(); test = test.copy()
train['Breed1_enc'], test['Breed1_enc'] = target_encode(train, test, 'Breed1')
train['State_enc'],  test['State_enc']  = target_encode(train, test, 'State')

rescuer_count = train.groupby('RescuerID').size().rename('rescuer_n_pets')
train['rescuer_n_pets'] = train['RescuerID'].map(rescuer_count).fillna(1)
test['rescuer_n_pets']  = test['RescuerID'].map(rescuer_count).fillna(1)

age_med_map = train.groupby(['Breed1','Type'])['Age'].median().to_dict()
global_age  = train['Age'].median()
for df_ in [train, test]:
    df_['age_median_bt'] = [age_med_map.get((b,t), global_age) for b,t in zip(df_['Breed1'], df_['Type'])]
    df_['age_rel_breed'] = df_['Age'] / (df_['age_median_bt'] + 1)

train = nlp_feats(train); test = nlp_feats(test)

# FE v4 (baseline de comparacion)
train_v4 = add_features_v4(train); test_v4 = add_features_v4(test)
# FE v5 (con HasName y AdoptionFriction)
train_v5 = add_features_v5(train); test_v5 = add_features_v5(test)

ALL_FEATURES_V4 = [
    'Type','Age','Breed1','Breed2','Gender','Color1','Color2','Color3',
    'MaturitySize','FurLength','Vaccinated','Dewormed','Sterilized',
    'Health','Quantity','Fee','State','VideoAmt','PhotoAmt',
    'HasPhoto','HasVideo','IsFree','AgeGroup','HealthScore','IsPureBreed','PhotoPerAnimal',
    'Age_x_PhotoAmt','IsPureBreed_x_Age','HealthScore_x_Photo','IsYoungAndFree','IsHealthyAndPhoto','FeePerAnimal',
    'sentiment_score','sentiment_magnitude','n_sentences','avg_label_score','n_labels','crop_confidence','desc_length',
    'Breed1_enc','State_enc',
    'rescuer_n_pets','age_rel_breed','word_count','unique_words','avg_word_len','uppercase_ratio','has_exclamation'
]
ALL_FEATURES_V5 = ALL_FEATURES_V4 + ['HasName', 'AdoptionFriction']

# Mejores params del run de 30 trials
best_params = {'objective': 'multiclass', 'num_class': 5, 'verbosity': -1,
               'num_leaves': 51, 'lambda_l1': 0.10, 'lambda_l2': 7.58,
               'feature_fraction': 0.59, 'bagging_fraction': 0.98,
               'bagging_freq': 1, 'min_child_samples': 118, 'learning_rate': 0.093}

def run_cv(X_tr_all, X_te_all, features, label=''):
    X_train_ = X_tr_all[features]
    X_test_  = X_te_all[features]
    y_train_ = X_tr_all['AdoptionSpeed']
    y_test_  = X_te_all['AdoptionSpeed']

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    preds = np.zeros((len(X_test_), 5))
    cv_scores = []
    for tr_idx, val_idx in skf.split(X_train_, y_train_):
        X_tr, X_val = X_train_.iloc[tr_idx], X_train_.iloc[val_idx]
        y_tr, y_val = y_train_.iloc[tr_idx], y_train_.iloc[val_idx]
        m = lgb.train(best_params, lgb.Dataset(X_tr, label=y_tr),
                      num_boost_round=500, valid_sets=[lgb.Dataset(X_val, label=y_val)],
                      callbacks=[lgb.early_stopping(20, verbose=False)])
        cv_scores.append(cohen_kappa_score(y_val, m.predict(X_val).argmax(axis=1), weights='quadratic'))
        preds += m.predict(X_test_)
    kappa = cohen_kappa_score(y_test_, preds.argmax(axis=1), weights='quadratic')
    print(f'{label:35s}  CV: {np.mean(cv_scores):.4f}  Test: {kappa:.4f}  feat={len(features)}')
    return kappa

# Agregar AdoptionSpeed a los df para pasarlos a run_cv
train_v4['AdoptionSpeed'] = train['AdoptionSpeed']
test_v4['AdoptionSpeed']  = test['AdoptionSpeed']
train_v5['AdoptionSpeed'] = train['AdoptionSpeed']
test_v5['AdoptionSpeed']  = test['AdoptionSpeed']

print('='*70)
print('TEST RAPIDO: HasName + AdoptionFriction')
print('='*70)
k_v4 = run_cv(train_v4, test_v4, ALL_FEATURES_V4, 'FE v4 (48 feat)')
k_v5 = run_cv(train_v5, test_v5, ALL_FEATURES_V5, 'FE v5 +HasName +AdoptFric (50 feat)')
print('='*70)
print(f'Diferencia: {k_v5 - k_v4:+.4f}')
if k_v5 > k_v4:
    print('-> HasName + AdoptionFriction MEJORAN el kappa. Incorporar a FE v5.')
else:
    print('-> No mejoran. Mantener FE v4.')
print('='*70)
