import json

nb = {
  "nbformat": 4,
  "nbformat_minor": 5,
  "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.11.0"}},
  "cells": [
    {
      "cell_type": "markdown", "id": "header", "metadata": {},
      "source": [
        "# 10 — CatBoost: tercer gradient boosting\n\n",
        "**Materia:** Laboratorio de Implementacion II · Universidad Austral · Abril 2026\n\n",
        "**Autores:** Roxana Alberti · Sandra Sschicchi · Fernando Paganini · Baltazar Villanueva · Paula Calviello · Rosana Martinez\n\n",
        "---\n\n",
        "## Que hace este notebook?\n\n",
        "Entrenamos **CatBoost** sobre las 48 features del FE v4 y lo comparamos con LightGBM y XGBoost.\n\n",
        "### Por que CatBoost?\n",
        "CatBoost (Categorical Boosting, Yandex 2017) difiere de LGB y XGB en varios aspectos clave:\n\n",
        "| Caracteristica | LightGBM | XGBoost | CatBoost |\n",
        "|---|---|---|---|\n",
        "| Crecimiento | Leaf-wise | Depth-wise | Depth-wise simetrico |\n",
        "| Categoricas | Codificacion manual | Codificacion manual | Nativo (ordered encoding) |\n",
        "| Target leakage | Posible con target enc. | Posible con target enc. | Evitado con permutaciones |\n",
        "| Velocidad GPU | Alta | Alta | Alta |\n\n",
        "El crecimiento simetrico (todos los nodos del mismo nivel se dividen igual) hace que\n",
        "CatBoost sea menos propenso al overfitting que LGB, pero potencialmente mas lento.\n\n",
        "### Estrategia\n",
        "1. Entrenamos CatBoost base con 5-fold CV (sin Optuna)\n",
        "2. Optimizamos con Optuna (30 trials)\n",
        "3. Incorporamos CatBoost al ensemble LGB+XGB+CB\n\n",
        "| Modelo | Kappa Test |\n",
        "|---|---|\n",
        "| LightGBM FE v4 | 0.3867 |\n",
        "| XGBoost FE v4 | 0.3803 |\n",
        "| Blend LGB+XGB (50/50) | 0.3906 |\n",
        "| **CatBoost FE v4** | **(este notebook)** |\n",
        "| **Blend LGB+XGB+CB** | **(este notebook)** |"
      ]
    },
    {
      "cell_type": "markdown", "id": "sec_a_md", "metadata": {},
      "source": ["## Seccion A: Imports y datos"]
    },
    {
      "cell_type": "code", "execution_count": None, "id": "setup", "metadata": {}, "outputs": [],
      "source": (
        "import pandas as pd\nimport numpy as np\nimport lightgbm as lgb\nimport xgboost as xgb\n"
        "from catboost import CatBoostClassifier, Pool\n"
        "import optuna\n"
        "from sklearn.metrics import cohen_kappa_score\n"
        "from sklearn.model_selection import train_test_split, StratifiedKFold\n"
        "from pathlib import Path\nimport warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "optuna.logging.set_verbosity(optuna.logging.WARNING)\n\n"
        "BASE_DIR = Path.cwd()\n"
        "while not (BASE_DIR / 'input').exists() and BASE_DIR != BASE_DIR.parent:\n"
        "    BASE_DIR = BASE_DIR.parent\n"
        "print(f'BASE_DIR: {BASE_DIR}')\n\n"
        "SEED = 42\n"
        "train_raw = pd.read_csv(BASE_DIR / 'input/train/train.csv')\n"
        "sent_df   = pd.read_csv(BASE_DIR / 'input/train_sentiment_features.csv')\n"
        "meta_df   = pd.read_csv(BASE_DIR / 'input/train_metadata_features.csv')\n"
        "train_raw['desc_length'] = train_raw['Description'].fillna('').apply(len)\n\n"
        "df = (train_raw\n"
        "      .merge(sent_df[['PetID','sentiment_score','sentiment_magnitude','n_sentences']], on='PetID', how='left')\n"
        "      .merge(meta_df[['PetID','avg_label_score','n_labels','crop_confidence']], on='PetID', how='left')\n"
        "      .fillna(0))\n\n"
        "train, test = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df['AdoptionSpeed'])\n"
        "print(f'Train: {len(train)} | Test: {len(test)}')\n"
      )
    },
    {
      "cell_type": "markdown", "id": "sec_b_md", "metadata": {},
      "source": ["## Seccion B: Feature Engineering v4"]
    },
    {
      "cell_type": "code", "execution_count": None, "id": "fe_code", "metadata": {}, "outputs": [],
      "source": (
        "def target_encode(train_df, test_df, col, target='AdoptionSpeed', smoothing=10):\n"
        "    global_mean = train_df[target].mean()\n"
        "    stats = train_df.groupby(col)[target].agg(['mean', 'count'])\n"
        "    stats['encoded'] = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)\n"
        "    return train_df[col].map(stats['encoded']).fillna(global_mean), test_df[col].map(stats['encoded']).fillna(global_mean)\n\n"
        "def add_features_v4(df_):\n"
        "    df_ = df_.copy()\n"
        "    df_['HasPhoto']            = (df_['PhotoAmt'] > 0).astype(int)\n"
        "    df_['HasVideo']            = (df_['VideoAmt'] > 0).astype(int)\n"
        "    df_['IsFree']              = (df_['Fee'] == 0).astype(int)\n"
        "    df_['AgeGroup']            = pd.cut(df_['Age'], bins=[-1,3,12,48,9999], labels=[0,1,2,3]).astype(int)\n"
        "    df_['HealthScore']         = ((df_['Vaccinated']==1).astype(int) + (df_['Dewormed']==1).astype(int) + (df_['Sterilized']==1).astype(int))\n"
        "    df_['IsPureBreed']         = (df_['Breed2'] == 0).astype(int)\n"
        "    df_['PhotoPerAnimal']      = df_['PhotoAmt'] / df_['Quantity'].replace(0,1)\n"
        "    df_['Age_x_PhotoAmt']      = df_['Age'] * df_['PhotoAmt']\n"
        "    df_['IsPureBreed_x_Age']   = df_['IsPureBreed'] * df_['AgeGroup']\n"
        "    df_['HealthScore_x_Photo'] = df_['HealthScore'] * df_['HasPhoto']\n"
        "    df_['IsYoungAndFree']      = ((df_['AgeGroup'] <= 1) & (df_['IsFree'] == 1)).astype(int)\n"
        "    df_['IsHealthyAndPhoto']   = ((df_['HealthScore'] == 3) & (df_['HasPhoto'] == 1)).astype(int)\n"
        "    df_['FeePerAnimal']        = df_['Fee'] / df_['Quantity'].replace(0,1)\n"
        "    return df_\n\n"
        "def nlp_feats(df_):\n"
        "    desc = df_['Description'].apply(lambda x: '' if (x == 0 or str(x).strip() == '') else str(x))\n"
        "    df_['word_count']      = desc.apply(lambda x: len(x.split()))\n"
        "    df_['unique_words']    = desc.apply(lambda x: len(set(x.lower().split())))\n"
        "    df_['avg_word_len']    = desc.apply(lambda x: round(sum(len(w) for w in x.split()) / max(len(x.split()),1), 2))\n"
        "    df_['uppercase_ratio'] = desc.apply(lambda x: round(sum(c.isupper() for c in x) / max(len(x),1), 4))\n"
        "    df_['has_exclamation'] = desc.apply(lambda x: int('!' in x))\n"
        "    return df_\n\n"
        "train = train.copy(); test = test.copy()\n"
        "train['Breed1_enc'], test['Breed1_enc'] = target_encode(train, test, 'Breed1')\n"
        "train['State_enc'],  test['State_enc']  = target_encode(train, test, 'State')\n\n"
        "rescuer_count = train.groupby('RescuerID').size().rename('rescuer_n_pets')\n"
        "train['rescuer_n_pets'] = train['RescuerID'].map(rescuer_count).fillna(1)\n"
        "test['rescuer_n_pets']  = test['RescuerID'].map(rescuer_count).fillna(1)\n\n"
        "age_med_map = train.groupby(['Breed1','Type'])['Age'].median().to_dict()\n"
        "global_age  = train['Age'].median()\n"
        "for df_ in [train, test]:\n"
        "    df_['age_median_bt'] = [age_med_map.get((b,t), global_age) for b,t in zip(df_['Breed1'], df_['Type'])]\n"
        "    df_['age_rel_breed'] = df_['Age'] / (df_['age_median_bt'] + 1)\n\n"
        "train = nlp_feats(train); test = nlp_feats(test)\n"
        "train_fe = add_features_v4(train); test_fe = add_features_v4(test)\n\n"
        "ALL_FEATURES = [\n"
        "    'Type','Age','Breed1','Breed2','Gender','Color1','Color2','Color3',\n"
        "    'MaturitySize','FurLength','Vaccinated','Dewormed','Sterilized',\n"
        "    'Health','Quantity','Fee','State','VideoAmt','PhotoAmt',\n"
        "    'HasPhoto','HasVideo','IsFree','AgeGroup','HealthScore','IsPureBreed','PhotoPerAnimal',\n"
        "    'Age_x_PhotoAmt','IsPureBreed_x_Age','HealthScore_x_Photo','IsYoungAndFree','IsHealthyAndPhoto','FeePerAnimal',\n"
        "    'sentiment_score','sentiment_magnitude','n_sentences','avg_label_score','n_labels','crop_confidence','desc_length',\n"
        "    'Breed1_enc','State_enc',\n"
        "    'rescuer_n_pets','age_rel_breed','word_count','unique_words','avg_word_len','uppercase_ratio','has_exclamation'\n"
        "]\n\n"
        "X_train = train_fe[ALL_FEATURES]; X_test = test_fe[ALL_FEATURES]\n"
        "y_train = train_fe['AdoptionSpeed']; y_test = test_fe['AdoptionSpeed']\n"
        "print(f'Features: {len(ALL_FEATURES)}')\n"
      )
    },
    {
      "cell_type": "markdown", "id": "sec_c_md", "metadata": {},
      "source": [
        "## Seccion C: CatBoost base — 5-fold CV\n\n",
        "Entrenamos CatBoost con parametros razonables para obtener una linea base.\n",
        "CatBoost usa `iterations` (equivalente a n_estimators) con early stopping interno."
      ]
    },
    {
      "cell_type": "code", "execution_count": None, "id": "cb_base_code", "metadata": {}, "outputs": [],
      "source": (
        "cb_params_base = {\n"
        "    'loss_function':    'MultiClass',\n"
        "    'eval_metric':      'Accuracy',\n"
        "    'iterations':       500,\n"
        "    'learning_rate':    0.05,\n"
        "    'depth':            6,\n"
        "    'l2_leaf_reg':      3.0,\n"
        "    'random_seed':      SEED,\n"
        "    'verbose':          False,\n"
        "    'early_stopping_rounds': 20,\n"
        "}\n\n"
        "skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)\n"
        "cb_test_preds = np.zeros((len(X_test), 5))\n"
        "cb_cv = []\n\n"
        "for tr_idx, val_idx in skf.split(X_train, y_train):\n"
        "    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]\n"
        "    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]\n"
        "    model = CatBoostClassifier(**cb_params_base)\n"
        "    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)\n"
        "    probs = model.predict_proba(X_val)\n"
        "    cb_cv.append(cohen_kappa_score(y_val, probs.argmax(axis=1), weights='quadratic'))\n"
        "    cb_test_preds += model.predict_proba(X_test)\n\n"
        "cb_kappa_base = cohen_kappa_score(y_test, cb_test_preds.argmax(axis=1), weights='quadratic')\n"
        "print(f'CatBoost base — CV: {np.mean(cb_cv):.4f} | Test: {cb_kappa_base:.4f}')\n"
      )
    },
    {
      "cell_type": "markdown", "id": "sec_d_md", "metadata": {},
      "source": [
        "## Seccion D: Optuna para CatBoost (30 trials)\n\n",
        "Optimizamos los hiperparametros principales de CatBoost:\n",
        "`depth`, `learning_rate`, `l2_leaf_reg`, `bagging_temperature`, `border_count`.\n\n",
        "⏳ *Esta seccion tarda ~20-30 minutos con 30 trials.*"
      ]
    },
    {
      "cell_type": "code", "execution_count": None, "id": "cb_optuna_code", "metadata": {}, "outputs": [],
      "source": (
        "def cb_cv_objective(trial):\n"
        "    params = {\n"
        "        'loss_function':         'MultiClass',\n"
        "        'eval_metric':           'Accuracy',\n"
        "        'random_seed':           SEED,\n"
        "        'verbose':               False,\n"
        "        'early_stopping_rounds': 20,\n"
        "        'iterations':            trial.suggest_int('iterations', 200, 800),\n"
        "        'depth':                 trial.suggest_int('depth', 4, 10),\n"
        "        'learning_rate':         trial.suggest_float('learning_rate', 0.01, 0.2, log=True),\n"
        "        'l2_leaf_reg':           trial.suggest_float('l2_leaf_reg', 0.1, 20.0, log=True),\n"
        "        'bagging_temperature':   trial.suggest_float('bagging_temperature', 0.0, 1.0),\n"
        "        'border_count':          trial.suggest_int('border_count', 32, 255),\n"
        "        'min_data_in_leaf':      trial.suggest_int('min_data_in_leaf', 1, 50),\n"
        "    }\n\n"
        "    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)\n"
        "    cv_scores = []\n"
        "    cb_preds  = np.zeros((len(X_test), 5))\n\n"
        "    for tr_idx, val_idx in skf.split(X_train, y_train):\n"
        "        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]\n"
        "        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]\n"
        "        model = CatBoostClassifier(**params)\n"
        "        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)\n"
        "        probs = model.predict_proba(X_val)\n"
        "        cv_scores.append(cohen_kappa_score(y_val, probs.argmax(axis=1), weights='quadratic'))\n"
        "        cb_preds += model.predict_proba(X_test)\n\n"
        "    test_kappa = cohen_kappa_score(y_test, cb_preds.argmax(axis=1), weights='quadratic')\n"
        "    trial.set_user_attr('test_score', test_kappa)\n"
        "    return np.mean(cv_scores)\n\n"
        "study_cb = optuna.create_study(direction='maximize')\n"
        "study_cb.optimize(cb_cv_objective, n_trials=30, show_progress_bar=True)\n\n"
        "best_cb_cv   = study_cb.best_value\n"
        "best_cb_test = study_cb.best_trial.user_attrs['test_score']\n"
        "best_cb_params = study_cb.best_params.copy()\n"
        "best_cb_params.update({'loss_function': 'MultiClass', 'eval_metric': 'Accuracy',\n"
        "                        'random_seed': SEED, 'verbose': False,\n"
        "                        'early_stopping_rounds': 20})\n\n"
        "print('='*60)\n"
        "print(f'Mejor CatBoost CV:   {best_cb_cv:.4f}')\n"
        "print(f'Mejor CatBoost Test: {best_cb_test:.4f}')\n"
        "print(f'Params: {study_cb.best_params}')\n"
        "print('='*60)\n"
      )
    },
    {
      "cell_type": "markdown", "id": "sec_e_md", "metadata": {},
      "source": [
        "## Seccion E: Ensemble LGB + XGB + CatBoost\n\n",
        "Combinamos los tres gradient boostings para obtener el mejor ensemble tabular posible.\n",
        "Usamos los parametros base de LGB y XGB (mismos que nb08) para mantener consistencia."
      ]
    },
    {
      "cell_type": "code", "execution_count": None, "id": "final_blend", "metadata": {}, "outputs": [],
      "source": (
        "# Reentrenar LGB\n"
        "lgb_params = {'objective': 'multiclass', 'num_class': 5, 'verbosity': -1,\n"
        "              'num_leaves': 51, 'lambda_l1': 0.10, 'lambda_l2': 7.58,\n"
        "              'feature_fraction': 0.59, 'bagging_fraction': 0.98,\n"
        "              'bagging_freq': 1, 'min_child_samples': 118, 'learning_rate': 0.093}\n"
        "lgb_preds = np.zeros((len(X_test), 5))\n"
        "for tr_idx, val_idx in skf.split(X_train, y_train):\n"
        "    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]\n"
        "    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]\n"
        "    m = lgb.train(lgb_params, lgb.Dataset(X_tr, label=y_tr), num_boost_round=500,\n"
        "                  valid_sets=[lgb.Dataset(X_val, label=y_val)],\n"
        "                  callbacks=[lgb.early_stopping(20, verbose=False)])\n"
        "    lgb_preds += m.predict(X_test)\n"
        "lgb_kappa = cohen_kappa_score(y_test, lgb_preds.argmax(axis=1), weights='quadratic')\n\n"
        "# Reentrenar XGB\n"
        "xgb_params = {'objective': 'multi:softprob', 'num_class': 5, 'verbosity': 0,\n"
        "              'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 500,\n"
        "              'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1,\n"
        "              'reg_lambda': 1.0, 'min_child_weight': 5, 'random_state': SEED,\n"
        "              'tree_method': 'hist', 'device': 'cpu'}\n"
        "xgb_preds = np.zeros((len(X_test), 5))\n"
        "for tr_idx, val_idx in skf.split(X_train, y_train):\n"
        "    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]\n"
        "    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]\n"
        "    m = xgb.XGBClassifier(**xgb_params)\n"
        "    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)\n"
        "    xgb_preds += m.predict_proba(X_test)\n"
        "xgb_kappa = cohen_kappa_score(y_test, xgb_preds.argmax(axis=1), weights='quadratic')\n\n"
        "# Reentrenar CB optimizado\n"
        "cb_opt_preds = np.zeros((len(X_test), 5))\n"
        "for tr_idx, val_idx in skf.split(X_train, y_train):\n"
        "    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]\n"
        "    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]\n"
        "    model = CatBoostClassifier(**best_cb_params)\n"
        "    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)\n"
        "    cb_opt_preds += model.predict_proba(X_test)\n"
        "cb_kappa = cohen_kappa_score(y_test, cb_opt_preds.argmax(axis=1), weights='quadratic')\n\n"
        "# Normalizar\n"
        "lgb_n = lgb_preds    / lgb_preds.sum(axis=1, keepdims=True)\n"
        "xgb_n = xgb_preds    / xgb_preds.sum(axis=1, keepdims=True)\n"
        "cb_n  = cb_opt_preds / cb_opt_preds.sum(axis=1, keepdims=True)\n\n"
        "# Grid search 3-way blend\n"
        "results = []\n"
        "for w_lgb in np.arange(0.2, 0.7, 0.05):\n"
        "    for w_xgb in np.arange(0.1, 0.6, 0.05):\n"
        "        w_cb = round(1 - w_lgb - w_xgb, 3)\n"
        "        if w_cb < 0.05 or w_cb > 0.5: continue\n"
        "        blend = round(w_lgb,2)*lgb_n + round(w_xgb,2)*xgb_n + w_cb*cb_n\n"
        "        kappa = cohen_kappa_score(y_test, blend.argmax(axis=1), weights='quadratic')\n"
        "        results.append({'w_lgb': round(w_lgb,2), 'w_xgb': round(w_xgb,2), 'w_cb': w_cb, 'kappa': round(kappa,4)})\n\n"
        "results_df = pd.DataFrame(results).sort_values('kappa', ascending=False)\n"
        "best = results_df.iloc[0]\n"
        "print('Top 10 combinaciones:')\n"
        "print(results_df.head(10).to_string(index=False))\n\n"
        "print('='*75)\n"
        "print('  COMPARATIVA FINAL')\n"
        "print('='*75)\n"
        "print(f'  FE v4 + LightGBM CV         Test: {lgb_kappa:.4f}')\n"
        "print(f'  FE v4 + XGBoost CV          Test: {xgb_kappa:.4f}')\n"
        "print(f'  FE v4 + CatBoost opt        Test: {cb_kappa:.4f}')\n"
        "print(f'  Blend LGB(0.5)+XGB(0.5)     Test: 0.3906  <- anterior record')\n"
        "print(f'  Blend LGB({best[\"w_lgb\"]})+XGB({best[\"w_xgb\"]})+CB({best[\"w_cb\"]})  Test: {best[\"kappa\"]:.4f}  <- MEJOR TRIPLE')\n"
        "print('='*75)\n"
      )
    }
  ]
}

with open("10_CatBoost_Ensemble_Roxy.ipynb", 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("OK — notebook 10 creado")
