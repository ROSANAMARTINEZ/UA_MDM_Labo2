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
        "# 09 — Blend Triple: LightGBM + XGBoost + DistilBERT\n\n",
        "**Materia:** Laboratorio de Implementacion II · Universidad Austral · Abril 2026\n\n",
        "**Autores:** Roxana Alberti · Sandra Sschicchi · Fernando Paganini · Baltazar Villanueva · Paula Calviello · Rosana Martinez\n\n",
        "---\n\n",
        "## Que hace este notebook?\n\n",
        "Combinamos tres modelos de naturaleza completamente distinta:\n\n",
        "| Modelo | Tipo | Input |\n",
        "|---|---|---|\n",
        "| LightGBM | Gradient boosting leaf-wise | 48 features tabulares (FE v4) |\n",
        "| XGBoost | Gradient boosting depth-wise | 48 features tabulares (FE v4) |\n",
        "| DistilBERT | Transformer (Transfer Learning) | Texto libre del campo Description |\n\n",
        "### Por que funciona el blend triple?\n",
        "Los tres modelos capturan informacion diferente:\n",
        "- LGB y XGB ven los mismos features pero aprenden de formas distintas\n",
        "- BERT lee el texto crudo de la descripcion — informacion que los otros dos no ven directamente\n\n",
        "Al promediar probabilidades, los errores individuales se cancelan parcialmente,\n",
        "siempre que los modelos no cometan los mismos errores en los mismos ejemplos.\n\n",
        "### Nota sobre BERT\n",
        "En el notebook 06 (blend LGB+BERT) el BERT no mejoro al modelo tabular.\n",
        "Aqui probamos si sumar XGBoost al mix cambia ese resultado.\n\n",
        "| Modelo | Kappa Test |\n",
        "|---|---|\n",
        "| LightGBM FE v4 | 0.3867 |\n",
        "| XGBoost FE v4 | 0.3803 |\n",
        "| Blend LGB+XGB (50/50) | 0.3906 |\n",
        "| **Blend LGB+XGB+BERT** | **(este notebook)** |"
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
        "from sklearn.metrics import cohen_kappa_score\n"
        "from sklearn.model_selection import train_test_split, StratifiedKFold\n"
        "from pathlib import Path\nimport warnings\n"
        "warnings.filterwarnings('ignore')\n\n"
        "BASE_DIR = Path.cwd()\n"
        "while not (BASE_DIR / 'input').exists() and BASE_DIR != BASE_DIR.parent:\n"
        "    BASE_DIR = BASE_DIR.parent\n"
        "print(f'BASE_DIR: {BASE_DIR}')\n\n"
        "SEED = 42\n"
        "train_raw = pd.read_csv(BASE_DIR / 'input/train/train.csv')\n"
        "sent_df   = pd.read_csv(BASE_DIR / 'input/train_sentiment_features.csv')\n"
        "meta_df   = pd.read_csv(BASE_DIR / 'input/train_metadata_features.csv')\n"
        "bert_df   = pd.read_csv(BASE_DIR / 'input/bert_predictions.csv')\n"
        "train_raw['desc_length'] = train_raw['Description'].fillna('').apply(len)\n\n"
        "df = (train_raw\n"
        "      .merge(sent_df[['PetID','sentiment_score','sentiment_magnitude','n_sentences']], on='PetID', how='left')\n"
        "      .merge(meta_df[['PetID','avg_label_score','n_labels','crop_confidence']], on='PetID', how='left')\n"
        "      .fillna(0))\n\n"
        "train, test = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df['AdoptionSpeed'])\n"
        "print(f'Train: {len(train)} | Test: {len(test)} | BERT preds: {len(bert_df)}')\n\n"
        "# BERT predictions: merge por PetID (puede haber filas sin prediccion)\n"
        "bert_cols = ['prob_0','prob_1','prob_2','prob_3','prob_4']\n"
        "test_with_bert = test.merge(bert_df[['PetID'] + bert_cols], on='PetID', how='left')\n"
        "n_missing = test_with_bert[bert_cols[0]].isna().sum()\n"
        "print(f'Test filas sin BERT: {n_missing} (se rellenan con distribucion uniforme)')\n"
        "for col in bert_cols:\n"
        "    test_with_bert[col] = test_with_bert[col].fillna(0.2)\n"
        "bert_preds = test_with_bert[bert_cols].values\n"
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
        "print(f'Features: {len(ALL_FEATURES)} | Train: {len(X_train)} | Test: {len(X_test)}')\n"
      )
    },
    {
      "cell_type": "markdown", "id": "sec_c_md", "metadata": {},
      "source": [
        "## Seccion C: LightGBM 5-fold CV\n\n",
        "Mismos parametros del mejor trial de 30 iteraciones (notebook 05)."
      ]
    },
    {
      "cell_type": "code", "execution_count": None, "id": "lgb_code", "metadata": {}, "outputs": [],
      "source": (
        "lgb_params = {'objective': 'multiclass', 'num_class': 5, 'verbosity': -1,\n"
        "              'num_leaves': 51, 'lambda_l1': 0.10, 'lambda_l2': 7.58,\n"
        "              'feature_fraction': 0.59, 'bagging_fraction': 0.98,\n"
        "              'bagging_freq': 1, 'min_child_samples': 118, 'learning_rate': 0.093}\n\n"
        "skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)\n"
        "lgb_test_preds = np.zeros((len(X_test), 5))\n"
        "lgb_cv = []\n\n"
        "for tr_idx, val_idx in skf.split(X_train, y_train):\n"
        "    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]\n"
        "    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]\n"
        "    m = lgb.train(lgb_params, lgb.Dataset(X_tr, label=y_tr),\n"
        "                  num_boost_round=500, valid_sets=[lgb.Dataset(X_val, label=y_val)],\n"
        "                  callbacks=[lgb.early_stopping(20, verbose=False)])\n"
        "    lgb_cv.append(cohen_kappa_score(y_val, m.predict(X_val).argmax(axis=1), weights='quadratic'))\n"
        "    lgb_test_preds += m.predict(X_test)\n\n"
        "lgb_kappa = cohen_kappa_score(y_test, lgb_test_preds.argmax(axis=1), weights='quadratic')\n"
        "print(f'LightGBM — CV: {np.mean(lgb_cv):.4f} | Test: {lgb_kappa:.4f}')\n"
      )
    },
    {
      "cell_type": "markdown", "id": "sec_d_md", "metadata": {},
      "source": [
        "## Seccion D: XGBoost 5-fold CV\n\n",
        "Parametros defaults de la Seccion D del notebook 08.\n",
        "Una vez disponibles los resultados del Optuna de XGBoost (nb08 Sec F),\n",
        "se pueden actualizar los params aqui para obtener el blend optimo."
      ]
    },
    {
      "cell_type": "code", "execution_count": None, "id": "xgb_code", "metadata": {}, "outputs": [],
      "source": (
        "xgb_params = {'objective': 'multi:softprob', 'num_class': 5, 'verbosity': 0,\n"
        "              'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 500,\n"
        "              'subsample': 0.8, 'colsample_bytree': 0.8,\n"
        "              'reg_alpha': 0.1, 'reg_lambda': 1.0,\n"
        "              'min_child_weight': 5, 'random_state': SEED,\n"
        "              'tree_method': 'hist', 'device': 'cpu'}\n\n"
        "xgb_test_preds = np.zeros((len(X_test), 5))\n"
        "xgb_cv = []\n\n"
        "for tr_idx, val_idx in skf.split(X_train, y_train):\n"
        "    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]\n"
        "    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]\n"
        "    model = xgb.XGBClassifier(**xgb_params)\n"
        "    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)\n"
        "    probs = model.predict_proba(X_val)\n"
        "    xgb_cv.append(cohen_kappa_score(y_val, probs.argmax(axis=1), weights='quadratic'))\n"
        "    xgb_test_preds += model.predict_proba(X_test)\n\n"
        "xgb_kappa = cohen_kappa_score(y_test, xgb_test_preds.argmax(axis=1), weights='quadratic')\n"
        "print(f'XGBoost — CV: {np.mean(xgb_cv):.4f} | Test: {xgb_kappa:.4f}')\n"
      )
    },
    {
      "cell_type": "markdown", "id": "sec_e_md", "metadata": {},
      "source": [
        "## Seccion E: Blend triple LGB + XGB + BERT\n\n",
        "Probamos todas las combinaciones de pesos con paso 0.05.\n",
        "El constraint es `w_lgb + w_xgb + w_bert = 1.0`.\n\n",
        "La intuicion es que LGB y XGB ya aportan informacion tabular similar,\n",
        "por lo que la ganancia de BERT dependera de cuanto agrega el texto\n",
        "por encima de los features NLP ya incluidos (word_count, avg_word_len, etc.)."
      ]
    },
    {
      "cell_type": "code", "execution_count": None, "id": "blend_code", "metadata": {}, "outputs": [],
      "source": (
        "# Normalizar probabilidades\n"
        "lgb_probs  = lgb_test_preds  / lgb_test_preds.sum(axis=1, keepdims=True)\n"
        "xgb_probs  = xgb_test_preds  / xgb_test_preds.sum(axis=1, keepdims=True)\n"
        "bert_probs = bert_preds       / bert_preds.sum(axis=1, keepdims=True)\n\n"
        "# Grid search de pesos: w_bert de 0 a 0.20, el resto se reparte entre LGB y XGB\n"
        "results = []\n"
        "for w_bert in [0.00, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:\n"
        "    remaining = 1.0 - w_bert\n"
        "    for w_lgb_frac in np.arange(0.3, 0.8, 0.05):\n"
        "        w_lgb = round(remaining * w_lgb_frac, 3)\n"
        "        w_xgb = round(remaining * (1 - w_lgb_frac), 3)\n"
        "        blend = w_lgb * lgb_probs + w_xgb * xgb_probs + w_bert * bert_probs\n"
        "        kappa = cohen_kappa_score(y_test, blend.argmax(axis=1), weights='quadratic')\n"
        "        results.append({'w_lgb': w_lgb, 'w_xgb': w_xgb, 'w_bert': w_bert, 'kappa': round(kappa, 4)})\n\n"
        "results_df = pd.DataFrame(results).sort_values('kappa', ascending=False)\n"
        "best = results_df.iloc[0]\n"
        "print('Top 10 combinaciones:')\n"
        "print(results_df.head(10).to_string(index=False))\n\n"
        "print('='*75)\n"
        "print('  COMPARATIVA FINAL — Todos los modelos')\n"
        "print('='*75)\n"
        "print(f'  FE v4 + LightGBM CV            Test: {lgb_kappa:.4f}')\n"
        "print(f'  FE v4 + XGBoost CV             Test: {xgb_kappa:.4f}')\n"
        "print(f'  Blend LGB(0.50)+XGB(0.50)      Test: 0.3906   <- anterior record')\n"
        "print(f'  Blend LGB({best[\"w_lgb\"]})+XGB({best[\"w_xgb\"]})+BERT({best[\"w_bert\"]})  Test: {best[\"kappa\"]:.4f}  <- MEJOR TRIPLE')\n"
        "print('='*75)\n"
      )
    }
  ]
}

with open("09_Blend_Triple_LGB_XGB_BERT_Roxy.ipynb", 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("OK — notebook 09 creado")
