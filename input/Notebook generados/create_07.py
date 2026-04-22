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
        "# 07 — Selección de Features con SHAP + Boruta\n\n",
        "**Materia:** Laboratorio de Implementación II · Universidad Austral · Abril 2026\n\n",
        "**Autores:** Roxana Alberti · Sandra Sschicchi · Fernando Paganini · Baltazar Villanueva · Paula Calviello · Rosana Martinez\n\n",
        "---\n\n",
        "**Objetivo:** identificar qué features del FE v4 (48 features) son realmente útiles usando:\n",
        "- **Boruta**: compara cada feature contra shadow features aleatorias — Acepta, Rechaza o deja Tentativa\n",
        "- **SHAP**: mide la contribución real de cada feature a las predicciones del modelo\n\n",
        "El resultado es un subconjunto limpio de features verificadas estadísticamente."
      ]
    },
    {
      "cell_type": "markdown", "id": "sec_a_md", "metadata": {},
      "source": ["## Sección A: Imports y datos"]
    },
    {
      "cell_type": "code", "execution_count": None, "id": "setup", "metadata": {}, "outputs": [],
      "source": (
        "import pandas as pd\n"
        "import numpy as np\n"
        "import lightgbm as lgb\n"
        "import shap\n"
        "from boruta import BorutaPy\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "from sklearn.metrics import cohen_kappa_score\n"
        "from sklearn.model_selection import train_test_split, StratifiedKFold\n"
        "from pathlib import Path\n"
        "import warnings, matplotlib.pyplot as plt\n"
        "warnings.filterwarnings('ignore')\n"
        "\n"
        "BASE_DIR = Path.cwd()\n"
        "while not (BASE_DIR / 'input').exists() and BASE_DIR != BASE_DIR.parent:\n"
        "    BASE_DIR = BASE_DIR.parent\n"
        "print(f'BASE_DIR: {BASE_DIR}')\n"
        "\n"
        "SEED = 42\n"
        "train_raw = pd.read_csv(BASE_DIR / 'input/train/train.csv')\n"
        "sent_df   = pd.read_csv(BASE_DIR / 'input/train_sentiment_features.csv')\n"
        "meta_df   = pd.read_csv(BASE_DIR / 'input/train_metadata_features.csv')\n"
        "train_raw['desc_length'] = train_raw['Description'].fillna('').apply(len)\n"
        "\n"
        "df = (train_raw\n"
        "      .merge(sent_df[['PetID','sentiment_score','sentiment_magnitude','n_sentences']], on='PetID', how='left')\n"
        "      .merge(meta_df[['PetID','avg_label_score','n_labels','crop_confidence']], on='PetID', how='left')\n"
        "      .fillna(0))\n"
        "\n"
        "train, test = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df['AdoptionSpeed'])\n"
        "print(f'Train: {len(train)} | Test: {len(test)}')\n"
      )
    },
    {
      "cell_type": "markdown", "id": "sec_b_md", "metadata": {},
      "source": ["## Sección B: Feature Engineering v4"]
    },
    {
      "cell_type": "code", "execution_count": None, "id": "fe_code", "metadata": {}, "outputs": [],
      "source": (
        "def target_encode(train_df, test_df, col, target='AdoptionSpeed', smoothing=10):\n"
        "    global_mean = train_df[target].mean()\n"
        "    stats = train_df.groupby(col)[target].agg(['mean', 'count'])\n"
        "    stats['encoded'] = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)\n"
        "    return train_df[col].map(stats['encoded']).fillna(global_mean), test_df[col].map(stats['encoded']).fillna(global_mean)\n"
        "\n"
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
        "    return df_\n"
        "\n"
        "def nlp_feats(df_):\n"
        "    desc = df_['Description'].apply(lambda x: '' if (x == 0 or str(x).strip() == '') else str(x))\n"
        "    df_['word_count']      = desc.apply(lambda x: len(x.split()))\n"
        "    df_['unique_words']    = desc.apply(lambda x: len(set(x.lower().split())))\n"
        "    df_['avg_word_len']    = desc.apply(lambda x: round(sum(len(w) for w in x.split()) / max(len(x.split()),1), 2))\n"
        "    df_['uppercase_ratio'] = desc.apply(lambda x: round(sum(c.isupper() for c in x) / max(len(x),1), 4))\n"
        "    df_['has_exclamation'] = desc.apply(lambda x: int('!' in x))\n"
        "    return df_\n"
        "\n"
        "train = train.copy(); test = test.copy()\n"
        "train['Breed1_enc'], test['Breed1_enc'] = target_encode(train, test, 'Breed1')\n"
        "train['State_enc'],  test['State_enc']  = target_encode(train, test, 'State')\n"
        "\n"
        "rescuer_count = train.groupby('RescuerID').size().rename('rescuer_n_pets')\n"
        "train['rescuer_n_pets'] = train['RescuerID'].map(rescuer_count).fillna(1)\n"
        "test['rescuer_n_pets']  = test['RescuerID'].map(rescuer_count).fillna(1)\n"
        "\n"
        "age_med_map = train.groupby(['Breed1','Type'])['Age'].median().to_dict()\n"
        "global_age  = train['Age'].median()\n"
        "for df_ in [train, test]:\n"
        "    df_['age_median_bt'] = [age_med_map.get((b,t), global_age) for b,t in zip(df_['Breed1'], df_['Type'])]\n"
        "    df_['age_rel_breed'] = df_['Age'] / (df_['age_median_bt'] + 1)\n"
        "\n"
        "train = nlp_feats(train); test = nlp_feats(test)\n"
        "train_fe = add_features_v4(train); test_fe = add_features_v4(test)\n"
        "\n"
        "ALL_FEATURES = [\n"
        "    'Type','Age','Breed1','Breed2','Gender','Color1','Color2','Color3',\n"
        "    'MaturitySize','FurLength','Vaccinated','Dewormed','Sterilized',\n"
        "    'Health','Quantity','Fee','State','VideoAmt','PhotoAmt',\n"
        "    'HasPhoto','HasVideo','IsFree','AgeGroup','HealthScore','IsPureBreed','PhotoPerAnimal',\n"
        "    'Age_x_PhotoAmt','IsPureBreed_x_Age','HealthScore_x_Photo','IsYoungAndFree','IsHealthyAndPhoto','FeePerAnimal',\n"
        "    'sentiment_score','sentiment_magnitude','n_sentences','avg_label_score','n_labels','crop_confidence','desc_length',\n"
        "    'Breed1_enc','State_enc',\n"
        "    'rescuer_n_pets','age_rel_breed','word_count','unique_words','avg_word_len','uppercase_ratio','has_exclamation'\n"
        "]\n"
        "\n"
        "X_train = train_fe[ALL_FEATURES].values\n"
        "X_test  = test_fe[ALL_FEATURES].values\n"
        "y_train = train_fe['AdoptionSpeed'].values\n"
        "y_test  = test_fe['AdoptionSpeed'].values\n"
        "print(f'Features totales: {len(ALL_FEATURES)}')\n"
      )
    },
    {
      "cell_type": "markdown", "id": "sec_c_md", "metadata": {},
      "source": [
        "## Sección C: Boruta — selección estadística\n\n",
        "Boruta crea shadow features (copias aleatorizadas) y compara la importancia real de cada feature\n",
        "contra el máximo de sus sombras en cada iteración. Una feature es:\n",
        "- **Aceptada**: supera consistentemente a sus sombras\n",
        "- **Rechazada**: queda sistemáticamente por debajo\n",
        "- **Tentativa**: resultado incierto después de max_iter iteraciones\n\n",
        "Usamos RandomForest como base (Boruta requiere un modelo con feature_importances_)."
      ]
    },
    {
      "cell_type": "code", "execution_count": None, "id": "boruta_code", "metadata": {}, "outputs": [],
      "source": (
        "rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)\n"
        "selector = BorutaPy(rf, n_estimators='auto', verbose=1, random_state=SEED, max_iter=50)\n"
        "selector.fit(X_train, y_train)\n"
        "\n"
        "results_df = pd.DataFrame({\n"
        "    'Feature':  ALL_FEATURES,\n"
        "    'Ranking':  selector.ranking_,\n"
        "    'Decision': ['Aceptada' if s else ('Tentativa' if t else 'Rechazada')\n"
        "                 for s, t in zip(selector.support_, selector.support_weak_)]\n"
        "}).sort_values('Ranking')\n"
        "\n"
        "print('=== BORUTA — Resultados ===')\n"
        "print(f\"Aceptadas  ({(results_df.Decision=='Aceptada').sum()}): {list(results_df[results_df.Decision=='Aceptada']['Feature'])}\")\n"
        "print(f\"Tentativas ({(results_df.Decision=='Tentativa').sum()}): {list(results_df[results_df.Decision=='Tentativa']['Feature'])}\")\n"
        "print(f\"Rechazadas ({(results_df.Decision=='Rechazada').sum()}): {list(results_df[results_df.Decision=='Rechazada']['Feature'])}\")\n"
        "print()\n"
        "print(results_df.to_string(index=False))\n"
      )
    },
    {
      "cell_type": "markdown", "id": "sec_d_md", "metadata": {},
      "source": [
        "## Sección D: SHAP — importancia real por feature\n\n",
        "SHAP calcula cuánto contribuye cada feature a cada predicción individual.\n",
        "A diferencia del feature importance de LightGBM (frecuencia de splits),\n",
        "SHAP mide el **impacto real en la predicción** usando teoría de juegos cooperativos."
      ]
    },
    {
      "cell_type": "code", "execution_count": None, "id": "shap_code", "metadata": {}, "outputs": [],
      "source": (
        "lgb_model = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,\n"
        "                                objective='multiclass', num_class=5,\n"
        "                                random_state=SEED, verbosity=-1)\n"
        "lgb_model.fit(X_train, y_train)\n"
        "\n"
        "explainer   = shap.TreeExplainer(lgb_model)\n"
        "shap_values = explainer.shap_values(X_test[:500])\n"
        "\n"
        "shap_abs = np.abs(np.array(shap_values)).mean(axis=0).mean(axis=0)\n"
        "shap_df  = pd.DataFrame({'Feature': ALL_FEATURES, 'SHAP_importance': shap_abs})\n"
        "shap_df  = shap_df.sort_values('SHAP_importance', ascending=False)\n"
        "\n"
        "print('=== SHAP — Top 20 features ===')\n"
        "print(shap_df.head(20).to_string(index=False))\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(10, 8))\n"
        "top20 = shap_df.head(20)\n"
        "ax.barh(top20['Feature'][::-1], top20['SHAP_importance'][::-1], color='#6366f1')\n"
        "ax.set_title('SHAP Feature Importance — Top 20', fontsize=13)\n"
        "ax.set_xlabel('Mean |SHAP value|')\n"
        "plt.tight_layout()\n"
        "plt.savefig('shap_importance.png', dpi=100, bbox_inches='tight')\n"
        "plt.show()\n"
        "print('Gráfico guardado: shap_importance.png')\n"
      )
    },
    {
      "cell_type": "markdown", "id": "sec_e_md", "metadata": {},
      "source": [
        "## Sección E: Modelo final con features seleccionadas\n\n",
        "Entrenamos con las features Aceptadas + Tentativas por Boruta\n",
        "y comparamos el kappa contra el modelo de 48 features."
      ]
    },
    {
      "cell_type": "code", "execution_count": None, "id": "final_model", "metadata": {}, "outputs": [],
      "source": (
        "accepted  = list(results_df[results_df.Decision == 'Aceptada']['Feature'])\n"
        "tentative = list(results_df[results_df.Decision == 'Tentativa']['Feature'])\n"
        "selected  = accepted + tentative\n"
        "\n"
        "X_tr_sel = train_fe[selected].values\n"
        "X_te_sel = test_fe[selected].values\n"
        "\n"
        "params = {'objective': 'multiclass', 'num_class': 5, 'verbosity': -1,\n"
        "          'num_leaves': 51, 'lambda_l1': 0.10, 'lambda_l2': 7.58,\n"
        "          'feature_fraction': 0.59, 'bagging_fraction': 0.98,\n"
        "          'bagging_freq': 1, 'min_child_samples': 118, 'learning_rate': 0.093}\n"
        "\n"
        "skf      = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)\n"
        "ensemble = np.zeros((len(X_te_sel), 5))\n"
        "cv_scores = []\n"
        "\n"
        "for tr_idx, val_idx in skf.split(X_tr_sel, y_train):\n"
        "    X_tr, X_val = X_tr_sel[tr_idx], X_tr_sel[val_idx]\n"
        "    y_tr, y_val = y_train[tr_idx], y_train[val_idx]\n"
        "    lgb_tr  = lgb.Dataset(X_tr, label=y_tr)\n"
        "    lgb_val = lgb.Dataset(X_val, label=y_val)\n"
        "    m = lgb.train(params, lgb_tr, num_boost_round=500, valid_sets=[lgb_val],\n"
        "                  callbacks=[lgb.early_stopping(20, verbose=False)])\n"
        "    cv_scores.append(cohen_kappa_score(y_val, m.predict(X_val).argmax(axis=1), weights='quadratic'))\n"
        "    ensemble += m.predict(X_te_sel)\n"
        "\n"
        "kappa_sel = cohen_kappa_score(y_test, ensemble.argmax(axis=1), weights='quadratic')\n"
        "\n"
        "print('='*60)\n"
        "print(f'Features seleccionadas: {len(selected)} / {len(ALL_FEATURES)}')\n"
        "print(f'  Aceptadas: {len(accepted)} | Tentativas: {len(tentative)}')\n"
        "print(f'CV Kappa (5-fold):  {np.mean(cv_scores):.4f}')\n"
        "print(f'Test Kappa:         {kappa_sel:.4f}')\n"
        "print(f'Test Kappa FE v4:   0.3867')\n"
        "print(f'Diferencia:         {kappa_sel - 0.3867:+.4f}')\n"
        "print('='*60)\n"
      )
    }
  ]
}

with open("07_BorutaSHAP_Roxy.ipynb", 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("OK — notebook creado")
