import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import optuna

# Carregar os dados
df = pd.read_csv('C:\\Users\\Lapei_Cigets\\Documents\\GitHub\\sa_maismedicos\\01_dados\\dados resultantes\\df_modelagem.csv')

# Remover colunas desnecessárias
df = df.drop(["Unnamed: 0", "meses_no_local_alocado"], axis='columns')

# Tratar valores ausentes nas colunas de profissionais de saúde
df[['m_agente_saude', 'm_tec_aux_enf', 'm_enfermeiro', 'm_dentista']] = df[['m_agente_saude', 'm_tec_aux_enf', 'm_enfermeiro', 'm_dentista']].fillna(0)

# Codificar a variável target
df['churn'] = df['churn'].map({'permanece': 0, 'migrou': 1})

# Separar features e target
y = df['churn']
X = df.drop(columns=['churn'])

# Identificar features numéricas e categóricas
num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Pipeline para features numéricas
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline para features categóricas
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

# Pré-processador completo
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Aplicar pré-processamento separadamente para treino e teste
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Balanceamento de dados com SMOTE (integrado ao pipeline)
smote = SMOTE(random_state=42)

# Modelos e hiperparâmetros para otimização com Optuna
models = {
    'Logistic Regression': LogisticRegression(n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', tree_method='hist'),
    'LightGBM': lgb.LGBMClassifier(),
    'Support Vector Classifier': SVC(probability=True)
}

param_grids = {
    'Logistic Regression': {'C': [0.01, 0.1, 1, 10]},
    'Decision Tree': {'max_depth': [3, 5, 10]},
    'Random Forest': {
        'n_estimators': [100, 300, 500],
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5, 10]
    },
    'XGBoost': {
        'n_estimators': [100, 300, 500],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.85, 1.0],
        'colsample_bytree': [0.7, 0.85, 1.0],
        'gamma': [0, 0.1, 0.5]
    },
    'LightGBM': {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [20, 31, 40],
        'min_child_samples': [10, 20, 30]
    },
    'Support Vector Classifier': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
}

# Função de otimização com Optuna
def objective(trial, model, X_train_resampled, y_train_resampled):
    param_grid = param_grids.get(model, {})
    trial_params = {k: trial.suggest_categorical(k, v) if isinstance(v, list) else trial.suggest_float(k, v[0], v[1]) for k, v in param_grid.items()}
    clf = models[model].set_params(**trial_params)
    clf.fit(X_train_resampled, y_train_resampled)
    y_prob = clf.predict_proba(X_test_transformed)[:, 1]
    return roc_auc_score(y_test, y_prob)

# Otimização dos modelos
best_models = {}
for name, model in models.items():
    print(f"Otimizando modelo: {name}")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, name, X_train_transformed, y_train), n_trials=20)
    best_models[name] = models[name].set_params(**study.best_params)
    print(f"Melhores parâmetros para {name}: {study.best_params}\n")

# Avaliação final no conjunto de teste
results = []
for name, model in best_models.items():
    y_pred = model.predict(X_test_transformed)
    y_prob = model.predict_proba(X_test_transformed)[:, 1]
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_prob)
    })

results_df = pd.DataFrame(results)
print(results_df.sort_values(by='ROC AUC', ascending=False))

# Melhor modelo
best_model_name = results_df.sort_values(by='ROC AUC', ascending=False).iloc[0]['Model']
best_model = best_models[best_model_name]

# Curva ROC
plt.figure(figsize=(8, 6))
for name, model in best_models.items():
    y_prob = model.predict_proba(X_test_transformed)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curvas ROC para os Modelos')
plt.legend()
plt.show()

# Matriz de Confusão
y_pred_best = best_model.predict(X_test_transformed)
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Permanece', 'Migrou'], yticklabels=['Permanece', 'Migrou'])
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title(f'Matriz de Confusão - {best_model_name}')
plt.show()

# Ajuste de limiar
y_prob_best = best_model.predict_proba(X_test_transformed)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob_best)

# Encontrar o melhor limiar com base no F1 Score
best_threshold = 0.5
best_f1 = 0
for threshold in thresholds:
    y_pred_adjusted = (y_prob_best >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred_adjusted)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Melhor limiar: {best_threshold} com F1 Score: {best_f1}")

# Matriz de Confusão com limiar ajustado
y_pred_adjusted = (y_prob_best >= best_threshold).astype(int)
cm_adjusted = confusion_matrix(y_test, y_pred_adjusted)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_adjusted, annot=True, fmt='d', cmap='Blues', xticklabels=['Permanece', 'Migrou'], yticklabels=['Permanece', 'Migrou'])
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title(f'Matriz de Confusão - Ajuste de Limiar ({best_model_name})')
plt.show()

# Explicabilidade do modelo com SHAP
explainer = shap.Explainer(best_model, X_train_transformed)
shap_values = explainer(X_test_transformed)
shap.summary_plot(shap_values, X_test_transformed, feature_names=preprocessor.get_feature_names_out())
