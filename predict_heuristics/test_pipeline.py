import pandas as pd
import numpy as np
import h2o
from h2o.estimators import H2OGradientBoostingEstimator
from h2o.estimators import H2OXGBoostEstimator
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, auc

def convert_to_h2o_frame(data: pd.DataFrame) -> h2o.H2OFrame:
    """Converte um pd.DataFrame em um h2o.H2OFrame"""
    print('Performing conversion from pd.DataFrame to h2o.H2OFrame...')
    # Colunas de datas
    cols_to_date = [col for col in data.dtypes[data.dtypes == 'datetime64[ns]'].index]
    # Colunas a converter para categórica
    cols_to_factor = [col for col in data.dtypes[(data.dtypes == 'object') | (data.dtypes == 'category')].index]
    # Colunas a converter a numeric
    cols_numerics = [col for col in data.dtypes[(data.dtypes != 'object') & (data.dtypes != 'category') & (data.dtypes != 'datetime64[ns]')].index]
    # Adicionar os tipos de colunas ao dicionário col_dtypes
    col_dtypes = {}
    for col in cols_to_date:
        col_dtypes[col] = 'time'
    for col in cols_to_factor:
        col_dtypes[col] = 'enum'
    for col in cols_numerics:
        col_dtypes[col] = 'numeric'
    # Converte a h2o
    data = h2o.H2OFrame(data, column_types=col_dtypes, na_strings=['NA', 'none', 'None', 'nan', 'NaN', '<NA>'])
    print('Successful conversion from pd.DataFrame to h2o.H2OFrame.')
    return data


def create_dataset(dataset_features: pd.DataFrame, dataset_results: pd.DataFrame) -> pd.DataFrame:
    """Merge resultados e features dataset"""
    dataset_results = dataset_results[['Instancias'] + [col for col in dataset_results.columns if col.startswith('Obj_') or col.startswith('Time_')]]
    dataset_results = dataset_results[dataset_results['Obj_RF_T_0'] != np.inf]  # Remove infactíveis
    data = dataset_features.merge(dataset_results, left_on='instance', right_on='Instancias', how='inner').drop(columns=['Instancias', 'instance'])
    return data


def create_target(data: pd.DataFrame, target_type: str, is_binary: bool=False) -> pd.DataFrame:
    """Cria target com base nos dados de resolução do modelo"""
    objective_columns = [col for col in data.columns if col.startswith('Obj_')]
    time_columns = [col for col in data.columns if col.startswith('Time_')]
    # Seleciona o método com menor função objetivo
    if target_type == 'BEST':
        data['BEST'] = data[objective_columns].idxmin(axis=1)
        data['BEST'] = data['BEST'].str.replace('Obj_', '')
        if is_binary:
            data['BEST'] = data['BEST'].apply(lambda x: 'Gurobi' if x == 'RF_T_0' else 'RF')
    # Seleciona o método com menor função objetivo considerando penalização por tempo
    elif target_type == 'BEST_TIME':
        for obj_col, time_col in zip(objective_columns, time_columns):
            data[f'Adjusted_{obj_col}'] = (data[obj_col] * np.maximum(np.log(data[time_col]) / np.log(600), 1))
        adjusted_columns = [col for col in data.columns if col.startswith('Adjusted_')]
        data['BEST_TIME'] = data[adjusted_columns].idxmin(axis=1)
        data['BEST_TIME'] = data['BEST_TIME'].str.replace('Adjusted_Obj_', '')
        data = data.drop(columns=adjusted_columns)
        if is_binary:
            data['BEST_TIME'] = data['BEST_TIME'].apply(lambda x: 'Gurobi' if x == 'RF_T_0' else 'RF')
    # Remover colunas auxiliares
    data = data.drop(columns=objective_columns + time_columns)
    # Shuffle
    data = data.sample(frac=1)
    return data


if __name__ == '__main__':
    # PARAMETERS
    TARGET_TYPE = 'BEST'
    BINARY_CLASSIFICATION = False

    # Init H20 cluster
    h2o.init(port=2112)

    # Leitura dos dataset base
    df_resultados = pd.read_csv('resultados_instancias_tcc.csv')
    df_features = pd.read_csv('multi_plant_instance_features.csv')
    # Criação do dataset com features e resultados (função objetivo + tempo de resolução)
    dataset = create_dataset(dataset_features=df_features, dataset_results=df_resultados)
    # Seleciona método de target (testar diferentes tipos)
    dataset = create_target(dataset, target_type=TARGET_TYPE, is_binary=BINARY_CLASSIFICATION)
    # Converte para H2O Frame
    hf = convert_to_h2o_frame(dataset)
    # Train test split
    hf_train, hf_test = hf.split_frame(ratios=[.8], seed=2112)
    # Setup treino
    target = TARGET_TYPE
    predictors = [c for c in hf_train.columns if c != target]
    # Modelo
    gbm_model = H2OGradientBoostingEstimator(
    nfolds=20,
    keep_cross_validation_predictions=True,
    seed=2112,
    stopping_rounds=10,
    stopping_metric="AUTO",
    stopping_tolerance=0.001,
    balance_classes=False
    )
    # Treino
    model = gbm_model
    model.train(x=predictors, y=target, training_frame=hf_train)
    # Previsões
    actual = hf_train[target].as_data_frame(use_multi_thread=True)[target]
    predictions = model.cross_validation_holdout_predictions()
    predict = predictions[0].as_data_frame(use_multi_thread=True)['predict']
    # Confusion matrix
    cm = confusion_matrix(actual, predict)
    print("Confusion Matrix:\n", cm)
    # Métricas
        # Precisão
    precision = cm.trace() / cm.sum()
    print(precision)

    # Shutdown H2O
    # h2o.cluster().shutdown()
