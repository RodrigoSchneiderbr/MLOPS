import logging
import os
import yaml

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


logger = logging.getLogger("src.data_preprocessing.preprocess_data")


def load_data() -> pd.DataFrame:
    """Load the raw data from disk.

    Returns:
        pd.DataFrame: Raw input data
    """
    input_path = "C:\\Users\\schne\\OneDrive\\Desktop\\udemy\\MLOPS\\projeto_MLOPS\\mlops_project\\data\\raw\\raw.csv"
    logger.info(f"Loading raw data from {input_path}")
    data = pd.read_csv(input_path)
    return data


def load_params() -> dict[str, float | int]:
    """Load preprocessing parameters from params.yaml.

    Returns:
        dict[str, Any]: dictionary containing preprocessing parameters.
    """
    # Monta o caminho absoluto correto usando o ROOT_DIR que definimos no topo do script
    params_path = "C:\\Users\\schne\\OneDrive\\Desktop\\udemy\\MLOPS\\projeto_MLOPS\\mlops_project\\params.yaml"
    
    logger.info(f"Loading parameters from {params_path}")
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    return params["preprocess_data"]


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets using parameters from params.yaml.

    Args:
        data (pd.DataFrame): Input dataset

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Train and test datasets
    """
    params = load_params()
    logger.info("Splitting data into train and test sets...")
    train_data, test_data = train_test_split(
        data, test_size=params["test_size"], random_state=params["random_seed"]
    )
    return train_data, test_data


def preprocess_data(
    train_data: pd.DataFrame, test_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, SimpleImputer]:
    """Perform preprocessing steps on train and test sets.

    Args:
        train_data (pd.DataFrame): Training dataset
        test_data (pd.DataFrame): Test dataset

    Returns:
        Tuple containing:
            pd.DataFrame: Processed training data
            pd.DataFrame: Processed test data
            SimpleImputer: Fitted imputer
    """
    logger.info("Preprocessing data...")
    
    # Separate target column
    train_target = train_data['target']
    test_target = test_data['target']
    train_features = train_data.drop('target', axis=1)
    test_features = test_data.drop('target', axis=1)
    
    # Apply imputation
    imputer = SimpleImputer(strategy="mean")
    train_features_processed = pd.DataFrame(
        imputer.fit_transform(train_features), columns=train_features.columns
    )
    test_features_processed = pd.DataFrame(
        imputer.transform(test_features), columns=test_features.columns
    )
    
    # Merge target back with processed features
    train_processed = train_features_processed.assign(target=train_target.tolist())
    test_processed = test_features_processed.assign(target=test_target.tolist())
    
    return train_processed, test_processed, imputer


def save_artifacts(
    train_data: pd.DataFrame, test_data: pd.DataFrame, imputer: SimpleImputer
) -> None:
    """Save processed data and preprocessing artifacts."""
    
    # 1. Caminho absoluto da pasta de dados processados
    data_dir = "C:\\Users\\schne\\OneDrive\\Desktop\\udemy\\MLOPS\\projeto_MLOPS\\mlops_project\\data\\preprocessed"
    
    # GARANTIA: Cria a pasta 'preprocessed' automaticamente se ela não existir
    os.makedirs(data_dir, exist_ok=True)
    
    logger.info(f"Saving processed data to {data_dir}")
    train_path = os.path.join(data_dir, "train_preprocessed.csv")
    test_path = os.path.join(data_dir, "test_preprocessed.csv")

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    # 2. Caminho absoluto da pasta de artefatos (CORRIGIDO AQUI)
    artifacts_dir = "C:\\Users\\schne\\OneDrive\\Desktop\\udemy\\MLOPS\\projeto_MLOPS\\mlops_project\\artifacts"
    
    # GARANTIA: Cria a pasta 'artifacts' automaticamente se ela não existir
    os.makedirs(artifacts_dir, exist_ok=True)
    
    imputer_path = os.path.join(artifacts_dir, "[features]_mean_imputer.joblib")
    logger.info(f"Saving imputer to {imputer_path}")
    joblib.dump(imputer, imputer_path)


def main() -> None:
    """Main function to orchestrate the preprocessing pipeline."""
    raw_data = load_data()
    train_data, test_data = split_data(raw_data)
    train_processed, test_processed, imputer = preprocess_data(train_data, test_data)
    save_artifacts(train_processed, test_processed, imputer)
    logger.info("Data preprocessing completed")


if __name__ == "__main__":
    main()
