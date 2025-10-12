"""Script for general utilities."""

import yaml
import pandas as pd
import numpy as np

def fetch_data(path_per_variable: dict[str, str]) -> pd.DataFrame:
    """Fetch and transform the data.
    
    Args:
        path_per_variable: mapping from var name to path
    
    Returns:
        pd.DataFrame: the preprocessed dataframe with log-transformed sales and price
    """
    dfs = []
    for new_name, filepath in path_per_variable.items():
        df = pd.read_excel(filepath)[['brand62']].rename(columns={'brand62': new_name})
        
        if new_name in ['sales', 'price']:
            df[new_name] = np.log(df[new_name])
        dfs.append(df)
    
    return pd.concat(dfs, axis=1)

def load_config(path: str) -> dict[str, str | int | float]:
    """Load and parse a YAML configuration file.

    Args:
        path: the path to the YAML configuration file

    Returns:
        Dict[str, Any]: the config dictionary
    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config