import os
import yaml
import logging

def load_config(config_path):
    """
    Load configuration from YAML file
    
    Parameters:
    -----------
    config_path : str
        Path to the YAML configuration file
        
    Returns:
    --------
    dict
        Configuration dictionary
    """
    try:
        if not os.path.exists(config_path):
            logging.warning(f"Config file {config_path} not found. Using default configuration.")

                
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            logging.info(f"Configuration loaded from {config_path}")
            return config
    except Exception as e:
        logging.error(f"Error loading config: {str(e)}. Using default configuration.")
        