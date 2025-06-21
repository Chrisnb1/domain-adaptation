import yaml
import os

class Config:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config_data = self.load_config()

    def load_config(self):
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"El archivo {self.config_file} no se encuentra.")

        with open(self.config_file, 'r', encoding='utf-8') as file_config:
            return yaml.safe_load(file_config)
        
    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config_data

        for k in keys:
            value = value.get(k, default)
            if value is default:
                break

        return value
        