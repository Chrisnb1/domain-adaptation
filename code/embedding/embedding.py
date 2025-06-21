import torch
from sentence_transformers import SentenceTransformer
from utils.config import Config
from pathlib import Path

class Embedding:
    def __init__(self, model_path: str = None):
        if model_path:
            self.model_id = model_path
            self.model_name = Path(model_path).name
        else:
            config_path = Path(__file__).parent.parent / 'config.yaml'
            self.config = Config(config_path)
            self.model_id = self.config.get('embedding_model.id')
            self.model_name = self.config.get('embedding_model.name')
        self.model = SentenceTransformer(self.model_id, device="cuda" if torch.cuda.is_available() else "cpu")