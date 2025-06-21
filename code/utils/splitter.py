import yaml
import json
import time
import random
from pathlib import Path
from collections import defaultdict
from huggingface_hub import upload_folder

class SplitterDataset:
    def __init__(self, train_ratio=0.8, val_ratio=0.1, seed=42):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed

    def load_triplets(self, input_path: str):
        with open(input_path, 'r', encoding='utf-8') as infile:
            return [json.loads(line) for line in infile]
        
    def group_positive_context(self, triplets):
        grouped = defaultdict(list)
        for triplet in triplets:
            grouped[triplet['positive']].append(triplet)
        return grouped
    
    def split_contexts(self, contexts):
        random.shuffle(contexts)
        total = len(contexts)
        train = int(total * self.train_ratio)
        val = train + int(total * self.val_ratio)

        return contexts[:train], contexts[train:val], contexts[val:]
    
    def collect_triplets(self, grouped, context_ids):
        return [
            triplet 
            for ctx in context_ids
            for triplet in grouped[ctx]
        ]
    
    def save_triplets(self, output_path, triplets):
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for triplet in triplets:
                outfile.write(json.dumps(triplet, ensure_ascii=False) + '\n')

    def build_split_dataset(self, input_path, output_dir):
        random.seed(self.seed)
        triplets = self.load_triplets(input_path)
        grouped = self.group_positive_context(triplets)
        context_ids = list(grouped.keys())

        train_ctx, val_ctx, test_ctx = self.split_contexts(context_ids)

        train = self.collect_triplets(grouped, train_ctx)
        val = self.collect_triplets(grouped, val_ctx)
        test = self.collect_triplets(grouped, test_ctx)

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.save_triplets(f"{output_dir}/train.jsonl", train)
        self.save_triplets(f"{output_dir}/val.jsonl", val)
        self.save_triplets(f"{output_dir}/test.jsonl", test)

        print(f"Dataset generado correctamente en: '{output_dir}'")

    def push_huggingface(self, repo_id, local_dir):
        try:
            upload_folder(
                repo_id=repo_id,
                folder_path=local_dir,
                repo_type="dataset"
            )
        except Exception as e:
            print(f"Error al cargar Dataset: {e}")


 
if __name__ == "__main__":
    input_path = "../corpus/triplets_gemma.jsonl"
    output_dir = "../dataset"

    splitter = SplitterDataset()
    splitter.build_split_dataset(input_path, output_dir)
    splitter.push_huggingface(repo_id='', local_dir=output_dir)



