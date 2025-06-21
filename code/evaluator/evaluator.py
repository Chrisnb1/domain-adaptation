import csv
from sentence_transformers.util import cos_sim
from sentence_transformers.evaluation import TripletEvaluator, InformationRetrievalEvaluator
from datasets import load_dataset, concatenate_datasets
from embedding.embedding import Embedding
from tabulate import tabulate

class EmbeddingEvaluator(Embedding):
    def __init__(self, model_path: str = None):
        super().__init__(model_path=model_path)

    def load_valid_dataset(self, input_path: str, limit: int = None):
        data = load_dataset('json', data_files=input_path)['train']
        data = data.rename_column('query', 'anchor')
        if limit:
            data = data[:limit]
        anchors = data['anchor']
        positives = data['positive']
        negatives = data['negative']
        return anchors, positives, negatives
    
    def evaluate_triplets(self, triplet_path: str, split_name: str, limit: int = None):
        anchors, positives, negatives = self.load_valid_dataset(triplet_path)
        
        evaluator = TripletEvaluator(anchors, positives, negatives, name=split_name)
        results = evaluator(self.model)
        return results
    
    def load_corpus_and_queries(self, input_path: str):
        dataset = load_dataset('json', data_files={
            "train": f"{input_path}/train.jsonl",
            "validation": f"{input_path}/val.jsonl",
            "test": f"{input_path}/test.jsonl"
        })

        for split in dataset:
            if "id" not in dataset[split].column_names:
                dataset[split] = dataset[split].add_column("id", list(range(len(dataset[split]))))

        corpus = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
        corpus_dict = dict(zip(corpus["id"], corpus["positive"]))
        queries = dict(zip(dataset["test"]["id"], dataset["test"]["query"]))
        relevant_docs = {q_id: [q_id] for q_id in queries}
        return corpus_dict, queries, relevant_docs
    
    def evaluate_retriever(self, input_path: str):
        corpus, queries, relevant_docs = self.load_corpus_and_queries(input_path)
        
        evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=self.model_name,
            score_functions={"cosine": cos_sim},
        )
        results = evaluator(self.model)
        return results
    
    def clean_metric_name(self, metric):
        parts = metric.split('_cosine_', 1)
        if len(parts) > 1:
            metric_suffix = parts[1]
        else:
            metric_suffix = metric

        return metric_suffix


    def compare_models_table(self, base_metrics, finetuned_metrics):
        table_data = []

        base_cleaned = {self.clean_metric_name(k): v for k, v in base_metrics.items()}
        finetuned_cleaned = {self.clean_metric_name(k): v for k, v in finetuned_metrics.items()}

        common_metrics = set(base_cleaned.keys()) & set(finetuned_cleaned.keys())

        for metric in sorted(common_metrics):
            base_value = base_cleaned[metric]
            finetuned_value = finetuned_cleaned[metric]
            table_data.append([metric, f"{base_value:.4f}", f"{finetuned_value:.4f}"])

        headers = ["Métrica", "Modelo Base", "Modelo Ajustado"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))


    def save_metrics(self, base_metrics, adapted_metrics, adapted_name, output_path: str, format: str = "csv"):
        base_cleaned = {self.clean_metric_name(k): v for k, v in base_metrics.items()}
        adapted_cleaned = {self.clean_metric_name(k): v for k, v in adapted_metrics.items()}
        common_metrics = sorted(set(base_cleaned.keys()) & set(adapted_cleaned.keys()))

        new_column = {metric: round(adapted_cleaned[metric], 4) for metric in common_metrics}

        if format == "csv":
            existing_data = {}

            if os.path.exists(output_path):
                with open(output_path, mode='r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        metric = row["metric"]
                        existing_data[metric] = row

            updated_rows = []
            for metric in common_metrics:
                row = existing_data.get(metric, {"metric": metric, "base": round(base_cleaned[metric], 4)})
                row[adapted_name] = new_column[metric]
                updated_rows.append(row)

            all_columns = ["metric", "base"] + sorted(set(k for row in updated_rows for k in row if k not in ("metric", "base")))

            with open(output_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=all_columns)
                writer.writeheader()
                writer.writerows(updated_rows)
        else:
            raise ValueError("Formato no soportado")


