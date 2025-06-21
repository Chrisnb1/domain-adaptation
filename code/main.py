import pandas as pd
import matplotlib.pyplot as plt
from trainer.trainer import EmebddingTrainer
from evaluator.evaluator import EmbeddingEvaluator
from pathlib import Path
from datasets import load_dataset
from utils.config import Config


config = Config('config.yaml')
loss_config = config.get('training.loss')

BASE_DIR = Path(__file__)
OUTPUT_DIR =  BASE_DIR.parent / 'models'
DATASET_DIR = BASE_DIR.parent.parent / 'dataset'
EPOCH = config.get('training.num_epochs')
EPOCH_PATH = f"epoch_{EPOCH}"
LOSS = (
    "tpl_loss+mnrl_loss" if loss_config.get("tpl_loss") and loss_config.get("mnrl_loss")
    else "tpl_loss" if loss_config.get("tpl_loss")
    else "mnrl_loss" if loss_config.get("mnrl_loss")
    else None
)
RESULT_DIR = BASE_DIR.parent.parent / 'results'
MODEL_NAME = 'paraphrase-spanish-distilroberta-cowese'
MODEL_PATH = OUTPUT_DIR / LOSS / EPOCH_PATH / MODEL_NAME

if __name__ == "__main__":

    #----------Dataset----------#
    dataset = load_dataset('json', data_files={
        "train": f"{DATASET_DIR}/train.jsonl",
        "validation": f"{DATASET_DIR}/val.jsonl",
        "test": f"{DATASET_DIR}/test.jsonl"
    })
    dataset = dataset.rename_column('query', 'anchor')

    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    train_dataset = {
        "train_tpl": dataset["train"],
        "train_mnrl": dataset["train"]
    }

    eval_dataset = {
        "eval_tpl": dataset["validation"],
        "eval_mnrl": dataset["validation"]
    }
    
    #----------Entrenamiento----------#
    # trainer = EmebddingTrainer(output_dir=OUTPUT_DIR)
    # trainer.train(train_dataset, eval_dataset)

    #----------Validacion----------#
    # model_adapter_path = MODEL_PATH

    # evaluator_base = EmbeddingEvaluator()
    # retriever_results_base = evaluator_base.evaluate_retriever(DATASET_DIR)

    # evaluator_base = EmbeddingEvaluator(str(model_adapter_path))
    # retriever_results_adapted = evaluator_base.evaluate_retriever(DATASET_DIR)

    # evaluator_base.compare_models_table(retriever_results_base, retriever_results_adapted)

    #----------Metricas----------#
    metrics_csv = RESULT_DIR / "metrics.csv"
    # evaluator_base.save_metrics(
    #     retriever_results_base, 
    #     retriever_results_adapted,
    #     adapted_name=f"{LOSS}_{EPOCH}",
    #     output_path=metrics_csv,
    #     format="csv"
    #     )
    
    #----------Visualizar----------#
    df = pd.read_csv(metrics_csv, index_col=0)
    df_t = df.transpose()

    # for metric in df.index:
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(df_t.index, df_t[metric], marker='o')
    #     plt.title(f"Comparación de {metric}")
    #     plt.xlabel("Modelo")
    #     plt.ylabel(metric)
    #     plt.xticks(rotation=45, ha='right')
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.show()

    selected_metrics = ["precision@10", "accuracy@10", "map@100", "mrr@10", "ndcg@10", "recall@10"]
    df_selected = df.loc[selected_metrics]
    df_selected = df_selected.transpose()

    ax = df_selected.plot(kind='bar', figsize=(14, 6))
    plt.title("Comparación de modelos en múltiples métricas")
    plt.xlabel("Modelos")
    plt.ylabel("Valor de métrica")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Métrica", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(axis='y')
    plt.show()



