from sentence_transformers import SentenceTransformerTrainingArguments, SentenceTransformerTrainer
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.losses import TripletLoss, MultipleNegativesRankingLoss
from embedding.embedding import Embedding

class EmebddingTrainer(Embedding):
    def __init__(self, model_path: str = None, output_dir: str = None):
        super().__init__(model_path=model_path)
        self.output_dir = output_dir

    def _set_ouput_dir(self, new_output_dir):
        self.output_dir = new_output_dir

    def get_args(self):
        loss_dict = self.config.get('training.loss')
        active_losses = [key for key, value in loss_dict.items() if value]
        loss_str = "+".join(active_losses) if active_losses else "no_loss"
        epochs = self.config.get('training.num_epochs')

        run_name = f"{loss_str}_{epochs}_epoch"

        output_dir = f"{self.output_dir}/{loss_str}/epoch_{epochs}"
        self._set_ouput_dir(output_dir)
        return SentenceTransformerTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.get('training.num_epochs'),
            per_device_train_batch_size=self.config.get('training.per_device_train_batch_size'),
            per_device_eval_batch_size=self.config.get('training.per_device_eval_batch_size'),
            learning_rate=float(self.config.get('training.learning_rate')),
            warmup_ratio=self.config.get('training.warmup_ratio'),
            fp16=self.config.get('training.fp16'),
            bf16=self.config.get('training.bf16'),
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            lr_scheduler_type=self.config.get('training.lr_scheduler_type'),
            optim=self.config.get('training.optim'),
            eval_strategy=self.config.get('training.eval_strategy'),
            eval_steps=self.config.get('training.eval_steps'),
            save_strategy=self.config.get('training.save_strategy'),
            save_steps=self.config.get('training.save_steps'),
            save_total_limit=self.config.get('training.save_total_limit'),
            logging_steps=self.config.get('training.logging_steps'),
            run_name=run_name,
            report_to=self.config.get('training.report_to')
        )
    
    def train(self, train_dataset, eval_dataset):
        loss_config = self.config.get('training.loss')

        losses = {}

        if loss_config.get("tpl_loss", False):
            losses["train_tpl"] = TripletLoss(self.model)
            losses["eval_tpl"] = TripletLoss(self.model)

        if loss_config.get("mnrl_loss", False):
            losses["train_mnrl"] = MultipleNegativesRankingLoss(self.model)
            losses["eval_mnrl"] = TripletLoss(self.model)
        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=self.get_args(),
            train_dataset=train_dataset, #.select_columns(["anchor", "positive", "negative"]),
            eval_dataset=eval_dataset, #.select_columns(["anchor", "positive", "negative"]),
            loss=losses
        )
        trainer.train()
        self._save_model()
    
    def _save_model(self):
        model_name = f"{self.model_name}-cowese"
        path = f"{self.output_dir}/{model_name}"
        self.model.save_pretrained(path=path, model_name=model_name)

    def push_hf(self, path_model):
        self.model.push_to_hub(path_model)

