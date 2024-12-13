import os
import torch
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EvalPrediction,
    set_seed
)
from helper import get_genre_converter
from evaluation.metrics import compute_metrics as compute_metrics_multilabel
from globals import DATA_PATH, SPLIT_FOLDER
from datasets import Dataset
import numpy as np
import pandas as pd
import logging as log

os.environ["WANDB_DISABLED"] = "true"


class MovieGenreClassifier:
    def __init__(self, model_name, unique_genres, num_labels=21, seed=42069, prob_threshold=0.425, num_epochs=3):
        self.model_name = model_name
        self.unique_genres = unique_genres
        self.num_labels = num_labels
        self.seed = seed
        self.prob_threshold = prob_threshold
        self.num_epochs = num_epochs
        set_seed(self.seed)

    def load_data(self, file_path):
        """
        Load dataset from a CSV file with specific columns.
        """
        required_columns = ['movie_id', 'description', 'genre']
        dataset = pd.read_csv(
            file_path, converters=get_genre_converter(), usecols=required_columns)

        return dataset

    def split_data(self, data_path, save=False):
        """Split the dev data into 80/10/10 for train, validation, and test."""
        dataset = self.load_data(data_path)
        # Include movie_id alongside description
        X = dataset[['movie_id', 'description']].values
        y = np.vstack(dataset['genre'].values).astype(
            int)  # Stack rows of binary genre arrays

        X_train, y_train, X_temp, y_temp = iterative_train_test_split(
            X, y, test_size=0.2)
        X_val, y_val, X_test, y_test = iterative_train_test_split(
            X_temp, y_temp, test_size=0.5)

        # Split movie_id and description into separate columns
        train_movie_ids, train_descriptions = X_train[:, 0], X_train[:, 1]
        val_movie_ids, val_descriptions = X_val[:, 0], X_val[:, 1]
        test_movie_ids, test_descriptions = X_test[:, 0], X_test[:, 1]

        # Save processed data as pandas DataFrames
        train_data = pd.DataFrame({
            'movie_id': train_movie_ids,
            'description': train_descriptions,
            'genre': list(y_train)
        })
        eval_data = pd.DataFrame({
            'movie_id': val_movie_ids,
            'description': val_descriptions,
            'genre': list(y_val)
        })
        test_data = pd.DataFrame({
            'movie_id': test_movie_ids,
            'description': test_descriptions,
            'genre': list(y_test)
        })

        if save:
            dev_split_folder = os.path.join(DATA_PATH, SPLIT_FOLDER, 'dev')
            os.makedirs(dev_split_folder, exist_ok=True)

            train_data.to_csv(os.path.join(
                dev_split_folder, "train.csv"), index=False)

            eval_data.to_csv(os.path.join(
                dev_split_folder, "val.csv"), index=False)

            test_data.to_csv(os.path.join(
                dev_split_folder, "test.csv"), index=False)

        return train_data, eval_data, test_data

    def preprocess_function(self, examples):
        """Tokenize the input data and prepare ground truth labels."""
        tokenized_inputs = self.tokenizer(
            examples['description'], truncation=True, padding=True)
        tokenized_inputs["labels"] = torch.tensor(
            examples["genre"], dtype=torch.float32)

        return tokenized_inputs

    def compute_logits(self, preds):
        """Process the logits from the model predictions."""
        logits = preds[0] if isinstance(preds, tuple) else preds
        probabilities = torch.sigmoid(torch.tensor(logits)).numpy()
        y_pred = (probabilities > self.prob_threshold).astype(int)

        # If no class is predicted as positive
        no_positive_class = np.sum(y_pred, axis=1) == 0
        # Put the the one with highest prob as positive
        y_pred[no_positive_class, np.argmax(
            probabilities[no_positive_class], axis=1)] = 1

        return y_pred

    def compute_metrics(self, p: EvalPrediction, metric_names=None):
        """Helper fucntion to compute metrics."""
        y_pred = self.compute_logits(p.predictions)
        y_true = p.label_ids

        result = compute_metrics_multilabel(y_true, y_pred, metric_names)

        return result

    def load_model(self, model_path):
        """Load the model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=self.num_labels, problem_type="multi_label_classification"
        )

    def fine_tune(self, output_dir, train_data, eval_data):
        """Fine-tune the model on the training data."""
        self.load_model(self.model_name)

        train_dataset = Dataset.from_pandas(train_data)
        eval_dataset = Dataset.from_pandas(eval_data)

        tokenized_train_dataset = train_dataset.map(self.preprocess_function, batched=True)
        tokenized_eval_dataset = eval_dataset.map(self.preprocess_function, batched=True)

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        max_num_worker_suggest = None
        if hasattr(os, "sched_getaffinity"):
            try:
                max_num_worker_suggest = len(os.sched_getaffinity(0))
            except Exception:
                pass
        if max_num_worker_suggest is None:
            cpu_count = os.cpu_count()
            if cpu_count is not None:
                max_num_worker_suggest = cpu_count
        if max_num_worker_suggest is None:
            max_num_worker_suggest = 1

        log.info(f"max_num_worker_suggest: {max_num_worker_suggest}")

        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            dataloader_num_workers=max_num_worker_suggest,
            per_device_eval_batch_size=32,
            num_train_epochs=self.num_epochs,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=self.num_epochs
            # logging_dir=f'{output_dir}/logs',
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda p: self.compute_metrics(
                p, metric_names=['jaccard', 'hamming', 'accuracy', 'f1', 'precision', 'recall'])
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        self.trainer.train()

        self.best_model_path = os.path.join(output_dir, "best")
        if not os.path.exists(self.best_model_path):
            os.makedirs(self.best_model_path)
        self.trainer.save_model(self.best_model_path)

    def test(self, test_data):
        """Evaluate the model on the test data."""
        test_dataset = Dataset.from_pandas(test_data)
        tokenized_test_dataset = test_dataset.map(
            self.preprocess_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda p: self.compute_metrics(
                p, metric_names=['jaccard', 'hamming', 'accuracy', 'f1', 'precision', 'recall'])
        )

        predictions = trainer.predict(tokenized_test_dataset)

        return predictions

    def predict(self, data):
        """Predict the genres for the input data."""
        preds = self.test(data).predictions
        y_pred = self.compute_logits(preds)
        return y_pred
