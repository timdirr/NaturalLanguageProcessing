import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed
)

from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.metrics import jaccard_score, hamming_loss, accuracy_score, f1_score, precision_score, recall_score
import os
import re
import torch

os.environ["WANDB_DISABLED"] = "true"
UNIQUE_GENRES = ['Action',
                'Adventure',
                'Animation',
                'Biography',
                'Comedy',
                'Crime',
                'Drama',
                'Family',
                'Fantasy',
                'Film-Noir',
                'History',
                'Horror',
                'Music',
                'Musical',
                'Mystery',
                'Romance',
                'Sci-Fi',
                'Sport',
                'Thriller',
                'War',
                'Western']

# TODO: maybe use sklearn.preprocessing.MultiLabelBinarizer instead?
def get_genre_converter():
    '''
    Returns a dictionary that can be used to convert genres when loading from a CSV file.
    Returns:
    -------
    dict
        The dictionary that can be used to convert genres when loading from a CSV file.
    '''
    return {"genre": lambda x: re.sub(r"[\[\]']", '', x).split(' ')}

# TODO: get paths and other stuff from globals.py and other modules
class MovieGenreClassifier:
    def __init__(self, model_name, num_labels=21, random_seed=42):
        self.model_name = model_name
        self.num_labels = num_labels
        self.random_seed = random_seed
        self.unique_genres = UNIQUE_GENRES
        set_seed(self.random_seed) # get seed from globals.py

    def load_data(self, file_path):
        """
        Load dataset from a CSV file with specific columns.
        """
        required_columns = ['movie_id', 'description', 'genre']
        dataset = pd.read_csv(file_path, converters=get_genre_converter(), usecols=required_columns)
        # self.unique_genres = sorted(pd.Series(dataset['genre'].str.extractall(r"'([^']*)'")[0]).unique())
        print(self.unique_genres)
        return dataset

    def split_data(self, data_path):
        """Split the dev data into 80/10/10 for train, validation, and test."""
        dataset = self.load_data(data_path)
        X = dataset[['movie_id', 'description']].values  # Include movie_id alongside description
        y = np.vstack(dataset['genre'].values).astype(int)  # Stack rows of binary genre arrays

        X_train, y_train, X_temp, y_temp = iterative_train_test_split(X, y, test_size=0.2)
        X_val, y_val, X_test, y_test = iterative_train_test_split(X_temp, y_temp, test_size=0.5)

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
        val_data = pd.DataFrame({
            'movie_id': val_movie_ids,
            'description': val_descriptions,
            'genre': list(y_val)
        })
        test_data = pd.DataFrame({
            'movie_id': test_movie_ids,
            'description': test_descriptions,
            'genre': list(y_test)
        })

        dev_folder = f'data/split/dev'
        os.makedirs(dev_folder, exist_ok=True)

        train_data.to_csv(f"{dev_folder}/train.csv", index=False)
        val_data.to_csv(f"{dev_folder}/val.csv", index=False)
        test_data.to_csv(f"{dev_folder}/test.csv", index=False)

        return train_data, val_data, test_data


    def preprocess_function(self, examples):
        tokenized_inputs = self.tokenizer(examples['description'], truncation=True, padding=True)
        # # Add the labels to the tokenized data
        # print(examples)
        # print(examples["genre"])
        tokenized_inputs["labels"] = torch.tensor(examples["genre"], dtype=torch.float32)
        # print(tokenized_inputs["labels"]

        return tokenized_inputs


    def compute_metrics(self, y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, average='samples'),
            "classification_report": classification_report(y_true, y_pred, target_names=self.unique_genres, output_dict=True)
        }

    def compute_metrics_our(self, y_true, y_pred, metrics_names=None):
        """
        Get metrics for multilabel classification.
        """
        if metrics_names is None:
            metrics_names = ['jaccard', 'hamming', 'accuracy', 'f1', 'precision',
                             'recall', 'classification_report', 'confusion_matrix']
        metrics = {}
        if 'jaccard' in metrics_names:
            metrics['jaccard'] = jaccard_score(y_true, y_pred, average='samples')
        if 'hamming' in metrics_names:
            metrics['hamming'] = hamming_loss(y_true, y_pred)
        if 'accuracy' in metrics_names:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
        if 'f1' in metrics_names:
            metrics['f1'] = f1_score(y_true, y_pred, average='samples')
        if 'precision' in metrics_names:
            metrics['precision'] = precision_score(y_true, y_pred, average='samples')
        if 'recall' in metrics_names:
            metrics['recall'] = recall_score(y_true, y_pred, average='samples')
        if 'classification_report' in metrics_names:
            metrics['classification_report'] = classification_report(y_true, y_pred, target_names=self.unique_genres, output_dict=True)
        # if 'confusion_matrix' in metrics_names:
            # metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=self.unique_genres)
        return metrics

    def load_model(self, model_path):
        """Load the model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=self.num_labels, problem_type="multi_label_classification"
        )


    def fine_tune(self, output_dir, train_data, val_data):
        """Fine-tune the model on the training data."""
        self.load_model(self.model_name)

        # if train_data is None or val_data is None:
        #     train_data = __load_csv(f"data/split/dev/train.csv", converters=get_genre_converter())
        #     val_data = __load_csv(f"data/split/dev/val.csv", converters=get_genre_converter())
        
        # TODO: Creat custom Dataset class maybe?
        train_dataset = Dataset.from_pandas(train_data)
        val_dataset = Dataset.from_pandas(val_data)

        tokenized_train_dataset = train_dataset.map(self.preprocess_function, batched=True)
        tokenized_val_dataset = val_dataset.map(self.preprocess_function, batched=True)

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True
            # logging_dir=f'{output_dir}/logs',
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
            # compute_metrics=lambda eval_pred: self.compute_metrics_our(eval_pred.label_ids, eval_pred.predictions,
            #                                                            metrics_names=['jaccard', 'hamming', 'accuracy', 'f1', 'precision', 'recall'])
        )

        self.trainer.train()

        self.best_model_path = os.path.join(output_dir, "best")
        if not os.path.exists(self.best_model_path):
            os.makedirs(self.best_model_path)
        self.trainer.save_model(self.best_model_path)

    def test(self, model_path, test_data):
        """Evaluate the model on the test data."""
        self.load_model(model_path)

        test_dataset = Dataset.from_pandas(test_data)
        tokenized_test_dataset = test_dataset.map(self.preprocess_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=data_collator
            # compute_metrics=lambda eval_pred: self.compute_metrics_our(eval_pred.label_ids, eval_pred.predictions,
                                                                      #  metrics_names=['jaccard', 'hamming', 'accuracy', 'f1', 'precision', 'recall'])
        )

        predictions = trainer.predict(tokenized_test_dataset)
        logits = predictions.predictions
        # TODO: transform logits to probabilites? (sigmoid)
        # probability > 0.5 === logits > 0
        y_pred = (logits > 0).astype(int)
        y_true = np.array(test_data['genre'].tolist()).astype(int)

        return y_true, y_pred, logits

# TODO: fix paths
if __name__ == "__main__":
    os.chdir('/content/drive/My Drive/Colab Notebooks/')
    print(os.getcwd())
    classifier = MovieGenreClassifier(model_name="distilbert-base-uncased", num_labels=21)
    train_data, val_data, test_data = classifier.split_data("data/dev.csv")

    # Test base model
    y_true_base, y_pred_base, logits_base = classifier.test(model_path=f"distilbert-base-uncased", test_data=test_data)
    results_base = classifier.compute_metrics_our(y_true_base, y_pred_base)
    print("Base Model Results:", results_base)

    # Fine-tune model and test it again on best model
    classifier.fine_tune(output_dir="distilbert_movie_genres", train_data=train_data, val_data=val_data)

    y_true, y_pred, logits = classifier.test(model_path="distilbert_movie_genres/best", test_data=test_data)
    results = classifier.compute_metrics_our(y_true, y_pred)
    print("Test Results:", results)
    # results_df = pd.DataFrame({
    #     'movie_id': self.test_data['movie_id'],
    #     'predicted_genres': predicted_genres,  # Your model's predictions
    #     'true_genres': self.test_data['genres']
    # })
    # results_df.to_csv('evaluation_results.csv', index=False)
