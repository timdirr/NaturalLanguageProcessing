import os
import json
from classifier.dl import MovieGenreClassifier
from globals import DATA_PATH, MODEL_PATH, SPLIT_FOLDER, SEED, UNIQUE_GENRES
import pandas as pd
from tabulate import tabulate
import logging as log

log.basicConfig(level=log.INFO,
                format='%(asctime)s: %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')


def show_metrics(results, output_dir, save=True):
    """Display and save metrics to a file."""
    metrics_path = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_path, exist_ok=True)
    conf_mat = None
    conf_mat = results.pop("confusion_matrix", None)

    if save:
        with open(os.path.join(metrics_path, "best_metrics.json"), "w") as f:
            json.dump(results, f, indent=4)
        log.info(f"Metrics saved to {metrics_path}")

    classification_report = results.pop("classification_report", {})

    table_data = [[metric, round(value, 2)]
                  for metric, value in results.items()]
    table = tabulate(table_data, headers=[
                     "Metric", "Value"], tablefmt="pretty")

    log.info(f"\nTest Metrics:\n{table}")

    if classification_report:
        detailed_table_data = [
            [genre, round(metrics["precision"], 2), round(metrics["recall"], 2), round(
                metrics["f1-score"], 2), metrics["support"]]
            for genre, metrics in classification_report.items()
        ]
        detailed_table = tabulate(
            detailed_table_data,
            headers=["Genre", "Precision", "Recall", "F1-Score", "Support"],
            tablefmt="pretty"
        )
        log.info(f"\nClassification report:\n{detailed_table}")

    if conf_mat is not None:
        rows = []
        for genre, matrix in zip(UNIQUE_GENRES, conf_mat):
            tn, fp, fn, tp = matrix.ravel()
            rows.append({"Genre": genre, "TN": tn,
                        "FP": fp, "FN": fn, "TP": tp})

        df = pd.DataFrame(rows)
        table = tabulate(df, headers="keys",
                         tablefmt="pretty", showindex=False)
        log.info(f"\nConfusion matrix:\n{table}")


def save_predictions(output_dir, classifier, test_data, predictions):
    """Save predictions to a CSV file."""
    preds_path = os.path.join(output_dir, "predictions")
    os.makedirs(preds_path, exist_ok=True)

    results_df = pd.DataFrame({
        'movie_id': test_data['movie_id'],
        'y_pred': list(classifier.compute_logits(predictions.predictions)),
        'y_true': test_data['genre']
    })
    results_df.to_csv(os.path.join(preds_path, "best_preds.csv"), index=False)

    log.info(f"Predictons saved to {preds_path}")


def test_dl_model():
    classifier = MovieGenreClassifier(model_name="distilbert-base-uncased",
                                      unique_genres=UNIQUE_GENRES, num_labels=len(UNIQUE_GENRES), seed=SEED)
    dev_dataset_path = os.path.join(DATA_PATH, SPLIT_FOLDER, "dev.csv")
    train_data, val_data, test_data = classifier.split_data(dev_dataset_path)
    output_dir = os.path.join(MODEL_PATH, "distilbert_movie_genres")

    # Test base model
    # predictions_base = classifier.test(model_path=f"distilbert-base-uncased", test_data=test_data)
    # results_base = classifier.compute_metrics(predictions_base)
    # print("Base Model Results:", results_base)

    # Fine-tune model
    classifier.fine_tune(output_dir=output_dir,
                         train_data=train_data, val_data=val_data)

    # Test fine-tuned model
    predictions = classifier.test(model_path=os.path.join(
        output_dir, 'best'), test_data=test_data)
    results = classifier.compute_metrics(predictions)

    show_metrics(results, output_dir, save=True)
    save_predictions(output_dir, classifier, test_data, predictions)


if __name__ == "__main__":
    test_dl_model()
