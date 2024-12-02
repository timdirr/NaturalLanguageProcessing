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


def test_dl_model():
    classifier = MovieGenreClassifier(model_name="distilbert-base-uncased",
                                      unique_genres=UNIQUE_GENRES, num_labels=len(UNIQUE_GENRES), seed=SEED)
    dev_dataset_path = os.path.join(DATA_PATH, SPLIT_FOLDER, "dev.csv")
    train_data, val_data, test_data = classifier.split_data(dev_dataset_path)

    # Test base model
    # y_true_base, y_pred_base, logits_base = classifier.test(model_path=f"distilbert-base-uncased", test_data=test_data)
    # results_base = classifier.compute_metrics_our(y_true_base, y_pred_base)
    # print("Base Model Results:", results_base)

    output_dir = os.path.join(MODEL_PATH, "distilbert_movie_genres")
    # Fine-tune model
    classifier.fine_tune(output_dir=output_dir,
                         train_data=train_data, val_data=val_data)

    # Test fine-tuned model
    predictions = classifier.test(model_path=os.path.join(
        output_dir, 'best'), test_data=test_data)
    results = classifier.compute_metrics(predictions)

    metrics_path = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_path, exist_ok=True)

    with open(os.path.join(metrics_path, "best_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    classification_report = results.pop("classification_report", {})

    table_data = [[metric, round(value, 2)]
                  for metric, value in results.items()]
    table = tabulate(table_data, headers=[
                     "Metric", "Value"], tablefmt="pretty")

    log.info("\nTest Metrics:\n")
    log.info(table)

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
        log.info("\nClassification report:\n")
        log.info(detailed_table)

    log.info(f"\nMetrics saved to {metrics_path}")

    preds_path = os.path.join(output_dir, "predictions")
    os.makedirs(preds_path, exist_ok=True)

    results_df = pd.DataFrame({
        'movie_id': test_data['movie_id'],
        'y_pred': list(classifier.compute_logits(predictions.predictions)),
        'y_true': test_data['genre']
    })
    results_df.to_csv(os.path.join(preds_path, "best_preds.csv"), index=False)

    log.info(f"Predictons saved to {preds_path}")

    return predictions
