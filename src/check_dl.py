# TODO: fix paths
import os

from classifier.dl import MovieGenreClassifier
from globals import DATA_PATH, EXPORT_PATH, SPLIT_FOLDER


if __name__ == "__main__":
    classifier = MovieGenreClassifier(model_name="distilbert-base-uncased", num_labels=21)
    path = os.path.join(DATA_PATH, SPLIT_FOLDER, "dev.csv")
    print(path)
    train_data, val_data, test_data = classifier.split_data(path)

    # Test base model
    y_true_base, y_pred_base, logits_base = classifier.test(model_path=f"distilbert-base-uncased", test_data=test_data)
    results_base = classifier.compute_metrics_our(y_true_base, y_pred_base)
    print("Base Model Results:", results_base)

    # Fine-tune model and test it again on best model
    classifier.fine_tune(output_dir=os.path.join(EXPORT_PATH, "distilbert_movie_genres"), train_data=train_data, val_data=val_data)

    y_true, y_pred, logits = classifier.test(model_path=os.path.join(EXPORT_PATH, "distilbert_movie_genres", 'best'), test_data=test_data)
    results = classifier.compute_metrics_our(y_true, y_pred)
    print("Test Results:", results)
    # results_df = pd.DataFrame({
    #     'movie_id': self.test_data['movie_id'],
    #     'predicted_genres': predicted_genres,  # Your model's predictions
    #     'true_genres': self.test_data['genres']
    # })
    # results_df.to_csv('evaluation_results.csv', index=False)
