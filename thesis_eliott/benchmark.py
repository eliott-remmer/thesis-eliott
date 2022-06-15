import utilities as ut
from baselines import bow_linear, dummy
from datasets import load_dataset
from sklearn.metrics import f1_score

if __name__ == "__main__":
    DATASET = "movies"
    PATH_TO_RESULTS = "thesis_eliott/results_baselines/"
    DATA_DIR = "/workspace/data"

    if DATASET == "movies":
        dataset_raw = load_dataset("movie_rationales")
        text = "review"
    elif DATASET == "tweets":
        dataset_raw = load_dataset(
            DATA_DIR + "/tweet-sentiment-extraction",
            data_files={"train": "train_for_fine_tune.csv", "test": "test_for_fine_tune.csv"},
        )
        text = "text"

    inputs_train = dataset_raw["train"][text]
    labels_train = dataset_raw["train"]["label"]
    inputs_val = dataset_raw["test"][text]
    labels_val = dataset_raw["test"]["label"]

    # baselines
    classifiers = [
        (dummy(labels_train, strategy="most_frequent"), "dummy_most_frequent"),
        (dummy(labels_train, strategy="stratified"), "dummy_stratified"),
        (bow_linear(inputs_train, labels_train), "bow_linear"),
    ]

    # evaluation
    for i, l, subset in [
        (inputs_train, labels_train, "train"),
        (inputs_val, labels_val, "val"),
    ]:
        for c, name in classifiers:
            f1_scores = []
            pred = c.predict(i)
            f1_scores.append(f1_score(l, pred, average="weighted"))

            ut.save_results(f1_scores, PATH_TO_RESULTS, "5grams_" + DATASET + "_" + name + "_" + subset + "_F1.txt")
