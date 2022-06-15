import re

import pandas as pd
from sklearn.model_selection import train_test_split


def remove_url(text):
    """
    It takes a string and replaces all URLs with the string <url>

    :param text: The text to be processed
    :return: the text with the url removed.
    """
    return re.sub(
        r"(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])",
        "<url>",
        text,  # noqa: W605
    )


def remove_non_letter_non_number(text):
    """
    It takes a string as input, and returns a string with all non-letter and non-number characters
    removed
    :param text: The text to be processed
    :return: a string with all non-letter and non-number characters removed.
    """
    return re.sub(r"[^A-Za-z0-9 ]+", "", text)


def main():
    # read train data
    df = pd.read_csv("data/tweet-sentiment-extraction/train.csv")

    # rename column and labels
    df = df.rename(columns={"sentiment": "label"})
    df.loc[df["label"] == "negative", "label"] = 0
    df.loc[df["label"] == "neutral", "label"] = 1
    df.loc[df["label"] == "positive", "label"] = 2

    # drop nan-values
    df = df.dropna()

    # split data into train and test to have a smaller test set with explanations
    df_train, df_test = train_test_split(df, test_size=0.1)

    # remove urls for fine-tuning
    df_train["text"] = df_train["text"].apply(remove_url)

    # save train and test with selected text for explaining
    df_train_expl = df_train
    df_test_expl = df_test

    df_train_expl.to_csv("data/tweet-sentiment-extraction/train_for_explain.csv", index=False)
    df_test_expl.to_csv("data/tweet-sentiment-extraction/test_for_explain.csv", index=False)

    # save cleaned csv train without "selected_text" for fine-tuning
    df_train = df_train.drop(columns=["selected_text"])
    df_train.to_csv("data/tweet-sentiment-extraction/train_for_fine_tune.csv", index=False)

    # save cleaned csv test data for fine-tuning
    df_test = df_test.drop(columns=["selected_text"])
    df_test.to_csv("data/tweet-sentiment-extraction/test_for_fine_tune.csv", index=False)


if __name__ == "__main__":
    main()
