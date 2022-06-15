import functools
import os
import sys

import numpy as np
import shap
import torch
import utilities as ut
from datasets import load_dataset
from sklearn.metrics import auc, precision_recall_curve
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

module_path = os.path.abspath(os.path.join("."))
if module_path not in sys.path:
    sys.path.append(module_path)
from data.movies.utils import annotations_from_jsonl, load_documents  # noqa: E402


def get_movies_dataset(split):
    """
    It loads the documents and annotations from the `data/movies` directory, and returns them as a tuple

    :param split: the split of the dataset to use. Can be one of "train", "val" or "test"
    :return: dataset containing annotations, documents containing input text
    """
    data_root = os.path.join("data", "movies")
    documents = load_documents(data_root)
    dataset = annotations_from_jsonl(os.path.join(data_root, split + ".jsonl"))
    return dataset, documents


def get_tweets_dataset(split):
    """
    It loads the dataset from the specified path and returns the dataframe for the specified split

    :param split: The split of the dataset to load
    :return: A dataframe with the columns textID, text, selected_text, and label.
    """
    dataset_raw = load_dataset("/workspace/data/tweet-sentiment-extraction", data_files={split: split + ".csv"})
    # dataset_raw = load_dataset(
    #    "../thesis-eliott/data/tweet-sentiment-extraction", data_files={split: split + ".csv"}
    # )  # for debugging
    return dataset_raw[split]


def get_attention_expl(model, tokenized_input, token_level=True):
    """
    The function takes a model, a tokenized input, and a boolean value for token_level. It
    returns the attention weights corresponding to the CLS token.

    :param model: the model we want to use to get the attention weights
    :param tokenized_input: a dictionary containing the input_ids and token_type_ids of the input text
    :param token_level: if True, returns the attention weights for each token. If False, returns the
    attention weights for each word, defaults to True (optional)
    :return: The attention weights for the tokens/words in the input.
    """
    input_ids = tokenized_input["input_ids"]
    token_type_ids = tokenized_input["token_type_ids"]
    input_id_list = input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)

    # run input through model to get attention weights
    attention = model(input_ids, token_type_ids=token_type_ids)[
        -1
    ]  # beware for using non-bert models, probably need to pass more than just input_ids

    # get cls attention
    cls_attn = ut.get_cls_attention(attention)

    if token_level:
        return cls_attn

    # concatenate attention vector, remove weights to ##-tokens
    cls_attn_words = ut.tokens2words(tokens, cls_attn)
    return cls_attn_words


def get_shap_expl(tokenized_input, ref_token, nsamples=500, token_level=True):
    """
    The function takes a tokenized input, a reference token, and a number of samples, and returns a vector of SHAP
    values for each token in the input

    :param tokenized_input: a dictionary containing the input_ids and token_type_ids of the input text
    :param ref_token: the token that is used as a reference for the baseline (currently [MASK] - could try others)
    :param nsamples: number of samples to use for the shap values. The higher the number, the more
    accurate the shap values will be, but the longer it will take to compute, defaults to 500 (optional)
    :param token_level: if True, returns the shap values for each token. If False, returns the shap
    values for each word, defaults to True (optional)
    :return: The shap values for each token in the input.
    """
    # make prediction on input
    input_ids_np = tokenized_input["input_ids"].detach().numpy()
    pred = predict_fn(input_ids_np)
    pred_label = pred.argmax()

    # create baseline
    baseline = input_ids_np.copy()
    baseline[:, 1:-1] = ref_token  # Keep CLS and SEP tokens fixed in baseline

    # define explainer
    predict_fn_label = functools.partial(
        predict_fn, label=pred_label
    )  # creates a copy of predict_fn which always sends the predicted label
    explainer = shap.KernelExplainer(predict_fn_label, baseline)

    # get shap values (~1 minute with nsamples = 500)
    phi = explainer.shap_values(input_ids_np, nsamples=nsamples)[0]

    if token_level:
        return phi

    # concatenate phi vector, remove weights to ##-tokens
    phi_words = ut.tokens2words(tokens, phi)
    return phi_words


def predict_fn(
    input_ids, attention_mask=None, batch_size=32, label=None, output_logits=False, repeat_input_ids=False, device="cpu"
):
    """
    Prediction function used by SHAP. It takes in a list of input_ids, and returns a list of probabilities for each class.
    Wrapper function for a Huggingface Transformers model into the format that KernelSHAP expects,
    i.e. where inputs and outputs are numpy arrays.

    :param input_ids: the input text, as a list of integers
    :param attention_mask: This is a binary mask that tells the model which tokens to pay attention to
    :param batch_size: The batch size to use for prediction, defaults to 32 (optional)
    :param label: the label to predict. If None, then all labels are predicted
    :param output_logits: If True, the output will be the logits of the model. If False, the output will
    be the softmax probabilities, defaults to False (optional)
    :param repeat_input_ids: if True, the input_ids will be repeated to match the shape of
    attention_mask.
    :param device: the device to run the model on, defaults to cpu (optional)
    :return: The probability of the input being in the class.
    """

    model.to(device)
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.ones_like(input_ids) if attention_mask is None else torch.tensor(attention_mask)

    if repeat_input_ids:
        assert input_ids.shape[0] == 1
        input_ids = input_ids.repeat(attention_mask.shape[0], 1)

    ds = torch.utils.data.TensorDataset(input_ids.long(), attention_mask.long())
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size)
    probas = []
    logits = []
    with torch.no_grad():
        for batch in dl:
            out = model(batch[0].to(device), attention_mask=batch[1].to(device))
            logits.append(out.logits.detach().cpu())
            probas.append(torch.nn.functional.softmax(out.logits.detach().cpu(), dim=1).detach())
    logits = torch.cat(logits, dim=0).numpy()
    probas = torch.cat(probas, dim=0).numpy()

    if label is not None:
        probas = probas[:, label]
        logits = logits[:, label]

    return (probas, logits) if output_logits else probas


if __name__ == "__main__":
    # set parameters
    SEED = 0
    PATH_TO_RESULTS = "thesis_eliott/results_explain/"
    DATASET = "tweets"
    SPLIT = "train_for_explain"
    METHOD = "random"

    MODEL = "bert-base-uncased-finetuned-" + DATASET
    NSAMPLES = 500  # shap parameter
    BASELINE = "[MASK]" if METHOD == "shap" else ""
    TOP_K = 0.599 if DATASET == "tweets" else 0.176
    K = 1 if METHOD == "shap" else TOP_K

    # set seed for reproducibility
    np.random.seed(SEED)

    # import model and tokenizer
    model_path = "models/" + MODEL
    model = BertForSequenceClassification.from_pretrained(model_path, output_attentions=True)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # load data
    if DATASET == "movies":
        dataset, documents = get_movies_dataset(SPLIT)
    elif DATASET == "tweets":
        dataset = get_tweets_dataset(SPLIT)

    # collect results
    ious = []
    pr_aucs = []

    # evaluation loop
    for i in tqdm(range(len(dataset))):
        # extract one example
        instance = dataset[i]
        if DATASET == "movies":  # datasets has different structure
            review = documents[instance.annotation_id]
            input_text_list = [word for sentence in review for word in sentence]
            evidences = instance.all_evidences()
            label = instance.classification
        elif DATASET == "tweets":
            input_text_list = instance["text"].split()
            evidences = instance["selected_text"].split()
            label = instance["label"]

        # tokenize input
        input_text_str = " ".join(input_text_list)
        tokenized_input = tokenizer.encode_plus(
            input_text_list, return_tensors="pt", truncation=True, is_split_into_words=True
        )  # truncated to 512 tokens
        input_ids = tokenized_input["input_ids"]
        token_type_ids = tokenized_input["token_type_ids"]
        input_id_list = input_ids[0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_id_list)

        # tokenize explanations and extract indices in input
        if DATASET == "movies":  # datasets has different structure
            indices_tmp = []
            for ev in evidences:
                if ev.text != "":
                    tokenized_expl = tokenizer.encode_plus(
                        ev.text, return_tensors="pt", truncation=True, add_special_tokens=False
                    )
                    input_ids_expl = tokenized_expl["input_ids"]
                    input_id_list_expl = input_ids_expl[0].tolist()
                    tokens_expl = tokenizer.convert_ids_to_tokens(input_id_list_expl)
                    expl_indices = ut.find_indices(tokens_expl, tokens)
                    if expl_indices is not None:  # explanations on content above 512 tokens
                        indices_tmp.append(expl_indices)
            gt_indices = [idx for indices in indices_tmp for idx in indices]

        elif DATASET == "tweets":
            tokenized_expl = tokenizer.encode_plus(
                evidences, return_tensors="pt", truncation=True, is_split_into_words=True, add_special_tokens=False
            )  # truncated to 512 tokens
            input_ids_expl = tokenized_expl["input_ids"]
            input_id_list_expl = input_ids_expl[0].tolist()
            tokens_expl = tokenizer.convert_ids_to_tokens(input_id_list_expl)
            gt_indices = ut.find_indices(tokens_expl, tokens)
            if gt_indices is None:  # skip example if there is no pure overlap
                continue

        gt_binary = ut.indices_to_binary(tokens, gt_indices)[1:-1]  # exclude [CLS] and [SEP]

        # produce explanation
        if METHOD == "attention":
            expl_weights = get_attention_expl(model, tokenized_input)[1:-1]  # exclude [CLS] and [SEP]
        elif METHOD == "shap":
            if BASELINE == "[MASK]":
                ref_token = tokenizer.mask_token_id
            elif BASELINE == "[PAD]":
                ref_token = tokenizer.pad_token_id
            elif BASELINE == "[UNK]":
                ref_token = tokenizer.unk_token_id
            expl_weights = get_shap_expl(tokenized_input, ref_token, nsamples=NSAMPLES)[1:-1]  # exclude [CLS] and [SEP]
        elif METHOD == "random":
            expl_weights = ut.get_random_weights(tokens)

        expl_indices = ut.get_top_k(expl_weights, tokens, k=K, output_indices=True, omit_scores=True)
        if len(gt_indices) != 0:  # no expl in ground truth gives nan value
            # pr auc evaluation
            precision, recall, thresholds = precision_recall_curve(gt_binary, expl_weights)
            pr_auc = auc(recall, precision)
            pr_aucs.append(pr_auc)

            # iou evaluation
            iou = ut.calculate_iou(set(gt_indices), set(expl_indices))
            ious.append(iou)

    ut.save_results(ious, PATH_TO_RESULTS, DATASET + "_" + SPLIT + "_" + METHOD + "_" + BASELINE + "_IOU.txt")
    ut.save_results(pr_aucs, PATH_TO_RESULTS, DATASET + "_" + SPLIT + "_" + METHOD + "_" + BASELINE + "_PRAUC.txt")
    print(
        "MODEL: %s\nDATASET: %s\nSPLIT: %s\nMETHOD: %s\nBASELINE: %s\nK: %s\nSEED: %s"
        % (MODEL, DATASET, SPLIT, METHOD, BASELINE, K, SEED)
    )
