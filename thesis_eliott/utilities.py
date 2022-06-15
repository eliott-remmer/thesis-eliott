import os

import numpy as np
import torch


def find_indices(sub_list, full_list):
    """
    returns indices in full_list of sub_list occureence
    example:
    full_list = ['sex', 'lax', '##ar', 'i', 'en', 'lax', 'ask']
    sub_list = ['i', 'en', 'lax']
    --> [3, 4, 5]
    Returns None if it is not a clean sublist
    :param sub_list: the list of words you want to find
    :param full_list: the list of words in the sentence
    :return: A list of indices of the sublist in the full list.
    """
    overlap = any(word in full_list for word in sub_list)  # are there any words in sub_list that are also in full_list?
    if not overlap:
        return None
    sl_len = len(sub_list)
    for ind, word in enumerate(full_list):
        if word == sub_list[0]:
            if full_list[ind : ind + sl_len] == sub_list:
                indices = list(range(ind, ind + sl_len))
                return indices


def get_cls_attention(attn):
    """
    Extract the attention weights by:
    - Taking the attention weights from the {last or first} multi-head attention layer assigned to the CLS token
    - Average each token across attention heads
    - Normalize across tokens
    adopted from https://github.com/hsm207/bert_attn_viz

    :param attn: (batch_size, num_heads, sequence_length, sequence_length)
    :return: norm_cls_attn[0] : (sequence_length)
    """
    layer = -1  # last or first
    CLS_pos = 0

    # extract attention layer and convert to np, #e.g. (1, 12, 5, 5), seq length = 5
    multihead_attn = attn[layer].detach().numpy()

    # For each multihead attention, get the attention weights going into the CLS token, (1, 12, 5)
    cls_attn = multihead_attn[:, :, CLS_pos, :]

    # Average across attention heads
    cls_attn = np.mean(cls_attn, axis=1)  # (1, 5)

    # normalize to [0, 1]
    normalized_cls_attn = (cls_attn - np.min(cls_attn)) / (np.max(cls_attn) - np.min(cls_attn))

    # return norm_cls_attn[0]
    return normalized_cls_attn[0]


def tokens2words(tokens, seq, token_prefix="##"):
    """
    Utility function to aggregate 'seq' on word-level based on 'tokens'.
    It takes a list of tokens and a list of words, and returns a list of words

    :param tokens: the list of tokens used for reference
    :param seq: the sequence of tokens/weights to be aggregated
    :param token_prefix: the prefix that is used to indicate that a token is a continuation of the
    previous token, defaults to ##
    :return: The aggregated list of tokens/weights
    """
    tmp = []
    for token, x in zip(tokens, seq):
        if token.startswith(token_prefix):
            if type(x) == str:
                x = x.replace(token_prefix, "")
            tmp[-1] += x
        else:
            if type(x) == str:
                tmp.append(x)
            else:
                tmp.append(x.item())

    return tmp if type(tmp[-1]) == str else torch.tensor(tmp)


def get_top_k(scores, tokens, k=0.2, output_indices=False, omit_scores=False, positive_only=True):
    """
    return top k percent tokens/indices based on scores
    connects scores with tokens/indices
    returns positive scores only

    :param scores: weights of each token in input
    :param tokens: input tokens
    :param k: ratio of tokens or scores to return
    :return: list of [scores, tokens] in ascending order
    """
    threshold = 0 if positive_only else -np.inf
    scores_and_tokens = []
    for i, score in enumerate(scores):
        if score > threshold:
            if output_indices:
                scores_and_tokens.append([score, i])
            else:
                scores_and_tokens.append([score, tokens[i]])
    scores_and_tokens.sort()  # sort ascending based on score
    top_start = int((len(scores) * (1 - k)))

    top_scores_and_tokens = scores_and_tokens[top_start:]
    if omit_scores:
        return [t[1] for t in top_scores_and_tokens]
    else:
        return top_scores_and_tokens


def calculate_iou(set1, set2):
    """
    The function takes two sets as input and returns the intersection of the two sets divided by the
    union of the two sets

    :param set1: The first set of words/indices
    :param set2: The second set of words/indices
    :return: The similarity between the two sets, IOU, jaccard similarity
    """
    intersection = set1 & set2
    union = set1 | set2
    similarity = len(intersection) / len(union)
    return similarity


def indices_to_binary(tokens, indices):
    """
    It takes a list of tokens and a list of indices, and returns a binary vector of length len(tokens)
    where the indices in the list are set to 1

    :param tokens: a list of all the tokens in the sentence
    :param indices: the indices of the tokens that correspond to the evidence
    :return: A binary vector of the same length as the tokens, with 1s at the indices of the tokens that
    are in the event.
    """
    ev_bin = np.zeros(len(tokens))
    for i in range(len(ev_bin)):
        if i in indices:
            ev_bin[i] = 1
    return ev_bin


def get_random_indices(tokens, k):
    """
    Given a list of tokens and a float k, return a list of random indices of length k*len(tokens)

    :param tokens: the list of tokens in the input sentence
    :param k: the ratio of tokens to include
    :return: A list of random indices
    """
    expl_len = int(len(tokens) * k)
    input_index_range = list(range(len(tokens)))
    indices = np.random.choice(input_index_range, expl_len)
    return indices


def get_random_weights(tokens):
    """
    It takes a list of tokens and returns a list of random weights.
    :param tokens: the list of tokens in the explanation
    :return: A list of random weights for each token in the explanation.
    """
    expl_len = len(tokens) - 2  # exclude [CLS] and [SEP] tokens
    weights = np.random.uniform(size=expl_len)
    return weights


def save_results(content, path, fname):
    """
    This function takes some content, a path, and a filename, and saves the content to a file
    in the path with the given filename.

    :param content: the list of objects to be written to the file
    :param path: the path to the directory where the data is stored
    :param fname: the name of the file to be saved
    """
    with open(os.path.join(path, fname), "w") as f:
        for item in content:
            f.write("%s\n" % item)
