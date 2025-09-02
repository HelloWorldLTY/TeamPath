import re
import string
from typing import Dict

import numpy as np
import evaluate
bleu_s = evaluate.load("bleu")

def get_multi_choice_prediction(response, all_choices=string.ascii_uppercase):
    """PathMMU response capture function"""
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    response = " " + response + " " # add space to avoid partial match

    ans_with_brack = False
    ans_with_half_brack = False
    ans_with_period = False
    candidates = []
    for choice in all_choices:  # (A) (B) (C) (D)
        if f'({choice})' in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # A) B) C) D)
            if f'{choice})' in response:
                candidates.append(choice)
                ans_with_half_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # A) B) C) D)
            if f'{choice}.' in response:
                candidates.append(choice)
                ans_with_period = True

    if len(candidates) == 0:
        for choice in all_choices: # A B C D
            if f' {choice} ' in response:
                candidates.append(choice)

    if len(candidates) > 1:
        start_indexes = []
        if ans_with_brack:
            for can in candidates:
                index = response.rfind(f'({can})')
                start_indexes.append(index) # -1 will be ignored anyway
        elif ans_with_half_brack:
            for can in candidates:
                index = response.rfind(f'{can})')
                start_indexes.append(index) # -1 will be ignored anyway
        elif ans_with_period:
            for can in candidates:
                index = response.rfind(f'{can}.')
                start_indexes.append(index)  # -1 will be ignored anyway
        else:
            for can in candidates:
                index = response.rfind(f" {can} ")
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]

    elif len(candidates) == 1: # only one candidate
        pred_index = candidates[0]
    else:
        pred_index = "[INVALID]"

    return pred_index


def extract_last_answer(text):
    pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> Dict:
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"  # no any character on both sides
    match = re.fullmatch(pattern, solution_str, re.DOTALL)
    follow_format = match is not None

    # remove <answer></answer> tags
    cleaned_solution = extract_last_answer(solution_str)
    if cleaned_solution is None:
        cleaned_solution = solution_str

    # capture the capital letter from the solution
    if len(ground_truth) <=1:
        pred = get_multi_choice_prediction(cleaned_solution)
    
        if pred.strip() == ground_truth.strip():
            reward = 1.0
        else:
            reward = 0.0
    
        acc = reward == 1.0
    else:
        pred = cleaned_solution
        if len(pred) ==0:
            reward = 0
        else:
            reward = bleu_s.compute(predictions=[cleaned_solution], references=[ground_truth], smooth=True)['bleu']
        acc = reward
    return {
        "score": reward,
        "acc": acc,
        "pred": pred,
        "follow_format": follow_format,
        "gt": ground_truth,
        "raw_solution": cleaned_solution,
    }