import torch
import torch.nn as nn
from transformers import BertTokenizer
import json
from data_prep import get_spk_to_utt, get_spk_to_grade, get_prompts_dict, align

def tokenize_text(utts, prompts):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_inputs_content = tokenizer(utts, prompts, padding=True, truncation=True, return_tensors="pt")
    encoded_inputs_blind = tokenizer(utts, padding=True, truncation=True, return_tensors="pt")

    input_ids_content = encoded_inputs_content['input_ids']
    input_ids_blind = encoded_inputs_blind['input_ids']
    mask_content = encoded_inputs_content['attention_mask']
    mask_blind = encoded_inputs_blind['attention_mask']
    token_ids_content = encoded_inputs_content['token_type_ids']
    token_ids_blind = encoded_inputs_blind['token_type_ids']
    return input_ids_content, input_ids_blind, mask_content, mask_blind, token_ids_content, token_ids_blind


def get_data(data_file, grades_file, prompts_mlf, grade_lim):
    '''
    Prepare data as tensors
    '''
    spk_to_utt = get_spk_to_utt(data_file)
    grade_dict = get_spk_to_grade(grades_file)
    prompts_dict = get_prompts_dict(prompts_mlf)
    utts, prompts, grades = align(spk_to_utt, grade_dict, prompts_dict, grade_lim)
    input_ids_content, input_ids_blind, mask_content, mask_blind, token_ids_content, token_ids_blind = tokenize_text(utts, prompts)
    labels = torch.FloatTensor(grades)

    return input_ids_content, input_ids_blind, mask_content, mask_blind, token_ids_content, token_ids_blind, labels
