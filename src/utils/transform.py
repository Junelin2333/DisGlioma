from typing import Any, Mapping
import torch

clinical_variable_token_list = ['age']
clinical_variable_status_token_list = ['age']

def map_age_category(age):
    if 15 <= age <= 47:
        return 0
    elif 48 <= age <= 63:
        return 1
    elif age >= 64:
        return 2
    else:
        return -1

def map_kps_category(kps):
    if 30 <= kps <= 50:
        return 0
    elif 60 <= kps <= 70:
        return 1
    elif 80 <= kps <= 100:
        return 2
    else:
        return -1
    
    
mapping_category_dict = {
    'age': map_age_category,
    'kps': map_kps_category,
}
    
def get_age_description(age):
    category = map_age_category(age)
    return age_mapping[category]
def get_kps_description(kps):
    category = map_kps_category(kps)
    return kps_mapping.get(category, "Performance status information is unavailable.")

sex_mapping = {0: "The patient is male.",
               1: "The patient is female."}

age_mapping = {
    0: "The patient is young.",
    1: "The patient is midlife.",
    2: "The patient is geriatric."
}

kps_mapping = {
    0: "The patient has a poor performance status.",
    1: "The patient has a moderate performance status.",
    2: "The patient has a good performance status."
}


mapping_dict = {
    "sex": sex_mapping,
    "age": age_mapping,
    "kps": kps_mapping,
}

CLS_TOKEN_OFFSET = 1

description_indices = {
    "sex": 3,
    "age": 3,
    "kps": 4
}

def get_tokenized_diff_index(text, diff_word, tokenizer):
    """
    Find the token index where the differing word starts after tokenization.

    Args:
        text (str): Full text string.
        diff_word (str): The word that differs.
        tokenizer: Tokenizer to use.

    Returns:
        int: Start index of the differing word after tokenization, or -1 if not found.
    """
    tokens = tokenizer.tokenize(text)
    for i, token in enumerate(tokens):
        # Subword tokenization may vary by tokenizer
        if diff_word in token or token in diff_word:
            return i
    return -1

def transform_label(label_dict):
    transformed = {}
    
    CLS_TOKEN_OFFSET = 1
    
    transformed = {}
       
    for key, value in label_dict.items():
        
        if key in mapping_category_dict:
            value = mapping_category_dict[key](value)
        
        if isinstance(value, str):
            transformed[key] = value
        else:
            transformed[key] = torch.tensor(value)
                
        if key in clinical_variable_token_list:
            if value is not None and value != -1:
                if key in mapping_dict and value in mapping_dict[key]:
                    description = mapping_dict[key][value]
                    transformed[f'{key}_description'] = description
                    
                    if key in description_indices:
                        diff_index = description_indices[key]
                        transformed[f'{key}_token_index'] = diff_index + CLS_TOKEN_OFFSET
                        
            else:
                transformed[f'{key}_description'] = ""
                transformed[f'{key}_token_index'] = -1
        
    return transformed


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return torch.isnan(torch.tensor(value)).item()
    value_str = str(value).strip().lower()
    return value_str in {"", "nan", "nat", "na", "none", "null"}


def _to_float(value: Any):
    if _is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_text(value: Any) -> str:
    if _is_missing(value):
        return ""
    return str(value).strip().lower()


def encode_sex(value: Any) -> int:
    text = _normalize_text(value)
    if text in {"male", "m", "0"}:
        return 0
    if text in {"female", "f", "1"}:
        return 1
    return -1


def encode_kps(value: Any) -> int:
    kps = _to_float(value)
    if kps is None:
        return -1
    return map_kps_category(int(round(kps)))


def encode_age(value: Any) -> int:
    age = _to_float(value)
    if age is None:
        return -1
    return map_age_category(int(round(age)))


def encode_local_clinical(sample: Mapping[str, Any]) -> dict[str, int]:
    return {
        "age": encode_age(sample.get("age")),
        "sex": encode_sex(sample.get("sex")),
        "kps": encode_kps(sample.get("kps")),

    }
