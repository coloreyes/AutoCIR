import pandas as pd
import os
import json
import csv
def write_top_file(path:str,reference_names,top_names):
    with open(path, 'w',encoding='utf-8') as f:
        json.dump(
            [
                {
                    "image_index": reference_names[count],
                    "top_names": top_names.tolist()[count]
                }
                for count in range(len(reference_names))
            ],
            f,
            indent=6)

def write_suggestions_file(path:str,reference_names,suggestions):
    with open(path, 'w',encoding='utf-8') as f:
        json.dump(
            [
                {
                    "image_index": reference_names[count],
                    "suggestion": suggestions[count]
                }
                for count in range(len(reference_names))
            ],
            f,
            indent=6)
    return

def write_modified_captions_file(path:str,reference_names,modified_captions):
    with open(path, 'w',encoding='utf-8') as f:
        json.dump(
            [
                {
                    "image_index": reference_names[count],
                    "modified_caption": modified_captions[count]
                }
                for count in range(len(reference_names))
            ],
            f,
            indent=6)
    return

def read_suggestions_file(path:str):
    suggestions = json.load(open(path,'r',encoding='utf-8'))
    if isinstance(suggestions,dict):
        suggestions = [item["suggestion"] for item in suggestions]
    elif isinstance(suggestions, list):
        suggestions_list = [item.get("suggestion") for item in suggestions if "suggestion" in item]   
        if len(suggestions_list) != 0:
            suggestions = suggestions_list
    elif isinstance(suggestions,set):
        suggestions = list(suggestions)
    return suggestions

def read_modified_captions_file(path:str):
    modified_captions = json.load(open(path, 'r',encoding='utf-8'))
    if isinstance(modified_captions,dict):
        modified_captions = list(modified_captions.values())
    elif isinstance(modified_captions, list):
        modified_list = [item.get("modified_caption") for item in modified_captions if "modified_caption" in item]
        if len(modified_list) != 0:
            modified_captions = modified_list
    elif isinstance(modified_captions,set):
        modified_captions = list(modified_captions)
    return modified_captions

def init_folder(dataset_path,task):
    os.makedirs(f'{dataset_path}/task', exist_ok=True)
    os.makedirs(f'{dataset_path}/task/{task}', exist_ok=True)
    os.makedirs(f'{dataset_path}/task/{task}/suggestions', exist_ok=True)
    os.makedirs(f'{dataset_path}/task/{task}/modified_captions', exist_ok=True)
    os.makedirs(f'{dataset_path}/task/{task}/new_captions', exist_ok=True)