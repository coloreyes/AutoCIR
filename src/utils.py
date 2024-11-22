import torch
from typing import Optional, Tuple, List, Dict, Union
import os
import tqdm
import pickle
import data_utils
from classes import Arch,Captioner
import clip
from check_prompt import CheckModel
from modify_caption import modify_factory
import numpy as np
import file_utils
class FeatureExtractor:
    def __init__(self, clip_model, device: torch.device, arch=None):
        self.clip_model = clip_model
        self.device = device
        self.arch = arch

    @torch.no_grad()
    def extract_image_features(
        self, dataset_name: str, dataset: torch.utils.data.Dataset,
        batch_size: Optional[int] = 4, num_workers: Optional[int] = 0,
        preload: Optional[str] = None, **kwargs
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Extracts image features from a dataset using a CLIP model.
        """
        if preload and os.path.exists(preload):
            return self.load_precomputed_features(preload)
        
        loader = self.create_data_loader(dataset, batch_size, num_workers)
        index_features, index_names, index_ranks, aux_data = self.initialize_variables(dataset_name)

        for batch in tqdm(loader, desc="Extracting features"):
            batch_features, names, index_rank, aux_data = self.process_batch(batch, dataset_name, aux_data)
            index_features.append(batch_features.cpu())
            index_names.extend(names)
            if index_rank is not None:
                index_ranks.extend(index_rank)

        index_features = torch.vstack(index_features)
        self.finalize_aux_data(aux_data, index_features, dataset_name, index_ranks)

        if preload:
            self.save_extracted_features(preload, index_features, index_names, index_ranks, aux_data)

        return index_features, index_names, index_ranks, aux_data

    def load_precomputed_features(self, preload: str) -> Tuple[torch.Tensor, List[str]]:
        print(f'Loading precomputed image features from {preload}!')
        with open(preload, 'rb') as f:
            extracted_data = pickle.load(f)
        return extracted_data['index_features'], extracted_data['index_names'], extracted_data.get('index_ranks', []), extracted_data.get('aux_data', {})

    def create_data_loader(self, dataset: torch.utils.data.Dataset, batch_size: int, num_workers: int):
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=data_utils.collate_fn
        )

    def initialize_variables(self, dataset_name: str):
        index_features, index_names, index_ranks, aux_data = [], [], [], []
        if 'genecis' in dataset_name:
            aux_data = {'ref_features': [], 'instruct_features': []}
        return index_features, index_names, index_ranks, aux_data

    def process_batch(self, batch, dataset_name: str, aux_data: dict):
        if 'genecis' in dataset_name:
            return self.process_genecis_batch(batch, aux_data)
        else:
            return self.process_standard_batch(batch, aux_data)

    def process_genecis_batch(self, batch, aux_data):
        ref_images, n_gallery, _, h, w = batch[0], batch[3], batch[2], batch[4], batch[5]
        images = batch[3].view(-1, 3, h, w)
        names, index_rank = batch[1], batch[4]
        instructions = batch[1]

        images = images.to(self.device)
        batch_features = self.extract_batch_features(images)
        self.process_aux_data(ref_images, instructions, aux_data)

        return batch_features, names, index_rank, aux_data

    def process_standard_batch(self, batch, aux_data):
        images = batch['image'].squeeze(1) or batch['reference_image']
        names = batch['image_name'] or batch['reference_name']
        images = images.to(self.device)
        
        batch_features = self.extract_batch_features(images)
        return batch_features, names, None, aux_data

    def extract_batch_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features for a batch of images using the specified architecture."""
        with torch.cuda.amp.autocast():
            if self.arch == Arch.ViT_B32:
                return self.clip_model.get_image_features(images)
            return self.clip_model.encode_image(images)

    def process_aux_data(self, ref_images: torch.Tensor, instructions: List[str], aux_data: dict) -> None:
        """Process auxiliary data for reference and instruction features."""
        ref_images = ref_images.to(self.device)
        ref_features = self.extract_batch_features(ref_images)
        aux_data['ref_features'].append(ref_features.cpu())

        if hasattr(self.clip_model, 'tokenizer'):
            tokenized_instructions = self.clip_model.tokenizer(instructions, context_length=77).to(self.device)
        else:
            tokenized_instructions = clip.tokenize(instructions, context_length=77).to(self.device)

        instruct_features = self.clip_model.encode_text(tokenized_instructions).cpu()
        aux_data['instruct_features'].append(instruct_features)

    def finalize_aux_data(self, aux_data: dict, index_features: torch.Tensor, dataset_name: str, index_ranks: list) -> None:
        if 'genecis' in dataset_name:
            # Reshape features into gallery-set for GeneCIS-style problems.
            aux_data['ref_features'] = torch.vstack(aux_data['ref_features'])
            aux_data['instruct_features'] = torch.vstack(aux_data['instruct_features'])

    def save_extracted_features(self, preload: str, index_features: torch.Tensor, index_names: List[str], index_ranks: list, aux_data: dict) -> None:
        with open(preload, 'wb') as f:
            pickle.dump({
                'index_features': index_features,
                'index_names': index_names,
                'index_ranks': index_ranks,
                'aux_data': aux_data
            }, f)


@torch.no_grad()
def generate_predictions(
    device: torch.device, 
    dataset_name:str,
    llm_prompt_args: str,
    retrieval:str,
    clip_model,
    query_dataset: torch.utils.data.Dataset,
    preload_dict: Dict[str, Union[str,None]],
    max_check_num, 
    planner,
    dataset_path,
    compute_results_function,
    index_features,
    index_names,
    planner_key,
    corrector_key,
    corrector,
    split,
    **kwargs
) -> Tuple[torch.Tensor, List[str], list]:
    """
    Generates features predictions for the validation set of CIRCO
    """    
    torch.cuda.empty_cache()    
    batch_size = 4
    reload_caption_dict = {}
    print(f'Loading precomputed image captions from {preload_dict["captions"]}!')
    all_captions, relative_captions = [], []
    gt_img_ids, query_ids = [], []
    target_names, reference_names = [], []
    query_loader = torch.utils.data.DataLoader(
        dataset=query_dataset, batch_size=batch_size, num_workers=4, 
        pin_memory=False, collate_fn=data_utils.collate_fn, shuffle=False)            
    query_iterator = tqdm.tqdm(query_loader, position=0, desc='Generating image captions...')
    # load captions
    reload_caption_dict = file_utils.read_captions_file(preload_dict['captions'])         
    for batch in query_iterator:
        
        reference_names.extend(batch['reference_name'])
        if 'fashioniq' not in dataset_name:
            relative_captions.extend(batch['relative_caption'])
        else:
            rel_caps = batch['relative_captions']
            rel_caps = np.array(rel_caps).T.flatten().tolist()
            relative_captions.extend([
                f"{rel_caps[i].strip('.?, ')} and {rel_caps[i + 1].strip('.?, ')}" for i in range(0, len(rel_caps), 2)
                ])
                        
        if 'target_name' in batch:
            target_names.extend(batch['target_name'])
    
        gt_key = 'gt_img_ids'
        if 'group_members' in batch:
            gt_key = 'group_members'
        if gt_key in batch:
            gt_img_ids.extend(np.array(batch[gt_key]).T.tolist())

        query_key = 'query_id'
        if 'pair_id' in batch:
            query_key = 'pair_id'
        if query_key in batch:
            query_ids.extend(batch[query_key])
        # match captions and target images
        for target_name in batch['reference_name']:
            all_captions.append(reload_caption_dict[target_name])
    ### Modify Captions using LLM.
    suggestions = [''] * len(all_captions)
    if preload_dict['mods'] is None or not os.path.exists(preload_dict['mods']):
        modified_captions = LLM_modify_caption(planner,preload_dict,llm_prompt_args,all_captions,relative_captions,planner_key,device)
        if preload_dict['mods'] is not None:
            file_utils.write_modified_captions_file(preload_dict['mods'],reference_names=reference_names,modified_captions=modified_captions)
    else:
        print(f'Loading precomputed caption modifiers from {preload_dict["mods"]}!')
        modified_captions = file_utils.read_modified_captions_file(path=preload_dict["mods"])

    predicted_features = text_encoding(device, clip_model,modified_captions, batch_size=batch_size, mode=retrieval)
    if 'fashion' in dataset_name:
        _,sorted_index_names=compute_results_function(device=device,predicted_features=predicted_features,target_names=target_names,index_features=index_features,index_names=index_names,dataset_name=dataset_name,dataset_path=dataset_path)    
    elif 'cirr' in dataset_name:
        _,sorted_index_names=compute_results_function(device=device,predicted_features=predicted_features,reference_names=reference_names,targets=gt_img_ids,target_names=target_names,index_features=index_features,index_names=index_names,query_ids=query_ids,dataset_name=dataset_name,dataset_path=dataset_path,split=split)   
    else:
        _,sorted_index_names=compute_results_function(device=device,predicted_features=predicted_features,targets=gt_img_ids,target_names=target_names,index_features=index_features,index_names=index_names,query_ids=query_ids,dataset_name=dataset_name,dataset_path=dataset_path,split=split)    
    top_names = sorted_index_names[:,:10]
    file_utils.write_top_file(f"{dataset_path}/results/top_rank_loop_0.json",reference_names = reference_names,top_names = top_names)
    check_index = [True] * len(modified_captions)
    for i in range(max_check_num):
        print(f"{i} times:Start check modified captions and generate suggestions.")
        # load top image caption
        top_captions = []
        for names in top_names:
            top_captions_list = []
            for name in names:
                top_captions_list.append(reload_caption_dict[name])
            top_captions.append(top_captions_list)        
        print(f"Start check modified captions...")
        suggestions = check_prompt(corrector,top_captions,all_captions,relative_captions,corrector_key,modified_captions,device=device,check_index=check_index)
        file_utils.write_suggestions_file(path=f"{dataset_path}/results/suggestions/suggestions_loop_{i+1}.json",reference_names=reference_names,suggestions=suggestions)
        print(f"{i} times:Start remodified captions with suggestions.")
        modified_captions,check_index,input_suggestions = LLM_remodify_caption(LLM_model_name=planner,llm_prompt_args=llm_prompt_args,last_captions=modified_captions,all_captions=all_captions,
                                                 relative_captions=relative_captions,suggestions=suggestions,planner_key=planner_key,device=device,check_index=check_index)
        file_utils.write_suggestions_file(path=f"{dataset_path}/results/suggestions/input_suggestions_loop_{i+1}.json",reference_names=reference_names,suggestions = input_suggestions)
        file_utils.write_modified_captions_file(path=f'{dataset_path}/results/new_captions/suggestions_modified_captions_loop_{i+1}.json',reference_names=reference_names,modified_captions=modified_captions)
        predicted_features = text_encoding(device, clip_model, modified_captions, batch_size=batch_size, mode=retrieval)   
        if 'fashion' in dataset_name:
            _,sorted_index_names=compute_results_function(device=device,predicted_features=predicted_features,target_names=target_names,index_features=index_features,index_names=index_names,dataset_name=dataset_name,dataset_path=dataset_path)    
        elif 'cirr' in dataset_name:
            _,sorted_index_names=compute_results_function(device=device,predicted_features=predicted_features,reference_names=reference_names,targets=gt_img_ids,target_names=target_names,index_features=index_features,index_names=index_names,query_ids=query_ids,dataset_name=dataset_name,dataset_path=dataset_path,split=split,loop=i+1)   
        elif 'circo' in dataset_name:
            _,sorted_index_names=compute_results_function(device=device,predicted_features=predicted_features,targets=gt_img_ids,target_names=target_names,index_features=index_features,index_names=index_names,query_ids=query_ids,dataset_name=dataset_name,dataset_path=dataset_path,split=split,loop=i+1)    
        top_names = sorted_index_names[:,:10]
        file_utils.write_top_file(f"{dataset_path}/results/top_rank_loop_{i+1}.json",reference_names = reference_names,top_names = top_names)
    return {
        'predicted_features': predicted_features, 
        'target_names': target_names, 
        'targets': gt_img_ids,
        'reference_names': reference_names,
        'query_ids': query_ids,
        'start_captions': all_captions,
        'modified_captions': modified_captions,
        'instructions': relative_captions
    }

def extract_suggestions(original_suggestion):
    suggestion_split = original_suggestion.split('\n')
    total_suggestion = ""
    suggestion_flag = False
    for line in suggestion_split:
        if suggestion_flag:
            total_suggestion = total_suggestion+line
        elif 'suggestions' in line or 'Suggestions' in line:
            suggestion_flag = True
            total_suggestion = total_suggestion + line.split('uggestions')[1].strip()
    return total_suggestion        
    
def LLM_remodify_caption(LLM_model_name,llm_prompt_args,last_captions,all_captions,relative_captions,suggestions,planner_key,device:torch.device,check_index):
    LLM_factory = modify_factory(LLM_model_name,device=device)
    modified_captions = []
    base_prompt = eval(llm_prompt_args)
    input_suggestions = []
    for i in tqdm.trange(len(all_captions), position=1, desc=f'Remodifying captions with LLM...'):
        if "Good retrieval, no more loops needed" in suggestions[i]:
            input_suggestions.append("Good retrieval, no more loops needed")
            modified_captions.append(last_captions[i])
            check_index[i] = False
            continue
        total_suggestion = extract_suggestions(suggestions[i])
        if not total_suggestion:
            total_suggestion = suggestions[i]
        input_suggestions.append(total_suggestion)
        final_prompt = f'''Assume you are an experienced composed image retrieval expert, skilled at precisely generating new image descriptions based on a reference image's description and the user's modification instructions. 
        {base_prompt}
        Image Content: {all_captions[i]}
        Instruction: {relative_captions[i]}
        The caption you generated last time: {last_captions[i]}
        However, the caption you generated is not entirely appropriate, as it does not fully integrate the content of the reference image and the user's modification instructions, resulting in inaccurate retrieval results. 
        After comparing the retrieved images, the specific modification suggestions are as follows:{total_suggestion}
        Please think step by step and make appropriate adjustments to the caption you generated last time.
        Fully understand the reference image and the user's modification requests to create a caption that better aligns with the modifications, ensuring more accurate compositional retrieval.
        '''
        resp = LLM_factory.modify_function(final_prompt, planner_key,max_length = 800)
        if 'Error' in resp:
            modified_captions.append(last_captions[i])
            continue
        ## extract edited description
        resp = resp.split('\n')
        description = ""
        aug = False
        for line in resp:                    
            if line.strip().startswith('Edited Description:'):
                description = line.split(':')[1].strip()
                if description == "":
                    modified_captions.append(last_captions[i])
                else:
                    modified_captions.append(description)
                aug = True
                break
        if not aug:
            modified_captions.append(last_captions[i])   
        check_index[i] = True
    return modified_captions,check_index,input_suggestions

def LLM_modify_caption(LLM_model_name,preload_dict,llm_prompt_args,all_captions,relative_captions,planner_key,device:torch.device): 
    LLM_factory = modify_factory(LLM_model_name,device)
    modified_captions = []
    base_prompt = eval(llm_prompt_args)
    for i in tqdm.trange(len(all_captions), position=1, desc=f'Modifying captions with LLM...'):
        instruction = relative_captions[i]
        img_caption = all_captions[i]
        final_prompt = base_prompt + '\n' + "Image Content: " + img_caption
        final_prompt = final_prompt + '\n' + 'Instruction: '+ instruction
        resp = LLM_factory.modify_function(final_prompt, planner_key,max_length = 800)

        ## extract edited description
        resp = resp.split('\n')
        description = ""
        aug = False
        for line in resp:                    
            if line.strip().startswith('Edited Description:'):
                description = line.split(':')[1].strip()
                if description == "":
                    modified_captions.append(relative_captions[i])
                else:
                    modified_captions.append(description)
                aug = True
                break
        if not aug:
            modified_captions.append(relative_captions[i])   
    return modified_captions

def check_prompt(model_name,top_captions,all_captions,relative_captions,corrector_key,modified_captions,device:torch.device,check_index):
    prompt_checker = CheckModel.from_string(model_name)
    suggestions = []
    if prompt_checker is CheckModel.gpt_4o_mini:
        for i in tqdm.trange(len(relative_captions), position=1, desc=f'Generate suggestions with LLM...'):
            if not check_index[i]:
                suggestions.append("Good retrieval, no more loops needed")
                continue
            else:
                suggestion = prompt_checker.chat_gpt_4o_mini(top_captions[i],all_captions[i],relative_captions[i],corrector_key,modified_captions[i])
                if suggestion is None:
                    suggestion = ''
                suggestions.append(suggestion)
    elif prompt_checker is CheckModel.qwen_turbo:
        for i in tqdm.trange(len(relative_captions), position=1, desc=f'Generate suggestions with LLM...'):
            if not check_index[i]:
                suggestions.append("Good retrieval, no more loops needed")
                continue
            else:
                suggestion = prompt_checker.check_qwen_turbo(top_captions[i],all_captions[i],relative_captions[i],corrector_key,modified_captions[i])
                if suggestion is None:
                    suggestion = ''
                suggestions.append(suggestion)
    return suggestions

def text_encoding(device, clip_model,input_captions, batch_size=32, mode='default'):
    n_iter = int(np.ceil(len(input_captions)/batch_size))
    predicted_features = []
        
    for i in tqdm.trange(n_iter, position=0, desc='Encoding captions...'):
        captions_to_use = input_captions[i*batch_size:(i+1)*batch_size]
        if hasattr(clip_model, 'tokenizer'):
            tokenized_input_captions = clip_model.tokenizer(captions_to_use, context_length=77).to(device)
        else:
            tokenized_input_captions = clip.tokenize(captions_to_use, context_length=77, truncate=True).to(device)
        clip_text_features = clip_model.encode_text(tokenized_input_captions)
        predicted_features.append(clip_text_features)
    predicted_features = torch.vstack(predicted_features)        
        
    return torch.nn.functional.normalize(predicted_features, dim=-1)
