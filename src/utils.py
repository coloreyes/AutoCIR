import json
import os
from typing import Optional, Tuple, List, Dict, Union
import datetime
import pandas as pd
import clip
import numpy as np
from modify_caption import ModelFactory
import pickle
import torch
import tqdm
from transformers import CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipForConditionalGeneration
import data_utils
import prompts
from classes import Arch
from check_prompt import CheckModel, ModelHandler
import csv
import file_utils
if torch.cuda.is_available():
    dtype = torch.float16
else:
    dtype = torch.float32
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
def extract_image_features(device: torch.device, dataset_name: str, dataset: torch.utils.data.Dataset, clip_model, batch_size: Optional[int] = 4,
                           num_workers: Optional[int] = 0, preload: str=None,arch= None, **kwargs) -> Tuple[torch.Tensor, List[str]]:
    """
    Extracts image features from a dataset using a CLIP model.
    """
    if preload is not None and os.path.exists(preload):
        print(f'Loading precomputed image features from {preload}!')
        extracted_data = pickle.load(open(preload, 'rb'))
        index_features, index_names = extracted_data['index_features'], extracted_data['index_names']
        index_ranks = [] if 'index_ranks' not in extracted_data else extracted_data['index_ranks']        
        aux_data = {} if 'aux_data' not in extracted_data else extracted_data['aux_data']
    else:
        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                            num_workers=num_workers, pin_memory=True, collate_fn=data_utils.collate_fn)

        index_features, index_names, index_ranks, aux_data = [], [], [], []
        if 'genecis' in dataset_name:
            aux_data = {'ref_features': [], 'instruct_features': []}
        try:
            print(f"Extracting image features {dataset.__class__.__name__} - {dataset.split}")
        except Exception as e:
            pass

        # Extract features    
        index_rank = None
        for batch in tqdm.tqdm(loader):
            if 'genecis' in dataset_name:
                _, n_gallery, _, h, w = batch[3].size()
                images = batch[3].view(-1, 3, h, w)
                names, index_rank = batch[1], batch[4]
                ref_images = batch[0]
                instructions = batch[1]
            else:
                images = batch['image'].squeeze(1)
                names = batch['image_name']
                if images is None: images = batch['reference_image']
                if names is None: names = batch['reference_name']

            images = images.to(device)
            with torch.no_grad(),torch.amp.autocast(device_type="cuda"):
                if arch in [Arch.ViT_B_32_openai, Arch.ViT_B_16_openai, Arch.ViT_L_14_openai]:
                    batch_features = clip_model.get_image_features(images)
                else:
                    batch_features = clip_model.encode_image(images)
                index_features.append(batch_features.cpu())
                index_names.extend(names)
                if index_rank is not None:
                    index_ranks.extend(index_rank)
                if len(aux_data):
                    if arch in [Arch.ViT_B_32_openai, Arch.ViT_B_16_openai, Arch.ViT_L_14_openai]:
                        aux_data['ref_features'].append(clip_model.get_image_features(ref_images.to(device)).cpu())
                    else:
                        aux_data['ref_features'].append(clip_model.encode_image(ref_images.to(device)).cpu())
                    if hasattr(clip_model, 'tokenizer'):
                        aux_data['instruct_features'].append(clip_model.encode_text(clip_model.tokenizer(instructions, context_length=77).to(device)).cpu())
                    else:
                        aux_data['instruct_features'].append(clip_model.encode_text(clip.tokenize(instructions, context_length=77).to(device)).cpu())
        
        index_features = torch.vstack(index_features)
        
        if 'genecis' in dataset_name:
            # Reshape features into gallery-set for GeneCIS-style problems.
            index_features = index_features.view(-1, n_gallery, batch_features.size()[-1])
            index_ranks = torch.stack(index_ranks)
            aux_data['ref_features'] = torch.vstack(aux_data['ref_features'])
            aux_data['instruct_features'] = torch.vstack(aux_data['instruct_features'])
            
        if preload is not None:
            pickle.dump({'index_features': index_features, 'index_names': index_names, 'index_ranks': index_ranks, 'aux_data': aux_data}, open(preload, 'wb'))
            print(f"Save image feathers in {preload}")
    return index_features, index_names, index_ranks, aux_data

@torch.no_grad()
def generate_predictions(
    device: torch.device, dataset_name:str,blip_prompt_args:str,llm_prompt_args: str,retrieval:str,clip_model: clip.model.CLIP, blip_model: callable,
    query_dataset: torch.utils.data.Dataset, preload_dict: Dict[str, Union[str,None]],blip_transform,processor,LLM_model_name,arch:Arch,max_check_num, 
    blip,Check_LLM_model_name,dataset_path,compute_results_function,index_features,index_names,openai_key,task,split,**kwargs
) -> Tuple[torch.Tensor, List[str], list]:
    """
    Generates features predictions
    """    
    ### Generate BLIP Image Captions.
    torch.cuda.empty_cache()    
    batch_size = 4
    reload_caption_dict = {}
    if preload_dict['captions'] is None or not os.path.exists(preload_dict['captions']):
        all_captions, relative_captions = [], []
        gt_img_ids, query_ids = [], []
        target_names, reference_names = [], []
        
        query_loader = torch.utils.data.DataLoader(
            dataset=query_dataset, batch_size=batch_size, num_workers=4, 
            pin_memory=False, collate_fn=data_utils.collate_fn, shuffle=False)            
        query_iterator = tqdm.tqdm(query_loader, position=0, desc='Generating image captions...')
        
        for batch in query_iterator:
            
            if 'genecis' in dataset_name:
                blip_image = batch[2].to(device)
                relative_captions.extend(batch[1])
            else:
                blip_image = batch['blip_ref_img'][0].to(device)
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
                        
            query_iterator.set_postfix_str(f'Shape: {blip_image.size()}')
                
            captions = []
            blip_prompt = eval(blip_prompt_args)
            for i in tqdm.trange(blip_image.size(0), position=1, desc='Iterating over batch', leave=False):
                img = blip_image[i].unsqueeze(0)
                input_ids = blip_transform(text=blip_prompt, return_tensors="pt")['input_ids'].to(device)
                caption = blip_model.generate(pixel_values = img, input_ids = input_ids)
                captions.append(blip_transform.decode(caption[0], skip_special_tokens=True))
            all_captions.extend(captions)

        if preload_dict['captions'] is not None:
            res_dict = {
                'all_captions': all_captions, 
                'gt_img_ids': gt_img_ids, 
                'relative_captions': relative_captions,
                'target_names': target_names,
                'reference_names': reference_names,
                'query_ids': query_ids
            }
            pickle.dump(res_dict, open(preload_dict['captions'], 'wb'))
    else:
        print(f'Loading precomputed image captions from {preload_dict["captions"]}!')
        all_captions, relative_captions = [], []
        gt_img_ids, query_ids = [], []
        target_names, reference_names = [], []
        query_loader = torch.utils.data.DataLoader(
            dataset=query_dataset, batch_size=batch_size, num_workers=4, 
            pin_memory=False, collate_fn=data_utils.collate_fn, shuffle=False)            
        query_iterator = tqdm.tqdm(query_loader, position=0, desc='Loading image captions...')
        # load captions
        with open(preload_dict['captions'], 'r', encoding='utf-8') as blip_captions:
            reader = csv.reader(blip_captions)
            next(reader)
            reload_caption_dict = {caption[0]: caption[1] for caption in reader}         
        for batch in query_iterator:
            
            if 'genecis' in dataset_name:
                blip_image = batch[2].to(device)
                relative_captions.extend(batch[1])
            else:
                blip_image = batch['blip_ref_img']['pixel_values'][0].to(device)
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
            query_iterator.set_postfix_str(f'Shape: {blip_image.size()}')
    ### Modify Captions using LLM.
    suggestions = [''] * len(all_captions)
    if preload_dict['mods'] is None or not os.path.exists(preload_dict['mods']):
        modified_captions = LLM_modify_caption(LLM_model_name,preload_dict,llm_prompt_args,all_captions,relative_captions,openai_key,device)
        if preload_dict['mods'] is not None:
            file_utils.write_modified_captions_file(preload_dict['mods'],reference_names=reference_names,modified_captions=modified_captions)
    else:
        print(f'Loading precomputed caption modifiers from {preload_dict["mods"]}!')
        modified_captions = file_utils.read_modified_captions_file(path=preload_dict["mods"])

    predicted_features = text_encoding(device, clip_model,processor, modified_captions, batch_size=batch_size, mode=retrieval,arch=arch)
    if 'genecis' in dataset_name:
        output_metrics,sorted_index_names=compute_results_function(device=device,predicted_features=predicted_features,target_names=target_names,index_features=index_features,index_names=index_names)    
    elif 'fashion' in dataset_name:
        output_metrics,sorted_index_names=compute_results_function(device=device,predicted_features=predicted_features,target_names=target_names,index_features=index_features,index_names=index_names,dataset_name=dataset_name,dataset_path=dataset_path,task=task)    
    elif 'cirr' in dataset_name:
        output_metrics,sorted_index_names=compute_results_function(device=device,predicted_features=predicted_features,reference_names=reference_names,targets=gt_img_ids,target_names=target_names,index_features=index_features,index_names=index_names,query_ids=query_ids,dataset_name=dataset_name,dataset_path=dataset_path,task=task,split=split)   
    else:
        output_metrics,sorted_index_names=compute_results_function(device=device,predicted_features=predicted_features,targets=gt_img_ids,target_names=target_names,index_features=index_features,index_names=index_names,query_ids=query_ids,dataset_name=dataset_name,dataset_path=dataset_path,task=task,split=split)    
    top_names = sorted_index_names[:,:10]
    file_utils.write_top_file(f"{dataset_path}/task/{task}/top_rank_loop_0_{task}_{get_time()}.json",reference_names = reference_names,top_names = top_names)
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
        if preload_dict['suggestions'] is None or not os.path.exists(preload_dict['suggestions']):
            suggestions = check_prompt(Check_LLM_model_name,top_captions,all_captions,relative_captions,openai_key,modified_captions,device=device,check_index=check_index)
            file_utils.write_suggestions_file(f'{dataset_path}/task/{task}/suggestions/{blip}_{LLM_model_name}_suggestions_loop_{i+1}_{task}_{get_time()}.json',reference_names=reference_names,suggestions=suggestions)
        else:
            suggestions = json.load(open(preload_dict['suggestions'], 'r'))
        print(f"{i} times:Start remodified captions with suggestions.")
        modified_captions,check_index,input_suggestions = LLM_remodify_caption(LLM_model_name=LLM_model_name,llm_prompt_args=llm_prompt_args,last_captions=modified_captions,all_captions=all_captions,
                                                 relative_captions=relative_captions,suggestions=suggestions,openai_key=openai_key,device=device,check_index=check_index)
        file_utils.write_suggestions_file(path=f"{dataset_path}/task/{task}/suggestions/{blip}_{Check_LLM_model_name}_input_suggestions_loop_{i+1}_{task}_{get_time()}.json",reference_names=reference_names,suggestions = input_suggestions)
        file_utils.write_modified_captions_file(f'{dataset_path}/task/{task}/new_captions/{blip}_{Check_LLM_model_name}_suggestions_modified_captions_loop_{i+1}_{task}_{get_time()}.json',reference_names=reference_names,modified_captions=modified_captions)
        predicted_features = text_encoding(device, clip_model,processor, modified_captions, batch_size=batch_size, mode=retrieval,arch=arch)   
        if 'genecis' in dataset_name:
            output_metrics,sorted_index_names=compute_results_function(device=device,predicted_features=predicted_features,target_names=target_names,index_features=index_features,index_names=index_names)    
        elif 'fashion' in dataset_name:
            output_metrics,sorted_index_names=compute_results_function(device=device,predicted_features=predicted_features,target_names=target_names,index_features=index_features,index_names=index_names,dataset_name=dataset_name,dataset_path=dataset_path,task=task)    
        elif 'cirr' in dataset_name:
            output_metrics,sorted_index_names=compute_results_function(device=device,predicted_features=predicted_features,reference_names=reference_names,targets=gt_img_ids,target_names=target_names,index_features=index_features,index_names=index_names,query_ids=query_ids,dataset_name=dataset_name,dataset_path=dataset_path,task=task,split=split,loop=i+1)   
        elif 'circo' in dataset_name:
            output_metrics,sorted_index_names=compute_results_function(device=device,predicted_features=predicted_features,targets=gt_img_ids,target_names=target_names,index_features=index_features,index_names=index_names,query_ids=query_ids,dataset_name=dataset_name,dataset_path=dataset_path,task=task,split=split,loop=i+1)    
        top_names = sorted_index_names[:,:10]
        file_utils.write_top_file(f"{dataset_path}/task/{task}/top_rank_loop_{i+1}_{task}_{get_time()}.json",reference_names = reference_names,top_names = top_names)

        ## Perform text-to-image retrieval based on the modified captions.

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

def get_time()->str:
    now = datetime.datetime.now()
    return now.strftime("%Y.%m.%d-%H_%M_%S")

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
    
def LLM_remodify_caption(LLM_model_name,llm_prompt_args,last_captions,all_captions,relative_captions,suggestions,openai_key,device:torch.device,check_index):
    LLM_factory = ModelFactory(LLM_model_name,device=device)
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
        resp = LLM_factory.modify_function(final_prompt, openai_key,max_length = 800)
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

def LLM_modify_caption(LLM_model_name,preload_dict,llm_prompt_args,all_captions,relative_captions,openai_key,device:torch.device): 
    LLM_factory = ModelFactory(LLM_model_name,device)
    modified_captions = []
    base_prompt = eval(llm_prompt_args)
    for i in tqdm.trange(len(all_captions), position=1, desc=f'Modifying captions with LLM...'):
        instruction = relative_captions[i]
        img_caption = all_captions[i]
        final_prompt = base_prompt + '\n' + "Image Content: " + img_caption
        final_prompt = final_prompt + '\n' + 'Instruction: '+ instruction
        resp = LLM_factory.modify_function(final_prompt, openai_key,max_length = 800)

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

def check_prompt(model_name,top_captions,all_captions,relative_captions,openai_key,modified_captions,device:torch.device,check_index):
    model_handler = ModelHandler(model_type=model_name, device=device, openai_key=openai_key)

    suggestions = []
    for i in tqdm.trange(len(relative_captions), position=1, desc='Generate suggestions with LLM...'):
        if not check_index[i]:
            suggestions.append("Good retrieval, no more loops needed")
            continue
        suggestion = model_handler.chat_function(top_captions[i], all_captions[i], relative_captions[i], max_length=800)
        if suggestion is None:
            suggestion = ''
        suggestions.append(suggestion)
    return suggestions

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    
    
def get_recall(indices, targets): #recall --> wether next item in session is within top K recommended items or not
    """
    Code adapted from: https://github.com/hungthanhpham94/GRU4REC-pytorch/blob/master/lib/metric.py
    Calculates the recall score for the given predictions and targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B) or (BxN): torch.LongTensor. actual target indices.
    Returns:
        recall (float): the recall score
    """

    if len(targets.size()) == 1:
        # One hot label branch
        targets = targets.view(-1, 1).expand_as(indices)
        hits = (targets == indices).nonzero()
        if len(hits) == 0: return 0
        n_hits = (targets == indices).nonzero()[:, :-1].size(0)
        recall = float(n_hits) / targets.size(0)
        return recall
    else:        
        # Multi hot label branch
        recall = []
        for preds, gt in zip(indices, targets):            
            max_val = torch.max(torch.cat([preds, gt])).int().item()
            preds_binary = torch.zeros((max_val + 1,), device=preds.device, dtype=torch.float32).scatter_(0, preds, 1)
            gt_binary = torch.zeros((max_val + 1,), device=gt.device, dtype=torch.float32).scatter_(0, gt.long(), 1)
            success = (preds_binary * gt_binary).sum() > 0
            recall.append(int(success))        
        return torch.Tensor(recall).float().mean()
            
@torch.no_grad()            
def evaluate_genecis(device: torch.device, llm_prompt_args: str, clip_model: clip.model.CLIP, blip_model: callable, query_dataset: torch.utils.data.Dataset, preload_dict: Dict[str, Union[str,None]],LLM_model_name, topk: List[int] = [1,2,3], batch_size: int=32, **kwargs):
    val_loader = torch.utils.data.DataLoader(
        dataset=query_dataset, batch_size=batch_size, num_workers=8, 
        pin_memory=False, collate_fn=data_utils.collate_fn, shuffle=False)            
    query_iterator = tqdm.tqdm(val_loader, position=0, desc='Generating image captions...')

    meters = {k: AverageMeter() for k in topk}
    sims_to_save = []
    
    with torch.no_grad():
        LLM_factory = ModelFactory(LLM_model_name)
        for batch in query_iterator:
            ref_img = batch[0].to(device)
            original_caption = batch[1]
            caption = clip.tokenize(batch[1],context_length=77).to(device)
            blip_ref_img = batch[2].to(device)
            gallery_set = batch[3].to(device)
            target_rank = batch[4].to(device)

            bsz, n_gallery, _, h, w = gallery_set.size()
            imgs_ = torch.cat([ref_img,gallery_set.view(-1,3,h,w)],dim=0)
            
            # CLIP Encoding
            all_img_feats = clip_model.encode_image(imgs_).float()
            caption_feats = clip_model.encode_text(caption).float()

            # BLIP Captioning
            captions = []
            for i in tqdm.trange(bsz, position=1, desc=f'Captioning image with BLIP', leave=False):
                caption = blip_model.generate({"image": blip_ref_img[i].unsqueeze(0), "prompt": prompts.blip_prompt})
                captions.append(caption[0])
            
            modified_captions = []
            base_prompt = eval(llm_prompt_args)

            # LLM Caption Updates
            for i in tqdm.trange(len(captions), position=1, desc=f'Modifying captions with LLM', leave=False):
                instruction = original_caption[i]
                img_caption = captions[i]
                final_prompt = base_prompt + '\n' + "Image Content: " + img_caption
                final_prompt = final_prompt + '\n' + 'Instruction: '+ instruction
                resp = LLM_factory.modify_function(final_prompt, max_length = 800)

                resp = resp.split('\n')

                description = ""
                for line in resp:                        
                    if line.startswith('Edited Description:'):
                        description = line.split(':')[1].strip()
                        modified_captions.append(description)
                        break
                if description == "":
                    modified_captions.append(original_caption[i])

            predicted_feature = torch.nn.functional.normalize(clip_model.encode_text(clip.tokenize(modified_captions,context_length=77).to(device)))
            
            ##### COMPUTE RECALL - Base Evaluation.
            ref_feats, gallery_feats = all_img_feats.split((bsz,bsz*n_gallery),dim=0)
            gallery_feats = gallery_feats.view(bsz,n_gallery,-1)
            gallery_feats = torch.nn.functional.normalize(gallery_feats, dim=-1)

            #combined_feats = F.normalize(ref_feats + caption_feats)
            combined_feats = predicted_feature
            # Compute similarity
            similarities = combined_feats[:, None, :] * gallery_feats       # B x N x D
            similarities = similarities.sum(dim=-1)                         # B x N

            # Sort the similarities in ascending order (closest example is the predicted sample)
            _, sort_idxs = similarities.sort(dim=-1, descending=True)                   # B x N

            # Compute recall at K
            for k in topk:
                recall_k = get_recall(sort_idxs[:, :k], target_rank)
                meters[k].update(recall_k, bsz)
            sims_to_save.append(similarities.cpu())

        # Print results
        print_str = '\n'.join([f'Recall @ {k} = {v.avg:.4f}' for k, v in meters.items()])
        print(print_str)

        return meters
    
def text_encoding(device, clip_model, clip_processor,input_captions, arch:Arch,batch_size=32, mode='default'):
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
    

prompt_ensemble = [
    'A bad photo of a {}',
    'A photo of many {}',
    'A sculpture of a {}',
    'A photo of the hard to see {}',
    'A low resolution photo of the {}',
    'A rendering of a {}',
    'Graffiti of a {}',
    'A bad photo of the {}',
    'A cropped photo of the {}',
    'A tattoo of a {}',
    'The embroidered {}',
    'A photo of a hard to see {}',
    'A bright photo of a {}',
    'A photo of a clean {}',
    'A photo of a dirty {}',
    'A dark photo of the {}',
    'A drawing of a {}',
    'A photo of my {}',
    'The plastic {}',
    'A photo of the cool {}',
    'A close-up photo of a {}',
    'A black and white photo of the {}',
    'A painting of the {}',
    'A painting of a {}',
    'A pixelated photo of the {}',
    'A sculpture of the {}',
    'A bright photo of the {}',
    'A cropped photo of a {}',
    'A plastic {}',
    'A photo of the dirty {}',
    'A jpeg corrupted photo of a {}',
    'A blurry photo of the {}',
    'A photo of the {}',
    'A good photo of the {}',
    'A rendering of the {}',
    'A {} in a video game',
    'A photo of one {}',
    'A doodle of a {}',
    'A close-up photo of the {}',
    'A photo of a {}',
    'The origami {}',
    'The {} in a video game',
    'A sketch of a {}',
    'A doodle of the {}',
    'A origami {}',
    'A low resolution photo of a {}',
    'The toy {}',
    'A rendition of the {}',
    'A photo of the clean {}',
    'A photo of a large {}',
    'A rendition of a {}',
    'A photo of a nice {}',
    'A photo of a weird {}',
    'A blurry photo of a {}',
    'A cartoon {}',
    'Art of a {}',
    'A sketch of the {}',
    'A embroidered {}',
    'A pixelated photo of a {}',
    'Itap of the {}',
    'A jpeg corrupted photo of the {}',
    'A good photo of a {}',
    'A plushie {}',
    'A photo of the nice {}',
    'A photo of the small {}',
    'A photo of the weird {}',
    'The cartoon {}',
    'Art of the {}',
    'A drawing of the {}',
    'A photo of the large {}',
    'A black and white photo of a {}',
    'The plushie {}',
    'A dark photo of a {}',
    'Itap of a {}',
    'Graffiti of the {}',
    'A toy {}',
    'Itap of my {}',
    'A photo of a cool {}',
    'A photo of a small {}',
    'A tattoo of the {}',
]
