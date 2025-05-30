from argparse import Namespace
import torch
import os
import data_utils
import utils
from datasets import COCOValSubset,FashionIQDataset,CIRRDataset,VAWValSubset,CIRCODataset
import compute_results
import clip
from transformers import CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipForConditionalGeneration
import termcolor
from classes import Captioner,Arch
import wandb
import file_utils
class Experiment:
    def __init__(self,args:Namespace) -> None:
        super().__init__()
        for arg in vars(args):
            value_arg = getattr(args, arg)
            self.__setattr__(arg, value_arg)
        self.device = torch.device(f'cuda:{self.device}' if torch.cuda.is_available() else 'cpu')
        
    def run(self):
        clip_model,clip_processor = self.load_Clip_model()
        blip_model,blip_processor = self.load_Blip_model()
        target_datasets,query_datasets,compute_results_function,pairings = self.load_dataset(clip_processor,blip_processor)
        self.evaluate(query_datasets, target_datasets, pairings,compute_results_function,clip_model,clip_processor,blip_model,blip_processor)
        print("Finish.")
        return
    
    def data_preprocessing(self):
        return
    def load_Clip_model(self):
        clip_device = self.device
        self.arch = Arch.from_string(self.clip)
        clip_model,clip_processor = self.arch.load_model_and_preprocess(device=clip_device)
        print('Done.')
        return clip_model,clip_processor
    
    def load_Blip_model(self):
        blip_device = self.device
        captioner = Captioner.from_string(self.blip)
        blip_model,blip_processor = captioner.load_model_and_preprocess(blip_device)
        return blip_model,blip_processor
    
    def load_dataset(self,clip_processor:CLIPProcessor,blip_processor:BlipProcessor):
        ### Load Evaluation Datasets.
        target_datasets, query_datasets, pairings = [], [], []
        
        if 'fashioniq' in self.dataset.lower():
            dress_type = self.dataset.split('_')[-1]
            target_datasets.append(FashionIQDataset(self.dataset_path, self.split, [dress_type], 'classic', clip_processor, blip_transform=blip_processor))
            query_datasets.append(FashionIQDataset(self.dataset_path, self.split, [dress_type], 'relative', clip_processor, blip_transform=blip_processor))
            pairings.append(dress_type)
            compute_results_function = compute_results.fiq
        
        elif self.dataset.lower() == 'cirr':
            split = 'test1' if self.split == 'test' else self.split
            target_datasets.append(CIRRDataset(self.dataset_path, split, 'classic', clip_processor, blip_transform=blip_processor))
            query_datasets.append(CIRRDataset(self.dataset_path, split, 'relative', clip_processor, blip_transform=blip_processor))
            compute_results_function = compute_results.cirr
            pairings.append('default')
            
        elif self.dataset.lower() == 'circo':
            target_datasets.append(CIRCODataset(self.dataset_path, self.split, 'classic', clip_processor, blip_transform=blip_processor,arch=self.arch))
            query_datasets.append(CIRCODataset(self.dataset_path, self.split, 'relative', clip_processor, blip_transform=blip_processor,arch=self.arch))
            compute_results_function = compute_results.circo
            pairings.append('default')
        
        elif 'genecis' in self.dataset.lower():   
            prop_file = '_'.join(self.dataset.lower().split('_')[1:])
            prop_file = os.path.join(self.dataset_path, 'genecis', prop_file + '.json')
            
            if 'object' in self.dataset.lower():
                datapath = os.path.join(self.dataset_path, 'coco2017', 'val2017')
                genecis_dataset = COCOValSubset(root_dir=datapath, val_split_path=prop_file, transform=clip_processor, blip_transform=blip_processor)                
            elif 'attribute' in self.dataset.lower():            
                datapath = os.path.join(self.dataset_path, 'Visual_Genome', 'VG_All')
                genecis_dataset = VAWValSubset(image_dir=datapath, val_split_path=prop_file, transform=clip_processor, blip_transform=blip_processor)
                
            target_datasets.append(genecis_dataset)
            query_datasets.append(genecis_dataset)
            compute_results_function = compute_results.genecis
            pairings.append('default')
            
        return target_datasets,query_datasets,compute_results_function,pairings

    def evaluate(self,query_datasets, target_datasets, pairings,compute_results_function,clip_model,clip_processor,blip_model,blip_processor):
        preload_dict = {key: None for key in ['img_features', 'captions', 'mods','suggestions']}
        file_utils.init_folder(self.dataset_path, self.task)
        if 'mods' in self.preload:
            # LLM-based caption modifications have to be queried only when BLIP model or BLIP prompt changes.
            preload_dict['mods'] = f'{self.dataset_path}/task/{self.task}/modified_captions/{self.preload_modified_captions_file}'
        if 'captions' in self.preload:
            preload_dict['captions'] = f'{self.dataset_path}/preload/image_captions/{self.preload_image_captions_file}'
        if 'suggestions' in self.preload:
            preload_dict['suggestions'] = f'{self.dataset_path}/task/{self.task}/suggestions/{self.preload_suggestions}'
        if 'img_features' in self.preload:
            preload_dict['img_features'] = f'{self.dataset_path}/preload/img_features/{self.clip}_{self.dataset}_{self.split}.pkl'
        for query_dataset, target_dataset, pairing in zip(query_datasets, target_datasets, pairings):
            termcolor.cprint(f'\n------ Evaluating Retrieval Setup: {pairing}', color='yellow', attrs=['bold'])
            
            ### General Input Arguments.
            input_kwargs = {
                'dataset_name':self.dataset,'blip_prompt_args':self.blip_prompt,'llm_prompt_args': self.llm_prompt,'retrieval':self.retrieval,
                'query_dataset': query_dataset, 'target_dataset': target_dataset, 'clip_model': clip_model, 
                'blip_model': blip_model, 'processor': clip_processor, 'device': self.device, 'split': self.split,
                'blip_transform': blip_processor, 'preload_dict': preload_dict,'arch':self.arch,'max_check_num':self.max_check_num,
                'blip':self.blip,'Check_LLM_model_name':self.Check_LLM_model_name,'dataset_path':self.dataset_path,'compute_results_function':compute_results_function,
                'openai_key':self.openai_key,"task":self.task
            }    
            
            ### Compute Target Image Features
            print(f'Extracting target image features using CLIP: {self.clip}.')
            index_features, index_names, index_ranks, aux_data = utils.extract_image_features(
                self.device, self.dataset, target_dataset, clip_model, preload=preload_dict['img_features'],arch = self.arch)
            index_features = torch.nn.functional.normalize(index_features.float(), dim=-1)
            input_kwargs.update({'index_features': index_features, 'index_names': index_names, 'index_ranks': index_ranks,'LLM_model_name':self.LLM_model_name})

                
            ### Compute Method-specific Query Features.
            # This part can be interchanged with any other method implementation.
            print(f'Generating conditional query predictions (CLIP: {self.clip}, BLIP: {self.blip}).')
            out_dict = utils.generate_predictions(**input_kwargs)
            input_kwargs.update(out_dict)
            
            ### Compute Dataset-specific Retrieval Scores.
            # This part is dataset-specific and declared above.
            print('Computing final retrieval metrics.')
            if self.dataset == 'genecis_focus_attribute':
                aux_data['ref_features'] = torch.nn.functional.normalize(aux_data['ref_features'].float().to(self.device))
                out_dict['predicted_features'] = torch.nn.functional.normalize(
                    (out_dict['predicted_features'].float() + aux_data['ref_features'])/2, dim=-1)

            input_kwargs.update(out_dict)                    
            result_metrics,labels = compute_results_function(**input_kwargs)    
            
            # Print metrics.
            print('\n')
            if result_metrics is not None:
                termcolor.cprint(f'Metrics for {self.dataset.upper()} ({self.split})- {pairing}', attrs=['bold'])
                for k, v in result_metrics.items():
                    print(f"{pairing}_{k} = {v:.2f}")
                    wandb.log({f"{pairing}_{k}": f"{v:.2f}"})
                wandb.finish()
            else:
                termcolor.cprint(f'No explicit metrics available for {self.dataset.upper()} ({self.split}) - {pairing}.', attrs=['bold'])            
