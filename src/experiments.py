from argparse import Namespace
import torch
import os
from pathlib import Path
from datasets import FashionIQDataset,CIRCODataset,CIRRDataset
import compute_results
import termcolor
import utils
import file_utils
from classes import Arch
class Experiment:
    def __init__(self,args:Namespace) -> None:
        super().__init__()
        # Get Arguments
        for arg in vars(args):
            value_arg = getattr(args,arg)
            self.__setattr__(arg,value_arg)
        self.device = torch.device(f'cuda:{self.device}' if torch.cuda.is_available() else 'cpu')
        self.current_directory = Path(__file__).resolve().parent
    def run(self):
        arch_model,arch_processor = self.load_arch_model()
        target_datasets,query_datasets,compute_results_function,pairings = self.load_dataset(arch_processor)
        self.evaluate(query_datasets, target_datasets, pairings,compute_results_function,arch_model,arch_processor)
        print("Finish.")
        return
    
    def load_arch_model(self):
        self.arch = Arch.from_string(s=self.clip)
        print('====================Loading Arch Model====================')
        clip_model,clip_processor = self.arch.load_model_and_preprocess(device=self.device,path=self.current_directory)
        print('Done.')
        return clip_model,clip_processor

    def load_dataset(self,clip_processor):
        ### Load Evaluation Datasets.
        target_datasets, query_datasets, pairings = [], [], []
        print('================Loading Evaluation Datasets================')
        if 'fashioniq' in self.dataset.lower():
            dress_type = self.dataset.split('_')[-1]
            target_datasets.append(FashionIQDataset(self.current_directory / self.dataset_path, self.split, [dress_type], 'classic', clip_processor))
            query_datasets.append(FashionIQDataset(self.current_directory / self.dataset_path, self.split, [dress_type], 'relative', clip_processor))
            pairings.append(dress_type)
            compute_results_function = compute_results.fiq
        
        elif self.dataset.lower() == 'cirr':
            split = 'test1' if self.split == 'test' else self.split
            target_datasets.append(CIRRDataset(self.current_directory / self.dataset_path, split, 'classic', clip_processor))
            query_datasets.append(CIRRDataset(self.current_directory / self.dataset_path, split, 'relative', clip_processor))
            compute_results_function = compute_results.cirr
            pairings.append('default')
            
        elif self.dataset.lower() == 'circo':
            target_datasets.append(CIRCODataset(self.current_directory / self.dataset_path, self.split, 'classic', clip_processor))
            query_datasets.append(CIRCODataset(self.current_directory / self.dataset_path, self.split, 'relative', clip_processor))
            compute_results_function = compute_results.circo
            pairings.append('default')
        else:
            raise print("Dataset must in ['cirr', 'circo','fashioniq_dress', 'fashioniq_toptee', 'fashioniq_shirt'].")
        return target_datasets,query_datasets,compute_results_function,pairings


    def evaluate(self,query_datasets, target_datasets, pairings,compute_results_function,clip_model,clip_processor,blip_processor):
        preload_dict = {key: None for key in ['img_features', 'captions', 'mods','suggestions']}
        file_utils.init_folder(self.current_directory / self.dataset_path)
        if 'mods' in self.preload:
            # LLM-based caption modifications have to be queried only when BLIP model or BLIP prompt changes.
            preload_dict['mods'] = f'{self.current_directory / self.dataset_path}/results/modified_captions/{self.preload_modified_captions_file}'
        if 'captions' in self.preload:
            preload_dict['captions'] = f'{self.current_directory / self.dataset_path}/{self.preload_image_captions_file}'
        if 'suggestions' in self.preload:
            preload_dict['suggestions'] = f'{self.current_directory / self.dataset_path}/results/suggestions/{self.preload_suggestions}'
        for query_dataset, target_dataset, pairing in zip(query_datasets, target_datasets, pairings):
            termcolor.cprint(f'\n===========Evaluating Retrieval Setup: {pairing}===========', color='yellow', attrs=['bold'])
            
            ### General Input Arguments.
            input_kwargs = {
                'dataset_name':self.dataset,'blip_prompt_args':self.blip_prompt,'llm_prompt_args': self.llm_prompt,'retrieval':self.retrieval,
                'query_dataset': query_dataset, 'target_dataset': target_dataset, 'clip_model': clip_model, 'processor': clip_processor,
                'device': self.device, 'split': self.split,'preload_dict': preload_dict,'arch':self.arch,'max_check_num':self.max_check_num,
                'planner':self.planner,'dataset_path':self.current_directory / self.dataset_path,'compute_results_function':compute_results_function,
                'planner_key':self.planner_key,'corrector_key':self.corrector_key,'corrector':self.corrector
            }    
            
            ### Compute Target Image Features
            print(f'Extracting target image features using CLIP: {self.clip}.')
            feature_extractor = utils.FeatureExtractor(clip_model=clip_model,device=self.device,arch=self.arch)
            index_features, index_names, index_ranks, aux_data = feature_extractor.extract_image_features(
                self.device, self.dataset, target_dataset, clip_model, preload=preload_dict['img_features'],arch = self.arch)
            index_features = torch.nn.functional.normalize(index_features.float(), dim=-1)
            input_kwargs.update({'index_features': index_features, 'index_names': index_names, 'index_ranks': index_ranks,'LLM_model_name':self.LLM_model_name})

                
            ### Compute Method-specific Query Features.
            print(f'================Generating conditional query predictions================')
            out_dict = utils.generate_predictions(**input_kwargs)
            input_kwargs.update(out_dict)
            result_metrics,labels = compute_results_function(**input_kwargs)    
            
            # Print metrics.
            print('\n')
            if result_metrics is not None:
                termcolor.cprint(f'Metrics for {self.dataset.upper()} ({self.split})- {pairing}', attrs=['bold'])
                for k, v in result_metrics.items():
                    print(f"{pairing}_{k} = {v:.2f}")
            else:
                termcolor.cprint(f'No explicit metrics available for {self.dataset.upper()} ({self.split}) - {pairing}.', attrs=['bold'])            
