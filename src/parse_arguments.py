import argparse
import prompts

def parse_arguments():
    parser = argparse.ArgumentParser()
    # Base Arguments
    parser.add_argument("--device", type=int, default=0, 
                        help="GPU ID to use.")
    parser.add_argument("--preload", nargs='+', type=str, default=['captions','mods'],# ['captions','mods']
                        help='List of properties to preload is computed once before.[captions,mods]')    
    # Dataset Arguments ['dress', 'toptee', 'shirt']
    parser.add_argument("--dataset", default="fashioniq_dress", type=str, required=False, 
                        choices=['cirr', 'circo',
                                 'fashioniq_dress', 'fashioniq_toptee', 'fashioniq_shirt'],
                        help="Dataset to use")
    parser.add_argument("--split", type=str, default='val', choices=['val', 'test'],
                        help='Dataset split to evaluate on. Some datasets require special testing protocols s.a. cirr/circo.')
    parser.add_argument("--dataset_path", default="datasets/FASHIONIQ", type=str, required=False,
                        help="Path to the dataset")
    # Base Model Arguments
    available_prompts = [f'prompts.{x}' for x in prompts.__dict__.keys() if '__' not in x]
    parser.add_argument("--planner_prompt", default='prompts.structural_modifier_prompt_fashion', type=str, choices=available_prompts,
                        help='Denotes the base prompt to use to probe the planner. Has to be available in prompts.py')
    parser.add_argument("--clip", type=str, default='ViT-B-32', 
                        choices=['ViT-Base/32','ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'RN50x4', 'ViT-bigG-14',
                                 'ViT-B-32','ViT-B-16','ViT-L-14','ViT-H-14','ViT-g-14'],
                        help="Which CLIP text-to-image retrieval model to use."),
    # Planner Arguments
    parser.add_argument("--planner", default="GPT-4o-Mini", type=str,choices=['Qwen-Turbo','GPT-4o-Mini','GPT-3.5-Turbo','GPT-4o'],
                        help='Which modify captions model to use.')
    parser.add_argument("--captioner_prompt", default='prompts.blip_prompt', type=str, choices=available_prompts,
                        help='Denotes the base prompt to use alongside BLIP. Has to be available in prompts.py')    
    parser.add_argument("--planner_key", default="<your_planner_key_here>", type=str,
                        help='Account key for planner usage.')
    # Corrector Arguments
    parser.add_argument("--max_correct_num", default=1, type=int,
                        help='The maximum number of times the modified captions need to be corrected.')
    parser.add_argument("--corrector", default="GPT-4o-Mini", type=str,choices=['Qwen-Turbo','GPT-4o-Mini','GPT-3.5-Turbo','GPT-4o'],
                        help='Which correct captions model to use.')
    parser.add_argument("--corrector_key", default="<your_corrector_key_here>", type=str,
                        help='Account key for corrector usage.')
    # Text-to-Image Retrieval Arguments.
    parser.add_argument("--retrieval", type=str, default='default', choices=['default'],
                        help='Type of T2I Retrieval method.')
    return parser.parse_args()
