import json
import os
from typing import List, Dict, Union
import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import tqdm

@torch.no_grad()
def fiq(
    device: torch.device,
    predicted_features: torch.Tensor,
    target_names: List,
    index_features: torch.Tensor,
    index_names: List,
    split: str='val',
    **kwargs
) -> Dict[str, float]:
    """
    Compute the retrieval metrics on the Fashion-IQ validation set fiven the dataset, pseudo tokens and reference names.
    Computes Recall@10 and Recall@50.
    """
    # Move the features to the device
    index_features = torch.nn.functional.normalize(index_features).to(device)
    predicted_features = torch.nn.functional.normalize(predicted_features).to(device)

    # Compute the distances
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Check if the target names are in the top 10 and top 50
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    output_metrics = {
        'Recall@1': (torch.sum(labels[:, :1]) / len(labels)).item() * 100,
        'Recall@5': (torch.sum(labels[:, :5]) / len(labels)).item() * 100,
        'Recall@10': (torch.sum(labels[:, :10]) / len(labels)).item() * 100,
        'Recall@50': (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    }
    return output_metrics,sorted_index_names
    
@torch.no_grad()
def cirr(
    device: torch.device, 
    predicted_features: torch.Tensor, 
    reference_names: List, 
    targets: Union[np.ndarray,List], 
    target_names: List, 
    index_features: torch.Tensor, 
    index_names: List, 
    query_ids: Union[np.ndarray,List],
    preload_dict: Dict[str, Union[str, None]]=[],
    split: str='val',  
    **kwargs
) -> Dict[str, float]:
    """
    Compute the retrieval metrics on the CIRR validation set given the dataset, pseudo tokens and the reference names.
    Computes Recall@1, 5, 10 and 50. If given a test set, will generate submittable file.
    """   
    # Put on device.
    index_features = index_features.to(device)
    predicted_features = predicted_features.to(device)

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    if distances.ndim == 3:
        # If there are multiple features per instance, we average.
        distances = distances.mean(dim=1)
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    resize = len(sorted_index_names) if split == 'test' else len(target_names)
    reference_mask = torch.tensor(sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(resize, -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0], sorted_index_names.shape[1] - 1)
    
    # Compute the subset predictions and ground-truth labels
    targets = np.array(targets)
    group_mask = (sorted_index_names[..., None] == targets[:, None, :]).sum(-1).astype(bool)

    if split == 'test':
        sorted_group_names = sorted_index_names[group_mask].reshape(sorted_index_names.shape[0], -1)
        pairid_to_retrieved_images, pairid_to_group_retrieved_images = {}, {}
        for pair_id, prediction in zip(query_ids, sorted_index_names):
            pairid_to_retrieved_images[str(int(pair_id))] = prediction[:50].tolist()
        for pair_id, prediction in zip(query_ids, sorted_group_names):
            pairid_to_group_retrieved_images[str(int(pair_id))] = prediction[:3].tolist()            

        submission = {'version': 'rc2', 'metric': 'recall'}
        group_submission = {'version': 'rc2', 'metric': 'recall_subset'}

        submission.update(pairid_to_retrieved_images)
        group_submission.update(pairid_to_group_retrieved_images)
        submissions_folder_path = os.path.join(os.getcwd(), 'data', 'test_submissions', 'cirr')
        os.makedirs(submissions_folder_path, exist_ok=True)
        with open(os.path.join(submissions_folder_path, preload_dict['test']), 'w') as file:
            json.dump(submission, file, sort_keys=True)
        with open(os.path.join(submissions_folder_path, f"subset_{preload_dict['test']}"), 'w') as file:
            json.dump(group_submission, file, sort_keys=True)                        
        return None,sorted_index_names
            
    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(sorted_index_names == np.repeat(np.array(target_names), len(index_names) - 1).reshape(len(target_names), -1))    
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)

    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    output_metrics = {f'recall@{key}': (torch.sum(labels[:, :key]) / len(labels)).item() * 100 for key in [1, 5, 10, 50]}
    output_metrics.update({f'group_recall@{key}': (torch.sum(group_labels[:, :key]) / len(group_labels)).item() * 100 for key in [1, 2, 3]})

    return output_metrics,sorted_index_names


@torch.no_grad()
def circo(
    device: torch.device, 
    predicted_features: torch.Tensor, 
    targets: Union[np.ndarray,List], 
    target_names: List, 
    index_features: torch.Tensor, 
    index_names: List,
    query_ids: Union[np.ndarray,List],dataset_name,dataset_path,task,
    preload_dict: Dict[str, Union[str, None]]=[],
    split: str='val',
    loop:str = 0,
    **kwargs
) -> Dict[str, float]:
    """
    Compute the retrieval metrics on the CIRCO validation set given the pseudo tokens and the reference names.
    Computes mAP@5, 10, 25 and 50. If test-split, generates submittable file.
    """
    # Load the model
    # Put on device.
    index_features = index_features.to(device)#, dtype=torch.float16)
    predicted_features = predicted_features.to(device)    
    ### Compute Test Submission in case of test split.
    if split == 'test':
        print('Generating test submission file!')
        similarity = predicted_features @ index_features.T
        if similarity.ndim == 3:
            # If there are multiple features per instance, we average.
            similarity = similarity.mean(dim=1)                    
        sorted_indices = torch.topk(similarity, dim=-1, k=50).indices.cpu()
        sorted_index_names = np.array(index_names)[sorted_indices]
        # Return prediction dict to submit.
        queryid_to_retrieved_images = {
            query_id: query_sorted_names[:50].tolist() for (query_id, query_sorted_names) in zip(query_ids, sorted_index_names)            
        }
        submissions_folder_path = os.path.join(os.getcwd(), 'data', 'test_submissions', 'circo',loop)
        os.makedirs(submissions_folder_path, exist_ok=True)
        with open(os.path.join(submissions_folder_path, preload_dict['test']), 'w') as file:
            json.dump(queryid_to_retrieved_images, file, sort_keys=True)        
        return None,sorted_index_names
    
    ### Directly compute metrics when using validation split.
    retrievals = [5, 10, 25, 50]
    recalls = {key: [] for key in retrievals}
    maps = {key: [] for key in retrievals}
    sorted_index_names_list = []
    for predicted_feature, target_name, sub_targets in tqdm.tqdm(zip(predicted_features, target_names, targets), total=len(predicted_features), desc='Computing Metric.'):
        sub_targets = np.array(sub_targets)[np.array(sub_targets) != '']  # remove trailing empty strings added for collate_fn
        similarity = predicted_feature @ index_features.T
        if similarity.ndim == 2:
            # If there are multiple features per instance, we average.
            similarity = similarity.mean(dim=0)
        sorted_indices = torch.topk(similarity, dim=-1, k=50).indices.cpu()
        sorted_index_names = np.array(index_names)[sorted_indices]
        sorted_index_names_list.append(sorted_index_names)
        map_labels = torch.tensor(np.isin(sorted_index_names, sub_targets), dtype=torch.uint8)
        precisions = torch.cumsum(map_labels, dim=0) * map_labels  # Consider only positions corresponding to GTs
        precisions = precisions / torch.arange(1, map_labels.shape[0] + 1)  # Compute precision for each position

        for key in retrievals:
            maps[key].append(float(torch.sum(precisions[:key]) / min(len(sub_targets), key)))

        assert target_name == sub_targets[0], f"Target name not in GTs {target_name} {sub_targets}"
        single_gt_labels = torch.tensor(sorted_index_names == target_name)
        
        for key in retrievals:
            recalls[key].append(float(torch.sum(single_gt_labels[:key])))
    sorted_index_names_list = np.array(sorted_index_names_list)
    output_metrics = {f'mAP@{key}': np.mean(item) * 100 for key, item in maps.items()}
    output_metrics.update({f'recall@{key}': np.mean(item) * 100 for key, item in recalls.items()})
    return output_metrics,sorted_index_names_list