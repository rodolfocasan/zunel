# zunel/timbre_enhancement.py
import numpy as np
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F





def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float, eps: float = 1e-8) -> torch.Tensor:
    v0_norm = v0 / (torch.norm(v0, dim=-1, keepdim=True) + eps)
    v1_norm = v1 / (torch.norm(v1, dim=-1, keepdim=True) + eps)
    
    dot = torch.sum(v0_norm * v1_norm, dim=-1, keepdim=True)
    dot = torch.clamp(dot, -1.0 + eps, 1.0 - eps)
    
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    
    s0 = torch.sin((1.0 - t) * omega) / (sin_omega + eps)
    s1 = torch.sin(t * omega) / (sin_omega + eps)
    return s0 * v0 + s1 * v1


def multi_slerp(embeddings: List[torch.Tensor], weights: Optional[List[float]] = None) -> torch.Tensor:
    if len(embeddings) == 1:
        return embeddings[0]
    
    if weights is None:
        weights = [1.0 / len(embeddings)] * len(embeddings)
    
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    result = embeddings[0] * weights[0]
    for i in range(1, len(embeddings)):
        t = weights[i] / (1.0 - sum(weights[:i]))
        result = slerp(result, embeddings[i], t)
    return result


def compute_embedding_quality_score(embedding: torch.Tensor, reference_stats: dict) -> float:
    norm = torch.norm(embedding).item()
    
    norm_score = 1.0 - abs(norm - reference_stats.get('mean_norm', 1.0)) / reference_stats.get('std_norm', 1.0)
    norm_score = max(0.0, min(1.0, norm_score))
    return norm_score


def weighted_embedding_fusion(embeddings: List[torch.Tensor], quality_scores: List[float]) -> torch.Tensor:
    scores_tensor = torch.tensor(quality_scores, dtype=embeddings[0].dtype, device=embeddings[0].device)
    scores_tensor = F.softmax(scores_tensor * 2.0, dim=0)
    
    embeddings_stacked = torch.stack(embeddings, dim=0)
    scores_tensor = scores_tensor.view(-1, 1, 1)
    
    weighted = embeddings_stacked * scores_tensor
    return weighted.sum(dim=0)





class TimbreEnhancer:
    def __init__(self, diversity_weight: float = 0.3, quality_weight: float = 0.7):
        self.diversity_weight = diversity_weight
        self.quality_weight = quality_weight
        self.reference_stats = {
            'mean_norm': 1.0,
            'std_norm': 0.1
        }
    
    def update_reference_stats(self, embeddings: List[torch.Tensor]):
        norms = [torch.norm(e).item() for e in embeddings]
        self.reference_stats['mean_norm'] = np.mean(norms)
        self.reference_stats['std_norm'] = np.std(norms) + 1e-8
    
    def select_diverse_samples(self, embeddings: List[torch.Tensor], max_samples: int = 5) -> List[int]:
        if len(embeddings) <= max_samples:
            return list(range(len(embeddings)))
        
        embeddings_tensor = torch.stack(embeddings, dim=0).squeeze(1).squeeze(1)
        
        selected = [0]
        remaining = list(range(1, len(embeddings)))
        
        for _ in range(max_samples - 1):
            max_min_dist = -1
            best_idx = -1
            
            for idx in remaining:
                candidate = embeddings_tensor[idx].unsqueeze(0)
                selected_embeddings = embeddings_tensor[selected]
                
                dists = torch.cdist(candidate, selected_embeddings, p=2)
                min_dist = dists.min().item()
                
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = idx
            
            if best_idx != -1:
                selected.append(best_idx)
                remaining.remove(best_idx)
        return selected
    
    def enhance_embeddings(self, embeddings: List[torch.Tensor], use_quality_weighting: bool = True) -> torch.Tensor:
        if len(embeddings) == 0:
            raise ValueError("No embeddings provided")
        
        if len(embeddings) == 1:
            return embeddings[0]
        
        self.update_reference_stats(embeddings)
        
        diverse_indices = self.select_diverse_samples(embeddings, max_samples=5)
        selected_embeddings = [embeddings[i] for i in diverse_indices]
        
        if use_quality_weighting:
            quality_scores = [
                compute_embedding_quality_score(emb, self.reference_stats) 
                for emb in selected_embeddings
            ]
            
            return weighted_embedding_fusion(selected_embeddings, quality_scores)
        else:
            return multi_slerp(selected_embeddings)


def extract_timbre_specific_features(spectrogram: torch.Tensor, frame_shift: int = 256, sample_rate: int = 22050) -> dict:
    spec_db = 20 * torch.log10(spectrogram + 1e-8)
    
    spectral_centroid = torch.sum(spec_db * torch.arange(spec_db.size(1), device=spec_db.device).unsqueeze(0).unsqueeze(0), dim=1) / (torch.sum(spec_db, dim=1) + 1e-8)
    
    spectral_spread = torch.sqrt(
        torch.sum(spec_db * (torch.arange(spec_db.size(1), device=spec_db.device).unsqueeze(0).unsqueeze(0) - spectral_centroid.unsqueeze(1)) ** 2, dim=1) / 
        (torch.sum(spec_db, dim=1) + 1e-8)
    )
    
    formant_region_1 = spec_db[:, 20:40, :]
    formant_region_2 = spec_db[:, 40:80, :]
    formant_region_3 = spec_db[:, 80:120, :]
    
    formant_emphasis_1 = torch.mean(formant_region_1, dim=(1, 2))
    formant_emphasis_2 = torch.mean(formant_region_2, dim=(1, 2))
    formant_emphasis_3 = torch.mean(formant_region_3, dim=(1, 2))
    
    harmonic_content = torch.std(spec_db[:, :80, :], dim=1)
    
    return {
        'spectral_centroid': spectral_centroid.mean().item(),
        'spectral_spread': spectral_spread.mean().item(),
        'formant_emphasis': (formant_emphasis_1.mean().item(), 
                            formant_emphasis_2.mean().item(), 
                            formant_emphasis_3.mean().item()),
        'harmonic_content': harmonic_content.mean().item()
    }