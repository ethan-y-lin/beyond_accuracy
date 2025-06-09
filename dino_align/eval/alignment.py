from utils.metrics import AlignmentMetrics
import torch.nn.functional as F

def compute_alignment(image_embeds, desc_embeds, target_embeds, k=30, use_batch=False, verbose=True):
    """
    Evaluate CLIP alignment between image embeddings and descriptor embeddings,
    comparing against target embeddings using mutual k-NN.

    Args:
        image_embeds (torch.Tensor): Image embeddings
        desc_embeds (torch.Tensor): Descriptor embeddings (used to compute similarity)
        target_embeds (torch.Tensor): Target embeddings to evaluate mutual k-NN against
        k (int): Number of nearest neighbors to consider
        use_batch (bool): Whether to use the batch version of mutual_knn
        verbose (bool): Whether to print logging information

    Returns:
        float: Alignment accuracy score
    """
    if verbose:
        print("COMPUTING ALIGNMENT")
        print("Descriptor embedding shape:", desc_embeds.shape)

    similarity = 100.0 * image_embeds @ desc_embeds.T
    similarity = F.normalize(similarity, p=2, dim=-1)

    if use_batch:
        score = AlignmentMetrics.mutual_knn_batch(target_embeds, similarity, k)
    else:
        score = AlignmentMetrics.mutual_knn(target_embeds, similarity, k)
    if verbose:
        print(f"Alignment: {score}")
    return score