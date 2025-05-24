import torch
import torchaudio.functional as TAF
from tqdm import tqdm
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cross_decomposition import CCA

try:
    import pymp
    pymp_available = True
except ImportError:
    pymp_available = False
    print("Please install the pymp library using `pip install pymp` to speed up non-batched metrics")


class AlignmentMetrics:

    SUPPORTED_METRICS = [
        "cycle_knn",
        "mutual_knn",
        "lcs_knn",
        "cka",
        "unbiased_cka",
        "cknna",
        "svcca",
        "edit_distance_knn",
    ]

    @staticmethod
    def measure(metric, *args, **kwargs):
        """ metric is a string for the function """

        if metric not in AlignmentMetrics.SUPPORTED_METRICS:
            raise ValueError(f"Unrecognized metric: {metric}")

        return getattr(AlignmentMetrics, metric)(*args, **kwargs)


    @staticmethod
    def cycle_knn(feats_A, feats_B, topk):
        """
        LLM nearest neighbors -> Query Language Pair -> LVM nearest neighbors
        Args:
            feats_A: A torch tensor of shape N x feat_dim
            feats_B: A torch tensor of shape N x feat_dim

        Returns:
            acc: a float representing the accuracy
        """
        knn_A = compute_nearest_neighbors(feats_A, topk)
        knn_B = compute_nearest_neighbors(feats_B, topk)   
        return compute_knn_accuracy(knn_A[knn_B]).item()

    @staticmethod
    def select_relevant_descriptors(XY_T, n_components=10):
        """
        Runs PCA on XY_T (N x D), and for each principal component, picks 
        the single original column (descriptor) whose loading has the 
        largest absolute value. Then returns a new N x n_components matrix 
        containing just those selected columns from XY_T.
        
        Args:
            XY_T (torch.Tensor): Shape (N, D). 
                                N = number of samples, D = number of descriptors.
            n_components (int): Number of principal components to consider.
            
        Returns:
            selected_matrix (torch.Tensor):
                Shape (N, n_components). Each column is the original 
                descriptor from XY_T that aligns most with a PCA component.
            selected_indices (list of int):
                The column indices in XY_T that were selected.
        """

        # 1) Move to CPU numpy for PCA
        device = XY_T.device
        X_cpu = XY_T.detach().cpu().numpy()

        # 2) Fit PCA (center=True by default). 
        #    For very large (N, D), consider svd_solver='randomized'.
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(X_cpu)  # shape: (N, D)

        # 3) Each row of pca.components_ is a PC: shape (n_components, D)
        components = pca.components_  # numpy array

        # 4) For each PC, find the descriptor (column) with the largest |coefficient|
        used = set()
        selected_indices = []
        for i in range(n_components):
            pc = components[i]
            sorted_feats = np.argsort(-np.abs(pc))  # sort descending by absolute loading
            # pick the first not in used
            for f in sorted_feats:
                if f not in used:
                    used.add(f)
                    selected_indices.append(f)
                    break

        # 5) Slice columns from the original XY_T
        #    Convert the selected indices to a torch tensor
        indices_torch = torch.tensor(selected_indices, dtype=torch.long, device=device)
        # Use advanced indexing to grab those columns in XY_T
        selected_matrix = XY_T[:, indices_torch]  # shape: (N, n_components)

        return selected_matrix, selected_indices
    
    @staticmethod
    def random_column_sampling(XY_T, n_samples):
        """
        Randomly sample columns from XY_T and return the new subsampled matrix and the corresponding indices.

        Args:
            XY_T: A torch tensor of shape (N x D), where N is the number of samples and D is the number of descriptors
            n_samples: Number of columns (descriptors) to sample

        Returns:
            subsampled_matrix: A tensor of shape (N x n_samples) containing the randomly selected columns
            selected_indices: A list of indices corresponding to the selected columns in the original XY_T matrix
        """
        # Get the number of columns (descriptors) in XY_T
        _, num_columns = XY_T.shape

        # Randomly sample `n_samples` indices from the range [0, num_columns)
        selected_indices = torch.randint(0, num_columns, (n_samples,), dtype=torch.long)
            # Generate a random permutation of the column indices
        # shuffled_indices = torch.randperm(XY_T.size(1))

        # # Shuffle the columns of XY_T
        # shuffled_XY_T = XY_T[:, shuffled_indices]
        # Select the corresponding columns from XY_T
        subsampled_matrix = XY_T[:, selected_indices]

        return subsampled_matrix, selected_indices.tolist()
    @staticmethod
    def mutual_knn_with_selected_descriptors(feats_A, XY_T, topk, n_components=1000):
        """
        Computes the mutual KNN accuracy using the top relevant descriptors from PCA.

        Args:
            feats_A: A torch tensor of shape N x feat_dim
            XY_T: A torch tensor of shape N x D (similarity scores)
            topk: Number of nearest neighbors to consider
            n_components: Number of PCA components to use for descriptor selection

        Returns:
            A float representing the mutual KNN accuracy
        """
        # Select the most relevant descriptors and their indices
        # relevant_descriptors, selected_indices = AlignmentMetrics.select_relevant_descriptors(XY_T, min(n_components, XY_T.shape[1]))
        print("RELEVANT DESCRIPTOR SHAPE", XY_T.shape)
        # Compute KNN for feats_A and the relevant descriptors
        knn_A = compute_nearest_neighbors(feats_A, topk)
        knn_B = compute_nearest_neighbors(XY_T, topk)

        n = knn_A.shape[0]
        topk = knn_A.shape[1]

        # Create a range tensor for indexing
        range_tensor = torch.arange(n, device=knn_A.device).unsqueeze(1)

        # Create binary masks for knn_A and knn_B
        lvm_mask = torch.zeros(n, n, device=knn_A.device)
        llm_mask = torch.zeros(n, n, device=knn_A.device)

        lvm_mask[range_tensor, knn_A] = 1.0
        llm_mask[range_tensor, knn_B] = 1.0
        
        acc = (lvm_mask * llm_mask).sum(dim=1) / topk
        
        return acc.mean().item()

    @staticmethod
    def mutual_knn(feats_A, feats_B, topk):
        """
        Computes the mutual KNN accuracy.

        Args:
            feats_A: A torch tensor of shape N x feat_dim
            feats_B: A torch tensor of shape N x feat_dim

        Returns:
            A float representing the mutual KNN accuracy
        """
        knn_A = compute_nearest_neighbors(feats_A, topk)
        knn_B = compute_nearest_neighbors(feats_B, topk)   

        n = knn_A.shape[0]
        topk = knn_A.shape[1]

        # Create a range tensor for indexing
        range_tensor = torch.arange(n, device=knn_A.device).unsqueeze(1)

        # Create binary masks for knn_A and knn_B
        lvm_mask = torch.zeros(n, n, device=knn_A.device)
        llm_mask = torch.zeros(n, n, device=knn_A.device)

        lvm_mask[range_tensor, knn_A] = 1.0
        llm_mask[range_tensor, knn_B] = 1.0
        
        acc = (lvm_mask * llm_mask).sum(dim=1) / topk
        
        return acc.mean().item()
    
    @staticmethod
    def mutual_knn_batch(feats_A, feats_B, topk):
        """
        Computes mutual KNN accuracy with per-row set intersection,
        matching the dense mask-based version exactly.
        """
        knn_A = compute_nearest_neighbors(feats_A, topk=topk)  # [N, k]
        knn_B = compute_nearest_neighbors(feats_B, topk=topk)  # [N, k]

        # Make sure both are [N, k] on same device
        knn_A = knn_A.to(feats_A.device)
        knn_B = knn_B.to(feats_A.device)

        N = knn_A.shape[0]
        intersection_counts = torch.zeros(N, device=feats_A.device)

        for i in tqdm(range(N)):
            neighbors_A = knn_A[i]
            neighbors_B = knn_B[i]
            intersection_counts[i] = torch.isin(neighbors_A, neighbors_B).sum()

        return (intersection_counts / topk).mean().item()

    @staticmethod
    def lcs_knn(feats_A, feats_B, topk):
        knn_A = compute_nearest_neighbors(feats_A, topk)
        knn_B = compute_nearest_neighbors(feats_B, topk)        
        score = longest_ordinal_sequence(knn_A, knn_B).float().mean()
        return score
    
    
    @staticmethod
    def cka(feats_A, feats_B, kernel_metric='ip', rbf_sigma=1.0, unbiased=False):
        """Computes the unbiased Centered Kernel Alignment (CKA) between features."""
        
        if kernel_metric == 'ip':
            # Compute kernel matrices for the linear case
            K = torch.mm(feats_A, feats_A.T)
            L = torch.mm(feats_B, feats_B.T)
        elif kernel_metric == 'rbf':
            # COMPUTES RBF KERNEL
            K = torch.exp(-torch.cdist(feats_A, feats_A) ** 2 / (2 * rbf_sigma ** 2))
            L = torch.exp(-torch.cdist(feats_B, feats_B) ** 2 / (2 * rbf_sigma ** 2))
        else:
            raise ValueError(f"Invalid kernel metric {kernel_metric}")

        # Compute HSIC values
        hsic_fn = hsic_unbiased if unbiased else hsic_biased
        hsic_kk = hsic_fn(K, K)
        hsic_ll = hsic_fn(L, L)
        hsic_kl = hsic_fn(K, L)

        # Compute CKA
        #print('hsic', hsic_kl)
        cka_value = hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-6)        
        return cka_value.item()
    
    
    @staticmethod
    def unbiased_cka(*args, **kwargs):
        kwargs['unbiased'] = True
        return AlignmentMetrics.cka(*args, **kwargs)
    
    
    @staticmethod
    def svcca(feats_A, feats_B, cca_dim=10):

        # Center and scale the activations
        def preprocess_activations(act):
            act = act - torch.mean(act, axis=0)
            act = act / (torch.std(act, axis=0) + 1e-8)
            return act

        feats_A = preprocess_activations(feats_A)
        feats_B = preprocess_activations(feats_B)

        # Compute SVD
        U1, _, _ = torch.svd_lowrank(feats_A, q=cca_dim)
        U2, _, _ = torch.svd_lowrank(feats_B, q=cca_dim)
        
        U1 = U1.cpu().detach().numpy()
        U2 = U2.cpu().detach().numpy()

        # Compute CCA
        cca = CCA(n_components=cca_dim)
        cca.fit(U1, U2)
        U1_c, U2_c = cca.transform(U1, U2)

        # sometimes it goes to nan, this is just to avoid that
        U1_c += 1e-10 * np.random.randn(*U1_c.shape)
        U2_c += 1e-10 * np.random.randn(*U2_c.shape)

        # Compute SVCCA similarity
        svcca_similarity = np.mean(
            [np.corrcoef(U1_c[:, i], U2_c[:, i])[0, 1] for i in range(cca_dim)]
        )
        return svcca_similarity
    
    
    @staticmethod
    def edit_distance_knn(feats_A, feats_B, topk):
        """
        Computes the edit distance between the nearest neighbors of feats_A and feats_B.
        """
        knn_A = compute_nearest_neighbors(feats_A, topk)
        knn_B = compute_nearest_neighbors(feats_B, topk)
        
        # given N x topk with integer entries, compute edit distance
        n = knn_A.shape[0]
        topk = knn_A.shape[1]

        edit_distance = compute_distance(knn_A, knn_B, TAF.edit_distance)
        return 1 - torch.mean(edit_distance) / topk
    
    
    @staticmethod
    def cknna(feats_A, feats_B, topk=None, distance_agnostic=False, unbiased=True):
        """ similarity only cka variant """
        n = feats_A.shape[0]
                
        if topk < 2:
            raise ValueError("CKNNA requires topk >= 2")
        
        if topk is None:
            topk = feats_A.shape[0] - 1
                            
        K = feats_A @ feats_A.T
        L = feats_B @ feats_B.T
        device = feats_A.device

        def similarity(K, L, topk):                         
            if unbiased:            
                K_hat = K.clone().fill_diagonal_(float("-inf"))
                L_hat = L.clone().fill_diagonal_(float("-inf"))
            else:
                K_hat, L_hat = K, L

            # get topk indices for each row
            # if unbiased we cannot attend to the diagonal unless full topk
            # else we can attend to the diagonal
            _, topk_K_indices = torch.topk(K_hat, topk, dim=1)
            _, topk_L_indices = torch.topk(L_hat, topk, dim=1)
            
            # create masks for nearest neighbors
            mask_K = torch.zeros(n, n, device=device).scatter_(1, topk_K_indices, 1)
            mask_L = torch.zeros(n, n, device=device).scatter_(1, topk_L_indices, 1)
            
            # intersection of nearest neighbors
            mask = mask_K * mask_L
                        
            if distance_agnostic:
                sim = mask * 1.0
            else:
                if unbiased:
                    sim = hsic_unbiased(mask * K, mask * L)
                else:
                    sim = hsic_biased(mask * K, mask * L)
            return sim

        sim_kl = similarity(K, L, topk)
        sim_kk = similarity(K, K, topk)
        sim_ll = similarity(L, L, topk)
                
        return sim_kl.item() / (torch.sqrt(sim_kk * sim_ll) + 1e-6).item()


def hsic_unbiased(K, L):
    """
    Compute the unbiased Hilbert-Schmidt Independence Criterion (HSIC) as per Equation 5 in the paper.
    > Reference: https://jmlr.csail.mit.edu/papers/volume13/song12a/song12a.pdf
    """
    m = K.shape[0]

    # Zero out the diagonal elements of K and L
    K_tilde = K.clone().fill_diagonal_(0)
    L_tilde = L.clone().fill_diagonal_(0)

    # Compute HSIC using the formula in Equation 5
    HSIC_value = (
        (torch.sum(K_tilde * L_tilde.T))
        + (torch.sum(K_tilde) * torch.sum(L_tilde) / ((m - 1) * (m - 2)))
        - (2 * torch.sum(torch.mm(K_tilde, L_tilde)) / (m - 2))
    )

    HSIC_value /= m * (m - 3)
    return HSIC_value


def hsic_biased(K, L):
    """ Compute the biased HSIC (the original CKA) """
    H = torch.eye(K.shape[0], dtype=K.dtype, device=K.device) - 1 / K.shape[0]
    return torch.trace(K @ H @ L @ H)

    
def compute_knn_accuracy(knn):
    """
    Compute the accuracy of the nearest neighbors. Assumes index is the gt label.
    Args:
        knn: a torch tensor of shape N x topk
    Returns:
        acc: a float representing the accuracy
    """
    n = knn.shape[0]
    acc = knn == torch.arange(n, device=knn.device).view(-1, 1, 1)
    acc = acc.float().view(n, -1).max(dim=1).values.mean()
    return acc
    

def compute_nearest_neighbors(feats, topk=1):
    """
    Compute the nearest neighbors of feats
    Args:
        feats: a torch tensor of shape N x D
        topk: the number of nearest neighbors to return
    Returns:
        knn: a torch tensor of shape N x topk
    """
    assert feats.ndim == 2, f"Expected feats to be 2D, got {feats.ndim}"
    knn = (
        (feats @ feats.T).fill_diagonal_(-1e8).argsort(dim=1, descending=True)[:, :topk]
    )
    return knn

def compute_nearest_neighbors_batch(feats, topk=1, batch_size=1024):
    """
    Compute nearest neighbors using batched matrix multiplication.
    Args:
        feats: torch.Tensor of shape [N, D]
        topk: int, number of neighbors
        batch_size: int, batch size for computing similarities
    Returns:
        knn: torch.Tensor of shape [N, topk]
    """
    N, D = feats.shape
    knn_indices = []

    feats = feats.to(torch.float32)
    for i in tqdm(range(0, N, batch_size)):
        batch = feats[i:i+batch_size]             # [B, D]
        sims = batch @ feats.T                    # [B, N]
        sims[:, i:i+batch_size] = -1e8            # exclude self neighbors
        topk_idx = sims.topk(topk, dim=1).indices
        knn_indices.append(topk_idx)

    return torch.cat(knn_indices, dim=0)  # [N, topk]

def longest_ordinal_sequence(X, Y):
    """ For each pair in X and Y, compute the length of the longest sub-sequence (LCS) """
    
    def lcs_length(x, y):
        """
        Compute the length of the longest common subsequence between two sequences.
        This is a classic dynamic programming implementation.
        """
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    lcs = compute_distance(X, Y, lcs_length)
    return lcs


def compute_distance(X, Y, dist_fn):
    """ compute distance in parallel"""
    B, N = X.shape
    distances = np.zeros(B)
    X, Y = X.cpu().numpy(), Y.cpu().numpy()

    if pymp_available:
        with pymp.Parallel(4) as p:
            for i in p.range(B):
                distances[i] = dist_fn(X[i], Y[i])
    else:
        for i in range(B):
            distances[i] = dist_fn(X[i], Y[i])
    return torch.tensor(distances)


def remove_outliers(feats, q, exact=False, max_threshold=None):
    if q == 1:
        return feats

    if exact:
        # sorts the whole tensor and gets the q-th percentile
        q_val = feats.view(-1).abs().sort().values[int(q * feats.numel())]
    else:
        # quantile for element in the tensor and take the average
        q_val = torch.quantile(feats.abs().flatten(start_dim=1), q, dim=1).mean()

    if max_threshold is not None:
        max_threshold = max(max_threshold, q_val)

    return feats.clamp(-q_val, q_val)



if __name__ == "__main__":
    import torch.nn.functional as F
    torch.manual_seed(0)
    feats_A = torch.randn(64, 8192)
    feats_B = torch.randn(64, 8192)
    feats_A = F.normalize(feats_A, dim=-1)
    feats_B = F.normalize(feats_B, dim=-1)

    import time 
    trials = 10

    t0 = time.time()
    for metric in AlignmentMetrics.SUPPORTED_METRICS:

        scores, times = [], []
        for t in range(trials):
            t_st = time.time()

            kwargs = {}
            if 'nn' in metric:
                kwargs['topk'] = 10
            if 'cca' in metric:
                kwargs['cca_dim'] = 10
            if 'kernel' in metric:
                kwargs['dist'] = 'sample'

            score = AlignmentMetrics.measure(metric, feats_A, feats_B, **kwargs)
            scores.append(score)
            times.append(time.time() - t_st)        
        print(f"{metric.rjust(20)}: {np.mean(scores):1.3f} [elapsed: {np.mean(times):.2f}s]")

    print(f'Total time: {time.time() - t0:.2f}s')