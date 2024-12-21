import torch
import torch.nn as nn
import torch.nn.functional as F

class image_motif(nn.Module):
    def __init__(self):
        super(image_motif, self).__init__()
        self.unfold = nn.Unfold(kernel_size=3)
        self.group = 16
        self.k = 3

    def forward(self, feature):
        _, C, H, W = feature.shape
        fold = nn.Fold(output_size=(H,W), kernel_size=3)

        feature = feature.chunk(self.group, dim=1)
        for i in range(self.group):
            motif = self.unfold(feature[i])

            distances = torch.cdist(motif, motif).cuda()


            _, width, height = distances.shape

            diagonal_indices = torch.arange(min(width, height))
            distances[:, diagonal_indices, diagonal_indices] = float('inf')

            _, min_indices = torch.topk(distances, k=1, dim=-1, largest=False)

            unique, counts = torch.unique(min_indices, return_counts=True)
            sorted_counts, sorted_indices = torch.sort(counts, descending=True)
            sorted_unique = unique[sorted_indices]
            motif_new = torch.zeros_like(motif)
            totalnum = 0
            for j in range(self.k):
                totalnum += sorted_counts[j]

            for o in range(self.k):
                motif_new += motif * motif[:, sorted_unique[o]].unsqueeze(1) * sorted_counts[o] // totalnum

            motif_new = fold(motif_new)

            if i == 0:
                motif_feature = motif_new
            else:
                motif_feature = torch.cat((motif_feature, motif_new), dim=1)

        return motif_feature