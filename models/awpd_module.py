import pywt
import torch


class AdaptiveWPD(torch.nn.Module):
    def __init__(self, max_level=5, entropy_threshold=0.85):
        super().__init__()
        self.max_level = max_level
        self.entropy_threshold = entropy_threshold

    def _calculate_entropy(self, coeffs):
        energy = torch.sum(coeffs ** 2, dim=-1)
        prob = energy / torch.sum(energy, dim=-1, keepdim=True)
        return -torch.sum(prob * torch.log2(prob + 1e-8), dim=-1)

    def forward(self, x):
        # x: (B, C, H, W)
        batch_size, channels = x.shape[0], x.shape[1]
        results = []

        for b in range(batch_size):
            for c in range(channels):
                signal = x[b, c].cpu().numpy()
                wp = pywt.WaveletPacket2D(
                    data=signal,
                    wavelet='db4',
                    mode='symmetric',
                    maxlevel=self.max_level
                )

                # Adaptive level selection
                optimal_level = 1
                for level in range(1, self.max_level + 1):
                    nodes = [node.path for node in wp.get_level(level, 'natural')]
                    entropy_sum = 0
                    for node in nodes:
                        coeff = torch.tensor(wp[node].data)
                        entropy_sum += self._calculate_entropy(coeff)

                    if entropy_sum / len(nodes) < self.entropy_threshold:
                        optimal_level = level
                    else:
                        break

                # Extract optimal level coefficients
                selected_coeffs = []
                for node in wp.get_level(optimal_level, 'natural'):
                    selected_coeffs.append(torch.tensor(node.data))

                results.append(torch.stack(selected_coeffs, dim=0))

        return torch.stack(results).to(x.device)  # (B*C, N_coeff, H', W')
