import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, encoding_dim),
            # nn.ReLU()
        )
        self.decoder = nn.Sequential(
            # nn.Linear(encoding_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()  # 使用 Sigmoid 激活函数将输出限制在 [0, 1] 范围内
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

