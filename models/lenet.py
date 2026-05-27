'''LeNet in PyTorch.

LeNet is a good first CNN because it has only two convolution blocks followed by
three fully connected layers. The local Digits tutorial resizes images to
3x32x32 so this network can share the same input shape as CIFAR-style models.
'''
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # Convolution layers learn local visual patterns. The input has 3
        # channels, and each 5x5 kernel produces a feature map.
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # After two 2x2 max-pooling operations, a 32x32 image becomes 5x5 with
        # 16 channels: [batch, 16, 5, 5]. Flattening gives 16*5*5 features.
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)

        # The final layer returns 10 logits, one for each class label 0..9.
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        # Input x shape: [batch_size, 3, 32, 32].
        out = F.relu(self.conv1(x))

        # Pooling halves spatial size and keeps the strongest activation in
        # each small region.
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)

        # Linear layers expect [batch_size, features], not image-shaped tensors.
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))

        # Do not apply softmax here. CrossEntropyLoss expects raw logits.
        out = self.fc3(out)
        return out
