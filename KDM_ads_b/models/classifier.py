from torch import nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, input_dim=1024, mid_dim=512):
        super(Classifier, self).__init__()
        # self.dropout = nn.Dropout(0.3)
        self.linear1 = nn.Linear(input_dim, mid_dim)
        self.linear2 = nn.Linear(mid_dim, 10)

    def forward(self, x):
        # x = self.dropout(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x
