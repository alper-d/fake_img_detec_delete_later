import torch.nn as nn
import torch
import torchvision
#from torch.nn.utils.rnn import pad_packed_sequence

RGB = 3

def gaussian_smooth(inp):
    val = torch.exp(inp)
    nominator = val * inp.shape[0] * inp.shape[1]
    denominator = torch.sum(torch.sum(torch.exp(val))) + torch.finfo(torch.float32).eps
    return torch.divide(nominator, denominator)


class MotionMag(nn.Module):
    def __init__(self):
        super(MotionMag, self).__init__()
        self.subs = nn.Conv2d(in_channels=RGB,out_channels=RGB, kernel_size=RGB, padding=1)
        self.relu1 = nn.ReLU()
        self.conv_mask1 = nn.Conv2d(in_channels=RGB,out_channels=RGB, kernel_size=RGB, padding=1)
        self.relu2 = nn.ReLU()
        self.conv_mask2 = nn.Conv2d(in_channels=RGB, out_channels=RGB, kernel_size=RGB, padding=1)
        self.relu3 = nn.ReLU()
        self.cnn_h = nn.Conv2d(in_channels=RGB, out_channels=RGB, kernel_size=RGB, padding=1)
        self.relu4 = nn.ReLU()
        nn.init.xavier_uniform_(self.subs.weight)
        nn.init.xavier_uniform_(self.conv_mask1.weight)
        nn.init.xavier_uniform_(self.conv_mask2.weight)
        nn.init.xavier_uniform_(self.cnn_h.weight)

    def forward(self, f_prev, f):
        out = self.relu1(self.subs(torch.sub(f, f_prev)))
        out2 = self.relu3(self.conv_mask2(self.relu2(self.conv_mask1(out))))
        #out3 = torch.multiply(gaussian_smooth(out2), out)
        out3 = torch.multiply(torchvision.transforms.functional.gaussian_blur(out2, kernel_size=3), out)
        return self.relu4(self.cnn_h(out3)) + out3 + f_prev


class FakeNet(nn.Module):
    def __init__(self, image_height=70, image_width=50, sequence_len=299):
        self.height = image_height
        self.width = image_width
        self.sequence_len = sequence_len
        super(FakeNet, self).__init__()
        self.feature_extract = nn.Conv2d(in_channels=RGB, out_channels=RGB, kernel_size=RGB, stride=1, padding= 1)
        self.cnn_relu = nn.ReLU()
        self.manipulator = MotionMag()
        #TODO CHANGE 299
        self.temporal_extractor = nn.LSTM(input_size=RGB*self.height*self.width, hidden_size=48, num_layers=1, batch_first=True)
        self.linear1 = nn.Linear(in_features=48*self.sequence_len, out_features=1)
        # self.relu1 = nn.ReLU()
        # self.linear2 = nn.Linear(in_features=100, out_features=1)
        nn.init.xavier_uniform_(self.feature_extract.weight)
        nn.init.xavier_uniform_(self.linear1.weight)
        # torch.nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.temporal_extractor.weight_ih_l0)

    def forward(self, batch):
        # pad_packed_sequence(batch, )

        # feats allocation prevents following ->
        # "one of the variables needed for gradient computation has been modified by an inplace operation"
        feats = torch.empty_like(batch)
        for i in range(batch.shape[1]):
            feats[:, i, ...] = self.cnn_relu(self.feature_extract(batch[:,i,...]))
            if i > 0:
                feats[:, i-1, ...] = self.manipulator(feats[:, i-1, ...], feats[:, i, ...])
        #del batch
        out = self.temporal_extractor(feats.view(feats.shape[0], feats.shape[1], -1))[0]
        out1 = self.linear1(out.reshape(out.shape[0], -1))
        # out2 = self.linear2(out1)

        return out1

def main():
    model = FakeNet()
    print('model:\n', model)


if __name__ == '__main__':
    main()