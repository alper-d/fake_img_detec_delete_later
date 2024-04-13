import torch.utils.data as data
import os
import skimage
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision
import torch
import json
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import imutils


class CustomPad:
    def __init__(self):
        """
        We discarded using custom padding and decided to apply during preprocessing stage.
        Reason is IO drastically limits the performance and 70x50 is manageable by memory
        """
        self.height = 350
        self.width = 250

    def __call__(self, image):
        h, w = image.shape[1], image.shape[2]

        if image.shape[1] < self.height and image.shape[2] < self.width:
            if h / w > (7 / 5):
                image = imutils.resize(image.numpy().transpose(1, 2, 0), height=self.height)
            else:
                image = imutils.resize(image.numpy().transpose(1, 2, 0), width=self.width)
            h = int(image.shape[1])
            w = int(image.shape[2])
            image = torch.from_numpy(image.transpose(2, 0, 1))

        extra_top = (self.height - h) // 2
        extra_bottom = (self.height - h) - extra_top
        extra_left = (self.width - w) // 2
        extra_right = (self.width - w) - extra_left
        padding = (extra_left, extra_top, extra_right, extra_bottom)

        return F.pad(image, padding, 255, 'constant')


class FakeDetectDataset(data.Dataset):
    def __init__(self, root_path, transforms_list=None, image_height=70, image_width=50):
        self.root_path = root_path
        self.width = image_width
        self.seq_len = 299
        self.height = image_height
        self.video_paths = sorted([f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))])
        # I edited metadata by making "FAKE" -> 0 , "REAL" -> 1
        with open(os.path.join(root_path, "metadata_edited.json"), 'r') as j:
            self.metadata = json.loads(j.read())
        if transforms_list != None:
            self.transformations = transforms.Compose(transforms_list)
        self.image_dict = {}
        dataset_tensor = torch.load(os.path.join(root_path, "tensor_chunk.pt"), weights_only=True)
        print(dataset_tensor.shape)
        for i, video_name in enumerate(self.video_paths):
            self.image_dict[video_name] = dataset_tensor[i, :].squeeze() / 255

    def __getitem__(self, index):
        video_name = self.video_paths[index]
        print(video_name)
        return self.image_dict[video_name], self.metadata[video_name + ".mp4"]["label"]

    def __len__(self):
        return len(self.video_paths)
