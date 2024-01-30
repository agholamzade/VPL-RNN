
import torch
from PIL import Image
from torch.utils.data import Dataset
import os

class GratingDataset(Dataset):
    """Grating dataset."""

    def __init__(self, root_dir, ref_dir, transform=None, num_seqs=1000):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            ref_orientation (int): The orientation of the reference grating
            separation_angle (int): The separation angle between the reference and test grating
            contrast (int): The contrast of the grating
            phase (int): The phase of the grating
            spatial_freq (int): The spatial frequency of the grating

        """

        self.root_dir = root_dir
        self.ref_dir = ref_dir
        self.transform = transform
        self.num_seqs = num_seqs
        self.image_list = os.listdir(root_dir)

    def __len__(self):
        return self.num_seqs


    def __getitem__(self, idx):
        # get the mode of idx and length of image list
        #mode = idx % len(self.image_list)
        mode = idx%2

        img_name = os.path.join(self.root_dir, self.image_list[mode])
        ref_name = self.ref_dir

        ref_label = 0.0

        # set the label of the image. If the image is rotated CW, the label is 1. If the image is rotated CCW, the label is 0.
        if 'CCW' in img_name:
            img_label = -1.0
        elif 'CW' in img_name:
            img_label = 1.0

        image = Image.open(img_name)
        ref_image = Image.open(ref_name)

        ref_images = torch.stack([self.transform(ref_image) for i in range(5)])
        images = torch.stack([self.transform(image) for i in range(5)])

        image_seq = torch.cat((ref_images, images))

        ref_labels = torch.full((5,), ref_label)
        img_labels = torch.full((5,), img_label)

        labels = torch.cat((ref_labels,img_labels))


        return image_seq, labels # dim 2 x 3 x w x h (batch dimension is first)
    # batch size of 10, 2 images in each (CW & CCW), flatten and split into 20 images 10 x 2 x 3 x w x h --> 20 x 3 x w x h


