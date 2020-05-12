###########################################################################
# Created by: GSP
# Email: 947883724@qq.com
# Copyright (c) 2020
###########################################################################
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import os.path
import sys
import torch

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def make_dataset(dir, class_to_idx, extensions):
    """create a dataset

    Args:
        dir (string): Root directory path.
        class_to_index(dict): Dict with items (class_name, class_index).
        extensions(list): a list of contain all allowed file extension

    Returns:
        list: List of (sample path, class_index) tuples
    """
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class GetMateData(data.Dataset):
    def __init__(self, root1, root2, transforms1 = None, transforms2 = None, loader = pil_loader):
        classes1, class_to_idx1 = self._find_classes(root1)
        classes2, class_to_idx2 = self._find_classes(root2)
        extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp']
        samples1 = make_dataset(root1, class_to_idx1, extensions)
        samples2 = make_dataset(root2, class_to_idx2, extensions)

        if len(samples1) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root1 + "\n"
                               "Supported extensions are: " + ",".join(extensions)))
        if len(samples2) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root2 + "\n"
                               "Supported extensions are: " + ",".join(extensions)))
        
        self.loader = loader

        self.root1 = root1
        self.classes1 = classes1
        self.class_to_idx1 = class_to_idx1
        self.samples1 = samples1
        self.targets1 = [s[1] for s in samples1]
        self.transform1 = transforms1

        self.root2 = root2
        self.classes2 = classes2
        self.class_to_idx2 = class_to_idx2
        self.samples2 = samples2
        self.targets2 = [s[1] for s in samples2]
        self.transform2 = transforms2

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path1, target1 = self.samples1[index]
        path2, target2 = self.samples2[index]
        samples1 = self.loader(path1)
        samples2 = self.loader(path2)
        if self.transform1 is not None:
            sample1 = self.transform1(samples1)
        if self.transform2 is not None:
            sample2 = self.transform2(samples2)
        
        return sample1, target1, sample2, target2

    def __len__(self):
        return len(self.samples1)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: root1 - {}  root2 - {}\n'.format(self.root1, self.root2)
        tmp = '    Transforms1 (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform1.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Transforms2 (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform2.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
