from io import BytesIO 

import lmdb
from PIL import Image
from torch.utils.data import Dataset


class MultiResolutionDataset(Dataset):
    """
    A PyTorch dataset class that loads image data from an LMDB database.
    It retrieves images based on resolution and applies transformations.

    Args:
        path (str): Path to the LMDB database.
        trnasform (callable): Function to apply image transformations.
        resolution (int, optional): Image resolution to use (default: 8)

    Example:
        dataset = MultiResolutionDataset(path="/path/to/lmdb", transform=my_transform, resolution=256)
        img = dataset[0]

    Returns:
        PIL.Image or Tensor: Trnasformed image data.
    """

    def __init__(self, path, transform, resolution=8):
        """
        Initializes the datset and sets up the LMDB enviroment.

        Args:
            path (str): Path to the LMDB datset.
            transform (callable): Function to apply image transformations.
            resolution (int): Image resolution to use (default: 8) 
        """
        # Open the LMDB database in read-only mode.
        self.env = lmdb.open(
            path,
            max_readers=32, # 32 concurrent readers.
            readonly=True, # Open in read-only mode.
            lock=False, # Enable multi-process access.
            readahead=False, # Disable read-ahead for memory efficiency
            meminit=False, # Disable memory initialization for performance.
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of images in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return self.length

    def __getitem__(self, index):
        """
        Retrieves and transforms an image at the given index.

        Args:
            index (int): The index of the image to retrieve.

        Returns:
            Transformed Image: The processed image data.
        """
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        # Convert the binary data to an image.
        buffer = BytesIO(img_bytes)
        img = Image.open(buffer) # Load the image using Pillow
        img = self.transform(img) # Apply transformations

        return img