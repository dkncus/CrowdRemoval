import yaml
from omegaconf import OmegaConf
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.utils import move_to_device
from torch.utils.data._utils.collate import default_collate
import matplotlib.pyplot as plt
import numpy as np
import torch

class InpaintingModel():
    def __init__(self, lama_root):
        # Get the root directory of the inpainting model
        self.lama_root = lama_root

        # Set the device to run on CPU
        self.device = torch.device('cpu')

        # Load the inpainting model
        self.inpainting_model = self.load_inpainting_model()

    # Load the inpainting model (PyTorch)
    def load_inpainting_model(self):
        # Set device as the CPU - No GPU Support (yet)
        train_config_path = f'{self.lama_root}/big-lama/config.yaml'

        # Collect the training configuration and set various parameters
        config_file = open(train_config_path, 'r')
        config = yaml.safe_load(config_file)
        train_config = OmegaConf.create(config)
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        # Get the model training checkpoint file - Pretrained weights
        checkpoint_path = f'{self.lama_root}/big-lama/models/best.ckpt'

        # Load pretrained weights of the model, tranfer to CPU
        inpainting_model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        inpainting_model.freeze()
        inpainting_model.to(self.device)

        return inpainting_model

    # Use model to predict inpainting based on a given image and image mask
    def predict_inpaint(self, image, mask):
        # Fit the image to the required dimensions
        image = image.astype(np.float32) / 255
        image = image.transpose(2, 0, 1)

        # Fit the inpainting mask to the required dimensions
        mask = mask.astype(np.uint8)
        mask = mask.reshape((mask.shape[0], mask.shape[1], 1))
        mask = mask.transpose(2, 0, 1)

        # Collate the data (mask and image) into a unified image
        data = {
            'image': image,
            'mask': mask
        }
        d = move_to_device(default_collate([data]), self.device)

        # Create the inpainted image by sending the data through the model
        inpainted = self.inpainting_model(d)

        # Reconfigure the result into a usable image
        result = inpainted['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
        result = np.clip(result * 255, 0, 255).astype('uint8')

        return result