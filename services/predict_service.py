import os
import logging
import io
import json
import uuid
import time
import imageio
from PIL import Image
import numpy
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
from .libs.hardmseg import HarDCPD, HarDMSEG
from .libs.pranet import PraNet
from datetime import datetime
from .utils import convert_base64_to_image
import base64

weights_hardmseg = 'static/checkpoints/hardmseg.pth'
weights_cpd = 'static/checkpoints/hardcpd.pth'
weights_pranet = 'static/checkpoints/pranet.pth'
image_output = 'static/images'


class PredictService():
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.transform = transforms.Compose([
            transforms.Resize((352, 352)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, model_name, image_name, base64_encode, del_image=True):
        response_json = {'message': 'Failed'}
        __PID = datetime.now().strftime('%Y%m%d_%H%M%S%f')
        # defined save output
        saved_image_name = '{}_{}.png'.format(str(uuid.uuid4()), str(int(time.time())))
        ln = os.path.join(image_output, saved_image_name)
        # convert base64_encode -> image -> read file
        image_path = convert_base64_to_image(__PID, image_name=image_name, base64_code=base64_encode)
        image_pil = Image.open(image_path).convert('RGB')
        width, height = image_pil.size
        # convert image -> tensor
        image_tensor = self.transform(image_pil).unsqueeze(0)
        # check cuda
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("Device: ", device)
        if model_name == 'hardmseg':
            print('Model: HarDMSEG')
            model = HarDMSEG()
            model.load_state_dict(torch.load(weights_hardmseg))
            model.to(device)
            model.eval()
            image_tensor = image_tensor.to(device)
            res = model(image_tensor)
            res = F.upsample(res, size=(width, height), mode='bilinear', align_corners=True)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            imageio.imwrite(ln, res)
            response_json = {
                'message': 'Success',
                'name': saved_image_name,
                'base64_encode': base64.b64encode(open(ln, 'rb').read()).decode('utf-8')
            }
        elif model_name == 'hardcpd':
            print("Model: HarDCPD")
            model = HarDCPD()
            model.load_state_dict(torch.load(weights_cpd))
            model.to(device)
            model.eval()
            image_tensor = image_tensor.to(device)
            _, res = model(image_tensor)
            res = F.upsample(res, size=(width, height), mode='bilinear', align_corners=True)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            imageio.imwrite(ln, res)
            response_json = {
                'message': 'Success',
                'name': saved_image_name,
                'base64_encode': base64.b64encode(open(ln, 'rb').read()).decode('utf-8')
            }
        elif model_name == 'pranet':
            print("Model: PraNet")
            model = PraNet()
            model.load_state_dict(torch.load(weights_pranet))
            model.to(device)
            model.eval()
            image_tensor = image_tensor.to(device)
            _, _, _, res = model(image_tensor)
            res = F.upsample(res, size=(width, height), mode='bilinear', align_corners=True)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            imageio.imwrite(ln, res)
            response_json = {
                'message': 'Success',
                'name': saved_image_name,
                'base64_encode': base64.b64encode(open(ln, 'rb').read()).decode('utf-8')
            }
        
        if del_image:
            os.remove(image_path)
            print("Delete input image")
        
        return response_json
            

