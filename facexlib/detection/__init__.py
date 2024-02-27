# import torch
# from copy import deepcopy

# from facexlib.utils import load_file_from_url
from .retinaface import RetinaFace
from tools.tpu_utils import load_model

def init_detection_model(model_name, half=False, device='cuda', model_rootpath=None, face_bmodel=None):
    if model_name == 'retinaface_resnet50':
        model = RetinaFace(network_name='resnet50', half=half, device=device, face_bmodel=face_bmodel)
        # model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth'
        # model = load_model('retinaface_resnet50_rgb_1_3_480_640.bmodel')
    # elif model_name == 'retinaface_mobile0.25':
    #     model = RetinaFace(network_name='mobile0.25', half=half, device=device)
    #     model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_mobilenet0.25_Final.pth'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    # model_path = load_file_from_url(
    #     url=model_url, model_dir='facexlib/weights', progress=True, file_name=None, save_dir=model_rootpath)
    #
    # # TODO: clean pretrained model
    # load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
    # # remove unnecessary 'module.'
    # for k, v in deepcopy(load_net).items():
    #     if k.startswith('module.'):
    #         load_net[k[7:]] = v
    #         load_net.pop(k)
    # model.load_state_dict(load_net, strict=True)
    # model.eval()
    # model = model.to(device)

    return model
