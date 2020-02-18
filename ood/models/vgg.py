import torch
import torchvision


def get_model(**kwargs_):
    model = torchvision.models.vgg16_bn(pretrained=False,**kwargs_)


    #download_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
    #model_name = download_url.split('/')[-1]
    #model_path = 'D:/Philipp/projects/generalization/experiment/model/'
    ##state_dict = torch.utils.model_zoo.load_url(download_url,
    ##    model_dir = model_path)     
    #
    #self.model = models.vgg16_bn().to(device).eval()
    #self.model.load_state_dict(state_dict)
    #self.model.double()
    return model