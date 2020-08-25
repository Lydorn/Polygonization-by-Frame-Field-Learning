import os

import torch
import torchvision

from lydorn_utils import print_utils


def get_backbone(backbone_params):
    set_download_dir()
    if backbone_params["name"] == "unet":
        from torchvision.models.segmentation._utils import _SimpleSegmentationModel
        from frame_field_learning.unet import UNetBackbone

        backbone = UNetBackbone(backbone_params["input_features"], backbone_params["features"])
        backbone = _SimpleSegmentationModel(backbone, classifier=torch.nn.Identity())
    elif backbone_params["name"] == "fcn50":
        backbone = torchvision.models.segmentation.fcn_resnet50(pretrained=backbone_params["pretrained"],
                                                                num_classes=21)
        backbone.classifier = torch.nn.Sequential(*list(backbone.classifier.children())[:-1],
                                                  torch.nn.Conv2d(512, backbone_params["features"], kernel_size=(1, 1),
                                                                  stride=(1, 1)))
    elif backbone_params["name"] == "fcn101":
        backbone = torchvision.models.segmentation.fcn_resnet101(pretrained=backbone_params["pretrained"],
                                                                 num_classes=21)
        backbone.classifier = torch.nn.Sequential(*list(backbone.classifier.children())[:-1],
                                                  torch.nn.Conv2d(512, backbone_params["features"], kernel_size=(1, 1),
                                                                  stride=(1, 1)))

    elif backbone_params["name"] == "deeplab50":
        backbone = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=backbone_params["pretrained"],
                                                                      num_classes=21)
        backbone.classifier = torch.nn.Sequential(*list(backbone.classifier.children())[:-1],
                                                  torch.nn.Conv2d(256, backbone_params["features"], kernel_size=(1, 1),
                                                                  stride=(1, 1)))
    elif backbone_params["name"] == "deeplab101":
        backbone = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=backbone_params["pretrained"],
                                                                       num_classes=21)
        backbone.classifier = torch.nn.Sequential(*list(backbone.classifier.children())[:-1],
                                                  torch.nn.Conv2d(256, backbone_params["features"], kernel_size=(1, 1),
                                                                  stride=(1, 1)))
    elif backbone_params["name"] == "unet_resnet":
        from torchvision.models.segmentation._utils import _SimpleSegmentationModel
        from frame_field_learning.unet_resnet import UNetResNetBackbone

        backbone = UNetResNetBackbone(backbone_params["encoder_depth"], num_filters=backbone_params["num_filters"],
                                      dropout_2d=backbone_params["dropout_2d"],
                                      pretrained=backbone_params["pretrained"],
                                      is_deconv=backbone_params["is_deconv"])
        backbone = _SimpleSegmentationModel(backbone, classifier=torch.nn.Identity())

    elif backbone_params["name"] == "ictnet":
        from torchvision.models.segmentation._utils import _SimpleSegmentationModel
        from frame_field_learning.ictnet import ICTNetBackbone

        backbone = ICTNetBackbone(in_channels=backbone_params["in_channels"],
                                  out_channels=backbone_params["out_channels"],
                                  preset_model=backbone_params["preset_model"],
                                  dropout_2d=backbone_params["dropout_2d"],
                                  efficient=backbone_params["efficient"])
        backbone = _SimpleSegmentationModel(backbone, classifier=torch.nn.Identity())
    else:
        print_utils.print_error("ERROR: config[\"backbone_params\"][\"name\"] = \"{}\" is an unknown backbone!"
                                "If it is a new backbone you want to use, "
                                "add it in backbone.py's get_backbone() function.".format(backbone_params["name"]))
        raise RuntimeError("Specified backbone {} unknown".format(backbone_params["name"]))
    return backbone


def set_download_dir():
    os.environ['TORCH_HOME'] = 'models'  # setting the environment variable
