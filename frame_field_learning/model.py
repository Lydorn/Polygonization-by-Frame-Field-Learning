import torch
from torchvision.models.segmentation._utils import _SimpleSegmentationModel
# from pytorch_memlab import profile, profile_every
from frame_field_learning import tta_utils


def get_out_channels(module):
    if hasattr(module, "out_channels"):
        return module.out_channels
    children = list(module.children())
    i = 1
    out_channels = None
    while out_channels is None and i <= len(children):
        last_child = children[-i]
        out_channels = get_out_channels(last_child)
        i += 1
    # If we get out of the loop but out_channels is None, then the prev child of the parent module will be checked, etc.
    return out_channels


class FrameFieldModel(torch.nn.Module):
    def __init__(self, config: dict, backbone, train_transform=None, eval_transform=None):
        """

        :param config:
        :param backbone: A _SimpleSegmentationModel network, its output features will be used to compute seg and framefield.
        :param train_transform: transform applied to the inputs when self.training is True
        :param eval_transform: transform applied to the inputs when self.training is False
        """
        super(FrameFieldModel, self).__init__()
        assert config["compute_seg"] or config["compute_crossfield"], \
            "Model has to compute at least one of those:\n" \
            "\t- segmentation\n" \
            "\t- cross-field"
        assert isinstance(backbone, _SimpleSegmentationModel), \
            "backbone should be an instance of _SimpleSegmentationModel"
        self.config = config
        self.backbone = backbone
        self.train_transform = train_transform
        self.eval_transform = eval_transform

        backbone_out_features = get_out_channels(self.backbone)

        # --- Add other modules if activated in config:
        seg_channels = 0
        if self.config["compute_seg"]:
            seg_channels = self.config["seg_params"]["compute_vertex"]\
                           + self.config["seg_params"]["compute_edge"]\
                           + self.config["seg_params"]["compute_interior"]
            self.seg_module = torch.nn.Sequential(
                torch.nn.Conv2d(backbone_out_features, backbone_out_features, 3, padding=1),
                torch.nn.BatchNorm2d(backbone_out_features),
                torch.nn.ELU(),
                torch.nn.Conv2d(backbone_out_features, seg_channels, 1),
                torch.nn.Sigmoid(),)

        if self.config["compute_crossfield"]:
            crossfield_channels = 4
            self.crossfield_module = torch.nn.Sequential(
                torch.nn.Conv2d(backbone_out_features + seg_channels, backbone_out_features, 3, padding=1),
                torch.nn.BatchNorm2d(backbone_out_features),
                torch.nn.ELU(),
                torch.nn.Conv2d(backbone_out_features, crossfield_channels, 1),
                torch.nn.Tanh(),
            )

    def inference(self, image):
        outputs = {}

        # --- Extract features for every pixel of the image with a U-Net --- #
        backbone_features = self.backbone(image)["out"]

        if self.config["compute_seg"]:
            # --- Output a segmentation of the image --- #
            seg = self.seg_module(backbone_features)
            seg_to_cat = seg.clone().detach()
            backbone_features = torch.cat([backbone_features, seg_to_cat], dim=1)  # Add seg to image features
            outputs["seg"] = seg

        if self.config["compute_crossfield"]:
            # --- Output a cross-field of the image --- #
            crossfield = 2 * self.crossfield_module(backbone_features)  # Outputs c_0, c_2 values in [-2, 2]
            outputs["crossfield"] = crossfield

        return outputs

    # @profile
    def forward(self, xb, tta=False):
        # print("\n### --- PolyRefine.forward(xb) --- ####")
        if self.training:
            if self.train_transform is not None:
                xb = self.train_transform(xb)
        else:
            if self.eval_transform is not None:
                xb = self.eval_transform(xb)

        if not tta:
            final_outputs = self.inference(xb["image"])
        else:
            final_outputs = tta_utils.tta_inference(self, xb, self.config["eval_params"]["seg_threshold"])

            # # Save image
            # image_seg_display = plot_utils.get_tensorboard_image_seg_display(image_display, final_outputs["seg"],
            #                                                                  crossfield=final_outputs["crossfield"])
            # image_seg_display = image_seg_display[1].cpu().detach().numpy().transpose(1, 2, 0)
            # skimage.io.imsave(f"out_final.png", image_seg_display)

        return final_outputs, xb
