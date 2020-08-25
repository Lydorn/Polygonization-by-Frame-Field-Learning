from collections import OrderedDict

import PIL
import numpy as np
import torch
import torchvision
import kornia

import torch_lydorn.kornia
import torch_lydorn.torchvision

from lydorn_utils import print_utils


class Print(object):
    """Convert polygons to a single graph"""

    def __init__(self):
        pass

    def __call__(self, sample):
        print("\n")
        print(sample.keys())
        for key, item in sample.items():
            if type(item) == np.ndarray or type(item) == torch.Tensor:
                if len(item.shape):
                    print(key, type(item), item.shape, item.dtype, item.min(), item.max())
                else:
                    print(key, type(item), item, item.dtype, item.min(), item.max())
            elif type(item) == PIL.Image.Image:
                print(key, type(item), item.size, item.mode, np.array(item).min(), np.array(item).max())
            elif type(item) == list:
                print(key, type(item[0]), len(item))
        # exit()
        # print(sample["image"].dtype)
        # print(sample["image"].shape)
        # print(sample["image"].min())
        # print(sample["image"].max())
        # for key, value in sample.items():
        #     print(key + ":")
        #     if type(value) == np.ndarray:
        #         print(value.shape)
        #     elif type(value) == list:
        #         print(len(value))
        #     else:
        #         print("a")
        return sample


class CudaDataAugmentation(object):
    def __init__(self, input_patch_size: int, vflip: bool, affine: bool, scaling: list, color_jitter: bool):
        self.input_patch_size = input_patch_size
        self.vflip = vflip
        self.affine = affine
        self.scaling = scaling
        self.color_jitter = None
        if color_jitter:
            self.color_jitter = kornia.augmentation.ColorJitter(brightness=0.05, contrast=0.05, saturation=.5, hue=.1)
        self.tensor_keys_bilinear = ["image", "gt_polygons_image", "distances",
                                     "valid_mask"]  # Affine transform applied with bilinear sampling
        self.tensor_keys_nearest = ["sizes", "gt_crossfield_angle"]  # Affine transform applied with nearest sampling

    @staticmethod
    def get_slices(batch, keys, last_slice_stop=0):
        slices = OrderedDict()
        for key in keys:
            s = slice(last_slice_stop, last_slice_stop + batch[key].shape[1])
            last_slice_stop += batch[key].shape[1]
            slices[key] = s
        return slices

    def __call__(self, batch):
        with torch.no_grad():
            batch_size, im_channels, height, width = batch["image"].shape
            device = batch["image"].device
            batch["valid_mask"] = torch.ones((batch_size, 1, height, width), dtype=torch.float,
                                             device=device)  # Apply losses only when valid_mask is True (pad with 0 when rotating)

            # Combine all images into one for faster/easier processing (store slices to separate them later on):
            tensor_keys_bilinear = [key for key in self.tensor_keys_bilinear if key in batch]
            tensor_keys_nearest = [key for key in self.tensor_keys_nearest if key in batch]
            tensor_keys = tensor_keys_bilinear + tensor_keys_nearest
            combined = torch.cat([batch[tensor_key] for tensor_key in tensor_keys], dim=1)
            slices_bilinear = self.get_slices(batch, tensor_keys_bilinear, last_slice_stop=0)
            slices_nearest = self.get_slices(batch, tensor_keys_nearest,
                                             last_slice_stop=slices_bilinear[tensor_keys_bilinear[-1]].stop)
            bilinear_slice = slice(slices_bilinear[tensor_keys_bilinear[0]].start,
                                   slices_bilinear[tensor_keys_bilinear[-1]].stop)
            nearest_slice = slice(slices_nearest[tensor_keys_nearest[0]].start,
                                  slices_nearest[tensor_keys_nearest[-1]].stop)

            # Rotation (and translation)
            if self.affine:
                angle: torch.Tensor = torch.empty(batch_size, device=device).uniform_(-180, 180)
                # To include corner pixels if angle=45 (coords are between -1 and 1 for grid_sample):
                max_offset = np.sqrt(2) - 1
                offset: torch.Tensor = torch.empty((batch_size, 2), device=device).uniform_(-max_offset, max_offset)
                downscale_factor = None
                if self.scaling is not None:
                    downscale_factor: torch.Tensor = torch.empty(batch_size, device=device).uniform_(*self.scaling)
                affine_grid = torch_lydorn.kornia.geometry.transform.get_affine_grid(combined, angle, offset,
                                                                                     downscale_factor)
                combined[:, bilinear_slice, ...] = \
                    torch.nn.functional.grid_sample(combined[:, bilinear_slice, ...],
                                                    affine_grid, mode='bilinear')



                # Rotate sizes and anglefield with mode='nearest'
                # because it makes no sense to interpolate size values and angle values:
                combined[:, nearest_slice, ...] = torch.nn.functional.grid_sample(combined[:, nearest_slice, ...],
                                                                                  affine_grid, mode='nearest')

                # Additionally the angle field's values themselves have to be rotated:
                combined[:, slices_nearest["gt_crossfield_angle"],
                ...] = torch_lydorn.torchvision.transforms.functional.rotate_anglefield(
                    combined[:, slices_nearest["gt_crossfield_angle"], ...], angle)

                # The sizes and distances should be adjusted as well because of the scaling.
                if downscale_factor is not None:
                    if "sizes" in slices_nearest:
                        size_equals_one = combined[:, slices_nearest["sizes"], ...] == 1
                        combined[:, slices_nearest["sizes"], :, :] /= downscale_factor[:, None, None, None]
                        combined[:, slices_nearest["sizes"], ...][size_equals_one] = 1
                    if "distances" in slices_bilinear:
                        distance_equals_one = combined[:, slices_bilinear["distances"], ...] == 1
                        combined[:, slices_bilinear["distances"], :, :] /= downscale_factor[:, None, None, None]
                        combined[:, slices_bilinear["distances"], ...][distance_equals_one] = 1

            # Center crop
            if self.input_patch_size is not None:
                prev_image_norm = combined.shape[2] + combined.shape[3]
                combined = torch_lydorn.torchvision.transforms.functional.center_crop(combined, self.input_patch_size)
                current_image_norm = combined.shape[2] + combined.shape[3]
                # Sizes and distances are affected by this because they are relative to the image's size.
                # All non-one pixels have to be renormalized:
                size_ratio = prev_image_norm / current_image_norm
                if "sizes" in slices_nearest:
                    combined[:, slices_nearest["sizes"], ...][
                        combined[:, slices_nearest["sizes"], ...] != 1] *= size_ratio
                if "distances" in slices_bilinear:
                    combined[:, slices_bilinear["distances"], ...][
                        combined[:, slices_bilinear["distances"], ...] != 1] *= size_ratio

            # vflip
            if self.vflip:
                to_flip: torch.Tensor = torch.empty(batch_size, device=device).uniform_(0, 1) < 0.5
                combined[to_flip] = kornia.geometry.transform.vflip(combined[to_flip])
                combined[
                    to_flip, slices_nearest[
                        "gt_crossfield_angle"], ...] = torch_lydorn.torchvision.transforms.functional.vflip_anglefield(
                    combined[to_flip, slices_nearest["gt_crossfield_angle"], ...])

            # Split data:
            batch["image"] = combined[:, slices_bilinear["image"], ...]
            if "gt_polygons_image" in slices_bilinear:
                batch["gt_polygons_image"] = combined[:, slices_bilinear["gt_polygons_image"], ...]
            if "distances" in slices_bilinear:
                batch["distances"] = combined[:, slices_bilinear["distances"], ...]
            batch["valid_mask"] = 0.99 < combined[:, slices_bilinear["valid_mask"],
                                         ...]  # Take a very high threshold to remove fuzzy pixels

            if "sizes" in slices_nearest:
                batch["sizes"] = combined[:, slices_nearest["sizes"], ...]
            batch["gt_crossfield_angle"] = combined[:, slices_nearest["gt_crossfield_angle"], ...]

            # Color jitter
            if self.color_jitter is not None and batch["image"].shape[1] == 3:
                batch["image"] = self.color_jitter(batch["image"])

            # --- Zero padding of sizes and distances is not correct, they should be padded with ones:
            if self.affine:
                if "sizes" in slices_nearest:
                    batch["sizes"][~batch["valid_mask"]] = 1
                if "distances" in slices_bilinear:
                    batch["distances"][~batch["valid_mask"]] = 1

        return batch


class CudaCrop(object):
    def __init__(self, input_patch_size: int):
        self.input_patch_size = input_patch_size
        self.tensor_keys = ["image", "gt_polygons_image", "distances", "valid_mask", "sizes", "gt_crossfield_angle"]

    def __call__(self, batch):
        for tensor_key in self.tensor_keys:
            if tensor_key in batch:
                batch[tensor_key] = torch_lydorn.torchvision.transforms.functional.center_crop(batch[tensor_key],
                                                                                               self.input_patch_size)
        return batch


def get_offline_transform(config, augmentations=False, to_patches=True):
    data_patch_size = config["dataset_params"]["data_patch_size"] if augmentations else config["dataset_params"][
        "input_patch_size"]
    transform_list = [
        torch_lydorn.torchvision.transforms.Map(
            transform=torch_lydorn.torchvision.transforms.TransformByKey(
                transform=torchvision.transforms.Compose([
                    torch_lydorn.torchvision.transforms.RemoveDoubles(epsilon=0.01),
                    torch_lydorn.torchvision.transforms.FilterPolyVertexCount(min=3),
                    torch_lydorn.torchvision.transforms.ApproximatePolygon(tolerance=0.01),
                    torch_lydorn.torchvision.transforms.FilterPolyVertexCount(min=3)
                ]), key="gt_polygons")),

        torch_lydorn.torchvision.transforms.FilterEmptyPolygons(key="gt_polygons"),
    ]
    if to_patches:
        transform_list.extend([
            torch_lydorn.torchvision.transforms.ToPatches(stride=config["dataset_params"]["input_patch_size"],
                                                          size=data_patch_size),
            torch_lydorn.torchvision.transforms.FilterEmptyPolygons(key="gt_polygons"),
        ])
    transform_list.extend([
        torch_lydorn.torchvision.transforms.Map(
            transform=torchvision.transforms.Compose([
                torch_lydorn.torchvision.transforms.TransformByKey(
                    transform=torch_lydorn.torchvision.transforms.Rasterize(fill=True, edges=True, vertices=True,
                                                                            line_width=4, antialiasing=True),
                    key=["image", "gt_polygons"], outkey="gt_polygons_image"),
                torch_lydorn.torchvision.transforms.TransformByKey(
                    transform=torch_lydorn.torchvision.transforms.AngleFieldInit(line_width=6),
                    key=["image", "gt_polygons"],
                    outkey="gt_crossfield_angle")
            ])),
    ])
    offline_transform = torchvision.transforms.Compose(transform_list)
    return offline_transform


def get_offline_transform_patch(raster: bool = True, fill: bool = True, edges: bool = True, vertices: bool = True,
                                distances: bool = True, sizes: bool = True, angle_field: bool = True):
    transform_list = []
    if raster:
        if not distances and not sizes:
            rasterize_transform = torch_lydorn.torchvision.transforms.TransformByKey(
                transform=torch_lydorn.torchvision.transforms.Rasterize(fill=fill, edges=edges, vertices=vertices,
                                                                        line_width=4, antialiasing=True,
                                                                        return_distances=False,
                                                                        return_sizes=False),
                key=["image", "gt_polygons"], outkey="gt_polygons_image")
        elif distances and sizes:
            rasterize_transform = torch_lydorn.torchvision.transforms.TransformByKey(
                    transform=torch_lydorn.torchvision.transforms.Rasterize(fill=fill, edges=edges, vertices=vertices,
                                                                            line_width=4, antialiasing=True,
                                                                            return_distances=True,
                                                                            return_sizes=True),
                    key=["image", "gt_polygons"], outkey=["gt_polygons_image", "distances", "sizes"])
        else:
            raise NotImplementedError
        transform_list.append(rasterize_transform)
    if angle_field:
        transform_list.append(
            torch_lydorn.torchvision.transforms.TransformByKey(
                transform=torch_lydorn.torchvision.transforms.AngleFieldInit(line_width=6),
                key=["image", "gt_polygons"],
                outkey="gt_crossfield_angle")
        )

    return torchvision.transforms.Compose(transform_list)


def get_online_cpu_transform(config, augmentations=False):
    if augmentations and config["data_aug_params"]["device"] == "cpu":
        print_utils.print_error("ERROR: CPU augmentations is not supported anymore. "
                                "Look at CudaDataAugmentation to see what additional augs would need to be implemented.")
        raise NotImplementedError
    online_transform_list = []
    # Convert to PIL images
    if not augmentations \
            or (augmentations and config["data_aug_params"]["device"] == "cpu"):
        online_transform_list.extend([
            torch_lydorn.torchvision.transforms.TransformByKey(transform=torchvision.transforms.ToPILImage(),
                                                               key="image"),
            torch_lydorn.torchvision.transforms.TransformByKey(transform=torchvision.transforms.ToPILImage(),
                                                               key="gt_polygons_image"),
            torch_lydorn.torchvision.transforms.TransformByKey(transform=torchvision.transforms.ToPILImage(),
                                                               key="gt_crossfield_angle"),
        ])
    # Add rotation data augmentation:
    if augmentations and config["data_aug_params"]["device"] == "cpu" and \
            config["data_aug_params"]["affine"]:
        online_transform_list.extend([
            torch_lydorn.torchvision.transforms.TransformByKey(
                transform=torch_lydorn.torchvision.transforms.SampleUniform(-180, 180),
                outkey="rand_angle"),
            torch_lydorn.torchvision.transforms.TransformByKey(transform=torchvision.transforms.functional.rotate,
                                                               key=["image", "rand_angle"], outkey="image",
                                                               resample=PIL.Image.BILINEAR),
            torch_lydorn.torchvision.transforms.TransformByKey(transform=torchvision.transforms.functional.rotate,
                                                               key=["gt_polygons_image", "rand_angle"],
                                                               outkey="gt_polygons_image",
                                                               resample=PIL.Image.BILINEAR),
            torch_lydorn.torchvision.transforms.TransformByKey(transform=torchvision.transforms.functional.rotate,
                                                               key=["gt_crossfield_angle", "rand_angle"],
                                                               outkey="gt_crossfield_angle",
                                                               resample=PIL.Image.NEAREST),
        ])

    # Crop to final size
    if not augmentations \
            or (augmentations and config["data_aug_params"]["device"] == "cpu"):
        if "input_patch_size" in config["dataset_params"]:
            online_transform_list.extend([
                torch_lydorn.torchvision.transforms.TransformByKey(
                    transform=torchvision.transforms.CenterCrop(config["dataset_params"]["input_patch_size"]),
                    key="image"),
                torch_lydorn.torchvision.transforms.TransformByKey(
                    transform=torchvision.transforms.CenterCrop(config["dataset_params"]["input_patch_size"]),
                    key="gt_polygons_image"),
                torch_lydorn.torchvision.transforms.TransformByKey(
                    transform=torchvision.transforms.CenterCrop(config["dataset_params"]["input_patch_size"]),
                    key="gt_crossfield_angle"),
                torch_lydorn.torchvision.transforms.TransformByKey(
                    transform=torch_lydorn.torchvision.transforms.CenterCrop(
                        config["dataset_params"]["input_patch_size"]),
                    key="distances"),
                torch_lydorn.torchvision.transforms.TransformByKey(
                    transform=torch_lydorn.torchvision.transforms.CenterCrop(
                        config["dataset_params"]["input_patch_size"]),
                    key="sizes"),
            ])

    # Random Horizontal flip:
    if augmentations and config["data_aug_params"]["device"] == "cpu" and \
            config["data_aug_params"]["vflip"]:
        online_transform_list.extend([
            torch_lydorn.torchvision.transforms.TransformByKey(
                transform=torch_lydorn.torchvision.transforms.RandomBool(p=0.5),
                outkey="rand_flip"),
            torch_lydorn.torchvision.transforms.TransformByKey(
                transform=torch_lydorn.torchvision.transforms.ConditionApply(
                    transform=torchvision.transforms.functional.vflip),
                key=["image", "rand_flip"], outkey="image"),
            torch_lydorn.torchvision.transforms.TransformByKey(
                transform=torch_lydorn.torchvision.transforms.ConditionApply(
                    transform=torchvision.transforms.functional.vflip),
                key=["gt_polygons_image", "rand_flip"], outkey="gt_polygons_image"),
            torch_lydorn.torchvision.transforms.TransformByKey(
                transform=torch_lydorn.torchvision.transforms.ConditionApply(
                    transform=torchvision.transforms.functional.vflip),
                key=["gt_crossfield_angle", "rand_flip"], outkey="gt_crossfield_angle"),
        ])

    # Other augs:
    if augmentations and config["data_aug_params"]["device"] == "cpu" and \
            config["data_aug_params"]["color_jitter"]:
        online_transform_list.append(
            torch_lydorn.torchvision.transforms.TransformByKey(
                transform=torchvision.transforms.ColorJitter(brightness=0.05, contrast=0.05,
                                                             saturation=.5, hue=.1),
                key="image")
        )
    # Convert to PyTorch tensors:
    online_transform_list.extend([
        # Print(),
        torch_lydorn.torchvision.transforms.TransformByKey(transform=torch_lydorn.torchvision.transforms.ToTensor(),
                                                           key="image"),
        torch_lydorn.torchvision.transforms.TransformByKey(
            transform=torchvision.transforms.Lambda(torch.from_numpy),
            key="image_mean"),
        torch_lydorn.torchvision.transforms.TransformByKey(
            transform=torchvision.transforms.Lambda(torch.from_numpy),
            key="image_std"),
        torch_lydorn.torchvision.transforms.TransformByKey(transform=torch_lydorn.torchvision.transforms.ToTensor(),
                                                           key="gt_polygons_image", ignore_key_error=True),
        torch_lydorn.torchvision.transforms.TransformByKey(torch_lydorn.torchvision.transforms.ToTensor(),
                                                           key="gt_crossfield_angle", ignore_key_error=True),
        torch_lydorn.torchvision.transforms.TransformByKey(transform=torch_lydorn.torchvision.transforms.ToTensor(),
                                                           key="distances", ignore_key_error=True),
        torch_lydorn.torchvision.transforms.TransformByKey(transform=torch_lydorn.torchvision.transforms.ToTensor(),
                                                           key="sizes", ignore_key_error=True),
    ])

    online_transform_list.append(
        torch_lydorn.torchvision.transforms.RemoveKeys(keys=["gt_polygons"])
    )

    online_transform = torchvision.transforms.Compose(online_transform_list)
    return online_transform


def get_eval_online_cpu_transform():
    online_transform = torchvision.transforms.Compose([
        # Print(),
        torch_lydorn.torchvision.transforms.TransformByKey(transform=torch_lydorn.torchvision.transforms.ToTensor(),
                                                           key="image"),
        torch_lydorn.torchvision.transforms.TransformByKey(transform=torchvision.transforms.Lambda(
            torch.from_numpy),
            key="image_mean"),
        torch_lydorn.torchvision.transforms.TransformByKey(transform=torchvision.transforms.Lambda(
            torch.from_numpy),
            key="image_std"),
        torch_lydorn.torchvision.transforms.TransformByKey(transform=torch_lydorn.torchvision.transforms.ToTensor(),
                                                           key="gt_polygons_image"),
        torch_lydorn.torchvision.transforms.TransformByKey(torch_lydorn.torchvision.transforms.ToTensor(),
                                                           key="gt_crossfield_angle"),
        torch_lydorn.torchvision.transforms.TransformByKey(transform=torch_lydorn.torchvision.transforms.ToTensor(),
                                                           key="distances"),
        torch_lydorn.torchvision.transforms.TransformByKey(transform=torch_lydorn.torchvision.transforms.ToTensor(),
                                                           key="sizes"),
        torch_lydorn.torchvision.transforms.RemoveKeys(keys=["gt_polygons"])
    ])
    return online_transform


def get_online_cuda_transform(config, augmentations=False):
    device_transform_list = [
        torch_lydorn.torchvision.transforms.TransformByKey(transform=torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda tensor: tensor.float().div(255))
        ]), key="image"),
        torch_lydorn.torchvision.transforms.TransformByKey(
            transform=torchvision.transforms.Lambda(lambda tensor: tensor.float().div(255)),
            key="gt_polygons_image"),
        torch_lydorn.torchvision.transforms.TransformByKey(
            transform=torchvision.transforms.Lambda(lambda tensor: np.pi * tensor.float().div(255)),
            key="gt_crossfield_angle"),
        torch_lydorn.torchvision.transforms.TransformByKey(
            transform=torchvision.transforms.Lambda(lambda tensor: tensor.float()),
            key="distances", ignore_key_error=True),
        torch_lydorn.torchvision.transforms.TransformByKey(
            transform=torchvision.transforms.Lambda(lambda tensor: tensor.float()),
            key="sizes", ignore_key_error=True),
        torch_lydorn.torchvision.transforms.TransformByKey(
            transform=torchvision.transforms.Lambda(lambda tensor: tensor.float()),
            key="class_freq", ignore_key_error=True),
    ]
    if augmentations and config["data_aug_params"]["device"] == "cpu":
        if config["data_aug_params"]["affine"]:
            # Rotate angle field
            device_transform_list.append(
                torch_lydorn.torchvision.transforms.TransformByKey(
                    transform=torch_lydorn.torchvision.transforms.functional.rotate_anglefield,
                    key=["gt_crossfield_angle", "rand_angle"],
                    outkey="gt_crossfield_angle"))
        if config["data_aug_params"]["vflip"]:
            device_transform_list.append(
                torch_lydorn.torchvision.transforms.TransformByKey(
                    transform=torch_lydorn.torchvision.transforms.functional.vflip_anglefield,
                    key=["gt_crossfield_angle", "rand_flip"],
                    outkey="gt_crossfield_angle"))
    if config["data_aug_params"]["device"] == "cuda":
        input_patch_size = config["dataset_params"]["input_patch_size"] if "input_patch_size" in config[
            "dataset_params"] else None  # No crop if None
        if augmentations:
            device_transform_list.append(CudaDataAugmentation(input_patch_size,
                                                              config["data_aug_params"]["vflip"],
                                                              config["data_aug_params"]["affine"],
                                                              config["data_aug_params"]["scaling"],
                                                              config["data_aug_params"]["color_jitter"]))
        elif input_patch_size is not None:
            device_transform_list.append(CudaCrop(input_patch_size))
    device_transform_list.append(
        torch_lydorn.torchvision.transforms.TransformByKey(
            transform=torch_lydorn.torchvision.transforms.functional.batch_normalize,
            key=["image", "image_mean", "image_std"],
            outkey="image"), )
    device_transform = torchvision.transforms.Compose(device_transform_list)
    return device_transform


def get_eval_online_cuda_transform(config):
    device_transform_list = [
        torch_lydorn.torchvision.transforms.TransformByKey(transform=torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda tensor: tensor.float().div(255))
        ]), key="image"),
        torch_lydorn.torchvision.transforms.TransformByKey(
            transform=torch_lydorn.torchvision.transforms.functional.batch_normalize,
            key=["image", "image_mean", "image_std"],
            outkey="image")
    ]
    device_transform = torchvision.transforms.Compose(device_transform_list)
    return device_transform
