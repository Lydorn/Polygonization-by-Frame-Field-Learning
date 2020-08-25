import kornia
import torch
import torch_lydorn.torchvision
# from pytorch_memlab import profile, profile_every
from frame_field_learning import measures
import cv2 as cv
import numpy as np


def compute_distance_transform(tensor: torch.Tensor) -> torch.Tensor:
    device = tensor.device
    array = tensor.cpu().numpy()
    shape = array.shape
    array = array.reshape(-1, *shape[-2:]).astype(np.uint8)
    dist_trans = np.empty(array.shape, dtype=np.float32)
    for i in range(array.shape[0]):
        dist_trans[i] = cv.distanceTransform(array[i], distanceType=cv.DIST_L2, maskSize=cv.DIST_MASK_5, dstType=cv.CV_64F)
    dist_trans = dist_trans.reshape(shape)
    dist_trans = torch.tensor(dist_trans, device=device)
    return dist_trans


def select_crossfield(all_outputs, final_seg):
    # Choose frame field from the replicate that best matches the final seg interior
    dice_loss = measures.dice_loss(all_outputs["seg"][:, :, 0, :, :], final_seg[None, :, 0, :, :])
    # Get index of the replicate that achieves the min dice_loss (as it's a loss, lower is better)
    indices_best = torch.argmin(dice_loss, dim=0)
    batch_range = torch.arange(all_outputs["seg"].shape[1])  # batch size
    # For each batch select frame field from the replicate in indices_best
    final_crossfield = all_outputs["crossfield"][indices_best, batch_range]
    return final_crossfield


def aggr_mean(all_outputs):
    final_outputs = {}
    if "seg" in all_outputs:
        final_seg = torch.mean(all_outputs["seg"], dim=0)
        final_outputs["seg"] = final_seg  # Final seg is between min and max: positive pixels are closer to min
        if "crossfield" in all_outputs:
            final_outputs["crossfield"] = select_crossfield(all_outputs, final_seg)
    else:
        raise NotImplementedError("Test Time Augmentation requires segmentation to be computed.")
    return final_outputs


def aggr_median(all_outputs):
    final_outputs = {}
    if "seg" in all_outputs:
        final_seg, _ = torch.median(all_outputs["seg"], dim=0)
        final_outputs["seg"] = final_seg  # Final seg is between min and max: positive pixels are closer to min
        if "crossfield" in all_outputs:
            final_outputs["crossfield"] = select_crossfield(all_outputs, final_seg)
    else:
        raise NotImplementedError("Test Time Augmentation requires segmentation to be computed.")
    return final_outputs


def aggr_dist_trans(all_outputs, seg_threshold):
    final_outputs = {}
    if "seg" in all_outputs:
        min_seg, _ = torch.min(all_outputs["seg"], dim=0)
        max_seg, _ = torch.max(all_outputs["seg"], dim=0)
        # Final seg will be between min and max seg. The idea is that we don't loose the sharp corners (which taking the mean does)
        dist_ext_to_min_seg = compute_distance_transform(min_seg < seg_threshold)
        dist_int_to_max_seg = compute_distance_transform(seg_threshold < max_seg)
        final_seg = dist_ext_to_min_seg < dist_int_to_max_seg
        final_outputs["seg"] = final_seg  # Final seg is between min and max: positive pixels are closer to min
        if "crossfield" in all_outputs:
            final_outputs["crossfield"] = select_crossfield(all_outputs, final_seg)
    else:
        raise NotImplementedError("Test Time Augmentation requires segmentation to be computed.")
    return final_outputs


def aggr_translated(all_outputs, seg_threshold, image_display=None):
    final_outputs = {}
    if "seg" in all_outputs:
        # Cleanup all_seg by multiplying with the mean seg
        all_seg = all_outputs["seg"]
        all_seg_mask: torch.Tensor = seg_threshold < all_seg
        mean_seg = torch.mean(all_seg_mask.float(), dim=0)
        mean_seg_mask = seg_threshold < mean_seg
        all_cleaned_seg = all_seg_mask * mean_seg[None, ...]
        # all_cleaned_seg_mask = seg_threshold < all_cleaned_seg
        # all_cleaned_seg[~all_cleaned_seg_mask] = 0  # Put 0 where seg is below threshold

        # # --- DEBUG SAVE
        # image_seg_display = plot_utils.get_tensorboard_image_seg_display(image_display, mean_seg)
        # image_seg_display = image_seg_display[0].cpu().detach().numpy().transpose(1, 2, 0)
        # skimage.io.imsave(f"image_seg_display_mean_seg.png", image_seg_display)
        # for i, cleaned_seg in enumerate(all_cleaned_seg):
        #     image_seg_display = plot_utils.get_tensorboard_image_seg_display(image_display, cleaned_seg)
        #     image_seg_display = image_seg_display[0].cpu().detach().numpy().transpose(1, 2, 0)
        #     skimage.io.imsave(f"image_seg_display_replicate_cleaned_{i}.png", image_seg_display)
        # # ---

        # Compute barycenter of all cleaned segs
        range_x = torch.arange(all_cleaned_seg.shape[4], device=all_cleaned_seg.device)
        range_y = torch.arange(all_cleaned_seg.shape[3], device=all_cleaned_seg.device)
        grid_y, grid_x = torch.meshgrid([range_x, range_y])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1)

        # Average of coordinates, weighted by segmentation confidence
        spatial_mean_xy = torch.sum(grid_xy[None, None, None, :, :, :] * all_cleaned_seg[:, :, :, :, :, None], dim=(3, 4)) / torch.sum(all_cleaned_seg[:, :, :, :, :, None], dim=(3, 4))
        # Median of all replicate's means
        median_spatial_mean_xy, _ = torch.median(spatial_mean_xy, dim=0)
        # Compute shift between each replicates and the average
        shift_xy = median_spatial_mean_xy[None, :, :, :] - spatial_mean_xy
        shift_xy *= 2  # The shift for the original segs is twice the shift between cleaned segs (assuming homogenous shifts and enough segs)
        shift_xy = shift_xy.view(-1, shift_xy.shape[-1])
        shape = all_outputs["seg"].shape
        shifted_seg = kornia.geometry.translate(all_outputs["seg"].view(-1, *shape[-3:]), shift_xy).view(shape)

        # # --- DEBUG SAVE
        # for i, replicate_shifted_seg in enumerate(shifted_seg):
        #     image_seg_display = plot_utils.get_tensorboard_image_seg_display(image_display, replicate_shifted_seg)
        #     image_seg_display = image_seg_display[0].cpu().detach().numpy().transpose(1, 2, 0)
        #     skimage.io.imsave(f"image_seg_display_replicate_shifted_{i}.png", image_seg_display)
        # # ---

        # Compute mean shifted seg
        mean_shifted_seg = torch.mean(shifted_seg, dim=0)
        # Select replicate seg (and crossfield) that best matches mean_shifted_seg
        dice_loss = measures.dice_loss(shifted_seg[:, :, 0, :, :], mean_shifted_seg[None, :, 0, :, :])
        # Get index of the replicate that achieves the min dice_loss (as it's a loss, lower is better)
        indices_best = torch.argmin(dice_loss, dim=0)
        batch_range = torch.arange(all_outputs["seg"].shape[1])  # batch size
        # For each batch select seg and frame field from the replicate in indices_best
        final_outputs["seg"] = shifted_seg[indices_best, batch_range]
        if "crossfield" in all_outputs:
            final_outputs["crossfield"] = all_outputs["crossfield"][indices_best, batch_range]

        # if "crossfield" in all_outputs:
        #     final_outputs["crossfield"] = select_crossfield(all_outputs, final_seg)
    else:
        raise NotImplementedError("Test Time Augmentation requires segmentation to be computed.")
    return final_outputs


def tta_inference(model, xb, seg_threshold):
    # Perform inference several times with transformed input image and aggregate results
    replicates = 4 * 2  # 4 rotations, each with vflip/no vflip

    # Init results tensors
    notrans_outputs = model.inference(xb["image"])
    output_keys = notrans_outputs.keys()
    all_outputs = {}
    for key in output_keys:
        all_outputs[key] = torch.empty((replicates, *notrans_outputs[key].shape), dtype=notrans_outputs[key].dtype,
                                       device=notrans_outputs[key].device)
        all_outputs[key][0] = notrans_outputs[key]
    # Flip image
    flipped_image = kornia.geometry.transform.vflip(xb["image"])
    flipped_outputs = model.inference(flipped_image)
    for key in output_keys:
        reversed_output = kornia.geometry.transform.vflip(flipped_outputs[key])
        all_outputs[key][1] = reversed_output

    # --- Apply transforms one by one and add results to all_outputs
    for k in range(1, 4):
        rotated_image = torch.rot90(xb["image"], k=k, dims=(-2, -1))
        rotated_outputs = model.inference(rotated_image)
        for key in output_keys:
            reversed_output = torch.rot90(rotated_outputs[key], k=-k, dims=(-2, -1))
            if key == "crossfield":
                angle = -k * 90
                # TODO: use a faster implementation of rotate_framefield that only handles angles [0, 90, 180, 270]
                reversed_output = torch_lydorn.torchvision.transforms.functional.rotate_framefield(reversed_output,
                                                                                                   angle)
            all_outputs[key][2 * k] = reversed_output

        # Flip rotated image
        flipped_rotated_image = kornia.geometry.transform.vflip(rotated_image)
        flipped_rotated_outputs = model.inference(flipped_rotated_image)
        for key in output_keys:
            reversed_output = torch.rot90(kornia.geometry.transform.vflip(flipped_rotated_outputs[key]), k=-k,
                                          dims=(-2, -1))
            if key == "crossfield":
                angle = -k * 90
                reversed_output = torch_lydorn.torchvision.transforms.functional.vflip_framefield(reversed_output)
                reversed_output = torch_lydorn.torchvision.transforms.functional.rotate_framefield(reversed_output,
                                                                                                   angle)
            all_outputs[key][2 * k + 1] = reversed_output

    # --- DEBUG
    # all_outputs["seg"] *= 0
    # for i in range(all_outputs["seg"].shape[0]):
    #     center = 512
    #     size = 100
    #     shift_x = random.randint(-20, 20)
    #     shift_y = random.randint(-20, 20)
    #     all_outputs["seg"][i][..., center + shift_y - size:center + shift_y + size, center + shift_x - size:center + shift_x + size] = 1
    #     # Add noise
    #     noise_center_x = random.randint(100, 1024-100)
    #     noise_center_y = random.randint(100, 1024-100)
    #     noise_size = 10
    #     all_outputs["seg"][i][..., noise_center_y - noise_size:noise_center_y + noise_size, noise_center_x - noise_size:noise_center_x + noise_size] = 1
    #     # Add more noise
    #     all_outputs["seg"][i] += 0.25 * torch.rand(all_outputs["seg"][i].shape, device=all_outputs["seg"][i].device)
    #     all_outputs["seg"][i] = torch.clamp(all_outputs["seg"][i], 0, 1)


    # # --- DEBUG SAVE
    # image_display = torch_lydorn.torchvision.transforms.functional.batch_denormalize(xb["image"],
    #                                                                                  xb["image_mean"],
    #                                                                                  xb["image_std"])
    # for i, replicate_seg in enumerate(all_outputs["seg"]):
    #     image_seg_display = plot_utils.get_tensorboard_image_seg_display(image_display, replicate_seg)
    #     image_seg_display = image_seg_display[0].cpu().detach().numpy().transpose(1, 2, 0)
    #     skimage.io.imsave(f"image_seg_display_replicate_{i}.png", image_seg_display)
    # # ---


    # --- Aggregate results
    # final_outputs = aggr_dist_trans(all_outputs, seg_threshold)
    # final_outputs = aggr_translated(all_outputs, seg_threshold, image_display=image_display)
    # final_outputs = aggr_translated(all_outputs, seg_threshold)
    final_outputs = aggr_mean(all_outputs)
    # final_outputs = aggr_median(all_outputs)

    # # --- DEBUG SAVE
    # image_seg_display = plot_utils.get_tensorboard_image_seg_display(image_display, final_outputs["seg"])
    # image_seg_display = image_seg_display[0].cpu().detach().numpy().transpose(1, 2, 0)
    # skimage.io.imsave("image_seg_display_final.png", image_seg_display)
    # # ---

    # input("Press <Enter>...")

    return final_outputs