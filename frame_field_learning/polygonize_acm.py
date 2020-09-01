import argparse
import fnmatch
import time

import numpy as np
import skimage
import skimage.measure
import skimage.io
from tqdm import tqdm
import shapely.geometry
import shapely.ops
import shapely.prepared
import cv2

from functools import partial

import torch

from frame_field_learning import polygonize_utils
from frame_field_learning import frame_field_utils

from torch_lydorn.torch.nn.functionnal import bilinear_interpolate
from torch_lydorn.torchvision.transforms import polygons_to_tensorpoly, tensorpoly_pad

from lydorn_utils import math_utils
from lydorn_utils import python_utils
from lydorn_utils import print_utils


DEBUG = False


def debug_print(s: str):
    if DEBUG:
        print_utils.print_debug(s)


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--raw_pred',
        nargs='*',
        type=str,
        help='Filepath to the raw pred file(s)')
    argparser.add_argument(
        '--im_filepath',
        type=str,
        help='Filepath to input image. Will retrieve seg and crossfield in the same directory')
    argparser.add_argument(
        '--dirpath',
        type=str,
        help='Path to directory containing seg and crossfield files. Will perform polygonization on all.')
    argparser.add_argument(
        '--bbox',
        nargs='*',
        type=int,
        help='Selects area in bbox for computation: [min_row, min_col, max_row, max_col]')
    argparser.add_argument(
        '--steps',
        type=int,
        help='Optim steps')

    args = argparser.parse_args()
    return args


class PolygonAlignLoss:
    def __init__(self, indicator, level, c0c2, data_coef, length_coef, crossfield_coef, dist=None, dist_coef=None):
        self.indicator = indicator
        self.level = level
        self.c0c2 = c0c2
        self.dist = dist

        self.data_coef = data_coef
        self.length_coef = length_coef
        self.crossfield_coef = crossfield_coef
        self.dist_coef = dist_coef

    def __call__(self, tensorpoly):
        """

        :param tensorpoly: closed polygon
        :return:
        """
        polygon = tensorpoly.pos[tensorpoly.to_padded_index]
        polygon_batch = tensorpoly.batch[tensorpoly.to_padded_index]

        # Compute edges:
        edges = polygon[1:] - polygon[:-1]
        # Compute edge mask to remove edges that connect two different polygons from loss
        # Also note the last poly_slice is not used, because the last edge of the last polygon is not connected to a non-existant next polygon:
        edge_mask = torch.ones((edges.shape[0]), device=edges.device)
        edge_mask[tensorpoly.to_unpadded_poly_slice[:-1, 1]] = 0

        midpoints = (polygon[1:] + polygon[:-1]) / 2
        midpoints_batch = polygon_batch[1:]

        midpoints_int = midpoints.round().long()
        midpoints_int[:, 0] = torch.clamp(midpoints_int[:, 0], 0, self.c0c2.shape[2] - 1)
        midpoints_int[:, 1] = torch.clamp(midpoints_int[:, 1], 0, self.c0c2.shape[3] - 1)
        midpoints_c0 = self.c0c2[midpoints_batch, :2, midpoints_int[:, 0], midpoints_int[:, 1]]
        midpoints_c2 = self.c0c2[midpoints_batch, 2:, midpoints_int[:, 0], midpoints_int[:, 1]]

        norms = torch.norm(edges, dim=-1)
        # Add edges with small norms to the edge mask so that losses are not computed on them
        edge_mask[norms < 0.1] = 0  # Less than 10% of a pixel
        z = edges / (norms[:, None] + 1e-3)

        # Align to crossfield
        align_loss = frame_field_utils.framefield_align_error(midpoints_c0, midpoints_c2, z, complex_dim=1)
        align_loss = align_loss * edge_mask
        total_align_loss = torch.sum(align_loss)

        # Align to level set of indicator:
        pos_indicator_value = bilinear_interpolate(self.indicator[:, None, ...], tensorpoly.pos, batch=tensorpoly.batch)
        # TODO: Try to use grid_sample with batch for speed: put batch dim to height dim and make a single big image.
        # TODO: Convert pos accordingly and take care of borders
        # height = self.indicator.shape[1]
        # width = self.indicator.shape[2]
        # normed_xy = tensorpoly.pos.roll(shifts=1, dims=-1)
        # normed_xy[: 0] /= (width-1)
        # normed_xy[: 1] /= (height-1)
        # centered_xy = 2*normed_xy - 1
        # pos_value = torch.nn.functional.grid_sample(self.indicator[None, None, ...], centered_batch_xy[None, None, ...], align_corners=True).squeeze()
        level_loss = torch.sum(torch.pow(pos_indicator_value - self.level, 2))

        # Align to minimum distance from the boundary
        dist_loss = None
        if self.dist is not None:
            pos_dist_value = bilinear_interpolate(self.dist[:, None, ...], tensorpoly.pos, batch=tensorpoly.batch)
            dist_loss = torch.sum(torch.pow(pos_dist_value, 2))

        length_penalty = torch.sum(
            torch.pow(norms * edge_mask, 2))  # Sum of squared norm to penalise uneven edge lengths
        # length_penalty = torch.sum(norms)

        losses_dict = {
            "align": total_align_loss.item(),
            "level": level_loss.item(),
            "length": length_penalty.item(),
        }
        coef_sum = self.data_coef + self.length_coef + self.crossfield_coef
        total_loss = (self.data_coef * level_loss + self.length_coef * length_penalty + self.crossfield_coef * total_align_loss)
        if dist_loss is not None:
            losses_dict["dist"] = dist_loss.item()
            total_loss += self.dist_coef * dist_loss
            coef_sum += self.dist_coef
        total_loss /= coef_sum
        return total_loss, losses_dict


class TensorPolyOptimizer:
    def __init__(self, config, tensorpoly, indicator, c0c2, data_coef, length_coef, crossfield_coef, dist=None, dist_coef=None):
        assert len(indicator.shape) == 3, "indicator: (N, H, W)"
        assert len(c0c2.shape) == 4 and c0c2.shape[1] == 4, "c0c2: (N, 4, H, W)"
        if dist is not None:
            assert len(dist.shape) == 3, "dist: (N, H, W)"


        self.config = config
        self.tensorpoly = tensorpoly

        # Require grads for graph.pos: this is what is optimized
        self.tensorpoly.pos.requires_grad = True

        # Save pos of endpoints so that they can be reset after each step (endpoints are not meant to be moved)
        self.endpoint_pos = self.tensorpoly.pos[self.tensorpoly.is_endpoint].clone()

        self.criterion = PolygonAlignLoss(indicator, config["data_level"], c0c2, data_coef, length_coef,
                                          crossfield_coef, dist=dist, dist_coef=dist_coef)
        self.optimizer = torch.optim.SGD([tensorpoly.pos], lr=config["poly_lr"])

        def lr_warmup_func(iter):
            if iter < config["warmup_iters"]:
                coef = 1 + (config["warmup_factor"] - 1) * (config["warmup_iters"] - iter) / config["warmup_iters"]
            else:
                coef = 1
            return coef

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_warmup_func)

    def step(self, iter_num):
        self.optimizer.zero_grad()
        loss, losses_dict = self.criterion(self.tensorpoly)
        # print("loss:", loss.item())
        loss.backward()
        # print(polygon_tensor.grad[0])
        self.optimizer.step()
        self.lr_scheduler.step(iter_num)

        # Move endpoints back:
        with torch.no_grad():
            self.tensorpoly.pos[self.tensorpoly.is_endpoint] = self.endpoint_pos
        return loss.item(), losses_dict

    def optimize(self):
        # if DEBUG:
        #     optim_iter = tqdm(range(self.config["steps"]), desc="Gradient descent", leave=True)
        # else:
        #     optim_iter = range(self.config["steps"])
        # # print("---------------------------------------------")
        # for iter_num in optim_iter:
        #     loss, losses_dict = self.step(iter_num)
        #     if DEBUG:
        #         optim_iter.set_postfix(loss=loss, **losses_dict)
        optim_iter = range(self.config["steps"])
        for iter_num in optim_iter:
            loss, losses_dict = self.step(iter_num)
        return self.tensorpoly


def contours_batch_to_tensorpoly(contours_batch):
    # Convert a batch of contours to a TensorPoly representation with PyTorch tensors
    tensorpoly = polygons_to_tensorpoly(contours_batch)
    # Pad contours so that we can treat them as closed:
    tensorpoly = tensorpoly_pad(tensorpoly, padding=(0, 1))
    return tensorpoly


def tensorpoly_to_contours_batch(tensorpoly):
    # Convert back to contours
    contours_batch = [[] for _ in range(tensorpoly.batch_size)]
    for poly_i in range(tensorpoly.poly_slice.shape[0]):
        s = tensorpoly.poly_slice[poly_i, :]
        contour = np.array(tensorpoly.pos[s[0]:s[1], :].detach().cpu())
        is_open = tensorpoly.is_endpoint[s[0]]  # Is open = if first vertex is an endpoint
        if not is_open:
            # Close contour
            contour = np.concatenate([contour, contour[:1, :]], axis=0)
        batch_i = tensorpoly.batch[s[0]]  # Batch of polygon = batch of first vertex
        contours_batch[batch_i].append(contour)
    return contours_batch


def print_contours_stats(contours):
    min_length = contours[0].shape[0]
    max_length = contours[0].shape[0]
    nb_vertices = 0
    for contour in contours:
        nb_vertices += contour.shape[0]
        if contour.shape[0] < min_length:
            min_length = contour.shape[0]
        if max_length < contour.shape[0]:
            max_length = contour.shape[0]
    print("Nb polygon:", len(contours), "Nb vertices:", nb_vertices, "Min lengh:", min_length, "Max lengh:", max_length)


def shapely_postprocess(contours, u, v, np_indicator, tolerance, config):
    if type(tolerance) == list:
        # Use several tolerance values for simplification. return a dict with all results
        out_polygons_dict = {}
        out_probs_dict = {}
        for tol in tolerance:
            out_polygons, out_probs = shapely_postprocess(contours, u, v, np_indicator, tol, config)
            out_polygons_dict["tol_{}".format(tol)] = out_polygons
            out_probs_dict["tol_{}".format(tol)] = out_probs
        return out_polygons_dict, out_probs_dict
    else:
        height = np_indicator.shape[0]
        width = np_indicator.shape[1]

        # debug_print("Corner-aware simplification")
        # Simplify contours a little to avoid some close-together corner-detection:
        # TODO: handle close-together corners better
        contours = [skimage.measure.approximate_polygon(contour, tolerance=min(1, tolerance)) for contour in contours]
        corner_masks = frame_field_utils.detect_corners(contours, u, v)
        contours = polygonize_utils.split_polylines_corner(contours, corner_masks)

        # Convert to Shapely:
        line_string_list = [shapely.geometry.LineString(out_contour[:, ::-1]) for out_contour in contours]

        line_string_list = [line_string.simplify(tolerance, preserve_topology=True) for line_string in line_string_list]

        # Add image boundary line_strings for border polygons
        line_string_list.append(
            shapely.geometry.LinearRing([
                (0, 0),
                (0, height - 1),
                (width - 1, height - 1),
                (width - 1, 0),
            ]))

        # debug_print("Merge polylines")

        # Merge polylines (for border polygons):
        multi_line_string = shapely.ops.unary_union(line_string_list)

        # debug_print("polygonize_full")

        # Find polygons:
        polygons, dangles, cuts, invalids = shapely.ops.polygonize_full(multi_line_string)
        polygons = list(polygons)

        # debug_print("Remove small polygons")

        # Remove small polygons
        polygons = [polygon for polygon in polygons if
                    config["min_area"] < polygon.area]

        # debug_print("Remove low prob polygons")

        # Remove low prob polygons
        filtered_polygons = []
        filtered_polygon_probs = []
        for polygon in polygons:
            prob = polygonize_utils.compute_geom_prob(polygon, np_indicator)
            # print("acm:", np_indicator.min(), np_indicator.mean(), np_indicator.max(), prob)
            if config["seg_threshold"] < prob:
                filtered_polygons.append(polygon)
                filtered_polygon_probs.append(prob)

        return filtered_polygons, filtered_polygon_probs


def post_process(contours, np_seg, np_crossfield, config):
    u, v = math_utils.compute_crossfield_uv(np_crossfield)  # u, v are complex arrays

    np_indicator = np_seg[:, :, 0]
    polygons, probs = shapely_postprocess(contours, u, v, np_indicator, config["tolerance"], config)

    return polygons, probs


def polygonize(seg_batch, crossfield_batch, config, pool=None, pre_computed=None):
    tic_start = time.time()

    assert len(seg_batch.shape) == 4 and seg_batch.shape[
        1] <= 3, "seg_batch should be (N, C, H, W) with C <= 3, not {}".format(seg_batch.shape)
    assert len(crossfield_batch.shape) == 4 and crossfield_batch.shape[
        1] == 4, "crossfield_batch should be (N, 4, H, W)"
    assert seg_batch.shape[0] == crossfield_batch.shape[0], "Batch size for seg and crossfield should match"


    # Indicator
    # tic = time.time()
    indicator_batch = seg_batch[:, 0, :, :]
    np_indicator_batch = indicator_batch.cpu().numpy()
    indicator_batch = indicator_batch.to(config["device"])
    # toc = time.time()
    # debug_print(f"Indicator to cpu: {toc - tic}s")

    # Distance image
    dist_batch = None
    if "dist_coef" in config:
        # tic = time.time()
        np_dist_batch = np.empty(np_indicator_batch.shape)
        for batch_i in range(np_indicator_batch.shape[0]):
            dist_1 = cv2.distanceTransform(np_indicator_batch[batch_i].astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_5, dstType=cv2.CV_64F)
            dist_2 = cv2.distanceTransform(1 - np_indicator_batch[batch_i].astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_5, dstType=cv2.CV_64F)
            np_dist_batch[0] = dist_1 + dist_2 - 1
        dist_batch = torch.from_numpy(np_dist_batch)
        dist_batch = dist_batch.to(config["device"])
        # skimage.io.imsave("dist.png", np_dist_batch[0])
        # toc = time.time()
        # debug_print(f"Distance image: {toc - tic}s")

    # debug_print("Init contours")
    if pre_computed is None or "init_contours_batch" not in pre_computed:
        # tic = time.time()
        init_contours_batch = polygonize_utils.compute_init_contours_batch(np_indicator_batch, config["data_level"], pool=pool)
        # toc = time.time()
        # debug_print(f"Init contours: {toc - tic}s")
    else:
        init_contours_batch = pre_computed["init_contours_batch"]

    # debug_print("Convert contours to tensorpoly")
    tensorpoly = contours_batch_to_tensorpoly(init_contours_batch)

    # debug_print("Optimize")

    # --- Optimize
    # tic = time.time()

    tensorpoly.to(config["device"])
    crossfield_batch = crossfield_batch.to(config["device"])
    dist_coef = config["dist_coef"] if "dist_coef" in config else None
    tensorpoly_optimizer = TensorPolyOptimizer(config, tensorpoly, indicator_batch, crossfield_batch,
                                               config["data_coef"],
                                               config["length_coef"], config["crossfield_coef"], dist=dist_batch, dist_coef=dist_coef)
    tensorpoly = tensorpoly_optimizer.optimize()

    out_contours_batch = tensorpoly_to_contours_batch(tensorpoly)

    # toc = time.time()
    # debug_print(f"Optimize contours: {toc - tic}s")

    # --- Post-process:
    # debug_print("Post-process")
    # tic = time.time()

    np_seg_batch = np.transpose(seg_batch.cpu().numpy(), (0, 2, 3, 1))
    np_crossfield_batch = np.transpose(crossfield_batch.cpu().numpy(), (0, 2, 3, 1))
    if pool is not None:
        post_process_partial = partial(post_process, config=config)
        polygons_probs_batch = pool.starmap(post_process_partial, zip(out_contours_batch, np_seg_batch, np_crossfield_batch))
        polygons_batch, probs_batch = zip(*polygons_probs_batch)
    else:
        polygons_batch = []
        probs_batch = []
        for i, out_contours in enumerate(out_contours_batch):
            polygons, probs = post_process(out_contours, np_seg_batch[i], np_crossfield_batch[i], config)
            polygons_batch.append(polygons)
            probs_batch.append(probs)

    # toc = time.time()
    # debug_print(f"Shapely post-process: {toc - tic}s")

    # toc = time.time()
    # print(f"Post-process: {toc - tic}s")
    # ---

    toc_end = time.time()
    # debug_print(f"Total: {toc_end - tic_start}s")

    return polygons_batch, probs_batch


def main():
    from frame_field_learning import framefield, inference
    import os

    def save_gt_poly(raw_pred_filepath, name):
        filapth_format = "/data/mapping_challenge_dataset/processed/val/data_{}.pt"
        sample = torch.load(filapth_format.format(name))
        polygon_arrays = sample["gt_polygons"]
        polygons = [shapely.geometry.Polygon(polygon[:, ::-1]) for polygon in polygon_arrays]
        base_filepath = os.path.join(os.path.dirname(raw_pred_filepath), name)
        filepath = base_filepath + "." + name + ".pdf"
        plot_utils.save_poly_viz(image, polygons, filepath)

    config = {
        "indicator_add_edge": False,
        "steps": 500,
        "data_level": 0.5,
        "data_coef": 0.1,
        "length_coef": 0.4,
        "crossfield_coef": 0.5,
        "poly_lr": 0.01,
        "warmup_iters": 100,
        "warmup_factor": 0.1,
        "device": "cuda",
        "tolerance": 0.5,
        "seg_threshold": 0.5,
        "min_area": 1,

        "inner_polylines_params": {
            "enable": False,
            "max_traces": 1000,
            "seed_threshold": 0.5,
            "low_threshold": 0.1,
            "min_width": 2,  # Minimum width of trace to take into account
            "max_width": 8,
            "step_size": 1,
        }
    }
    # --- Process args --- #
    args = get_args()
    if args.steps is not None:
        config["steps"] = args.steps

    if args.raw_pred is not None:
        # Load raw_pred(s)
        image_list = []
        name_list = []
        seg_list = []
        crossfield_list = []
        for raw_pred_filepath in args.raw_pred:
            raw_pred = torch.load(raw_pred_filepath)
            image_list.append(raw_pred["image"])
            name_list.append(raw_pred["name"])
            seg_list.append(raw_pred["seg"])
            crossfield_list.append(raw_pred["crossfield"])
        seg_batch = torch.stack(seg_list, dim=0)
        crossfield_batch = torch.stack(crossfield_list, dim=0)

        out_contours_batch, out_probs_batch = polygonize(seg_batch, crossfield_batch, config)

        for i, raw_pred_filepath in enumerate(args.raw_pred):
            image = image_list[i]
            name = name_list[i]
            polygons = out_contours_batch[i]
            base_filepath = os.path.join(os.path.dirname(raw_pred_filepath), name)
            filepath = base_filepath + ".poly_acm.pdf"
            plot_utils.save_poly_viz(image, polygons, filepath)

            # Load gt polygons
            save_gt_poly(raw_pred_filepath, name)
    elif args.im_filepath:
        # Load from filepath, look for seg and crossfield next to the image
        # Load data
        image = skimage.io.imread(args.im_filepath)
        base_filepath = os.path.splitext(args.im_filepath)[0]
        seg = skimage.io.imread(base_filepath + ".seg.tif") / 255
        crossfield = np.load(base_filepath + ".crossfield.npy", allow_pickle=True)

        # Select bbox for dev
        if args.bbox is not None:
            assert len(args.bbox) == 4, "bbox should have 4 values"
            bbox = args.bbox
            # bbox = [1440, 210, 1800, 650]  # vienna12
            # bbox = [2808, 2393, 3124, 2772]  # innsbruck19
            image = image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            seg = seg[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            crossfield = crossfield[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            extra_name = ".bbox_{}_{}_{}_{}".format(*bbox)
        else:
            extra_name = ""

        # Convert to torch and add batch dim
        seg_batch = torch.tensor(np.transpose(seg[:, :, :2], (2, 0, 1)), dtype=torch.float)[None, ...]
        crossfield_batch = torch.tensor(np.transpose(crossfield, (2, 0, 1)), dtype=torch.float)[None, ...]

        out_contours_batch, out_probs_batch = polygonize(seg_batch, crossfield_batch, config)

        polygons = out_contours_batch[0]
        # Save shapefile
        # save_utils.save_shapefile(polygons, base_filepath + extra_name, "poly_acm", args.im_filepath)

        # Save pdf viz
        filepath = base_filepath + extra_name + ".poly_acm.pdf"
        plot_utils.save_poly_viz(image, polygons, filepath, linewidths=1, draw_vertices=True, color_choices=[[0, 1, 0, 1]])
    elif args.dirpath:
        seg_filename_list = fnmatch.filter(os.listdir(args.dirpath), "*.seg.tif")
        sorted(seg_filename_list)
        pbar = tqdm(seg_filename_list, desc="Poly files")
        for id, seg_filename in enumerate(pbar):
            basename = seg_filename[:-len(".seg.tif")]
            # shp_filepath = os.path.join(args.dirpath, basename + ".poly_acm.shp")
            # Verify if image has already been polygonized
            # if os.path.exists(shp_filepath):
            #     continue

            pbar.set_postfix(name=basename, status="Loading data...")
            crossfield_filename = basename + ".crossfield.npy"
            metadata_filename = basename + ".metadata.json"
            seg = skimage.io.imread(os.path.join(args.dirpath, seg_filename)) / 255
            crossfield = np.load(os.path.join(args.dirpath, crossfield_filename), allow_pickle=True)
            metadata = python_utils.load_json(os.path.join(args.dirpath, metadata_filename))
            # image_filepath = metadata["image_filepath"]
            # as_shp_filename = os.path.splitext(os.path.basename(image_filepath))[0]
            # as_shp_filepath = os.path.join(os.path.dirname(os.path.dirname(image_filepath)), "gt_polygons", as_shp_filename + ".shp")

            # Convert to torch and add batch dim
            seg_batch = torch.tensor(np.transpose(seg[:, :, :2], (2, 0, 1)), dtype=torch.float)[None, ...]
            crossfield_batch = torch.tensor(np.transpose(crossfield, (2, 0, 1)), dtype=torch.float)[None, ...]

            pbar.set_postfix(name=basename, status="Polygonazing...")
            out_contours_batch, out_probs_batch = polygonize(seg_batch, crossfield_batch, config)

            polygons = out_contours_batch[0]

            # Save as shp
            # pbar.set_postfix(name=basename, status="Saving .shp...")
            # geo_utils.save_shapefile_from_shapely_polygons(polygons, shp_filepath, as_shp_filepath)

            # Save as COCO annotation
            base_filepath = os.path.join(args.dirpath, basename)
            inference.save_poly_coco(polygons, id, base_filepath, "annotation.poly")
    else:
        print("Showcase on a very simple example:")
        seg = np.zeros((6, 8, 3))
        # Triangle:
        seg[1, 4] = 1
        seg[2, 3:5] = 1
        seg[3, 2:5] = 1
        seg[4, 1:5] = 1
        # L extension:
        seg[3:5, 5:7] = 1

        u = np.zeros((6, 8), dtype=np.complex)
        v = np.zeros((6, 8), dtype=np.complex)
        # Init with grid
        u.real = 1
        v.imag = 1
        # Add slope
        u[:4, :4] *= np.exp(1j * np.pi/4)
        v[:4, :4] *= np.exp(1j * np.pi/4)
        # Add slope corners
        # u[:2, 4:6] *= np.exp(1j * np.pi / 4)
        # v[4:, :2] *= np.exp(- 1j * np.pi / 4)

        crossfield = math_utils.compute_crossfield_c0c2(u, v)

        seg_batch = torch.tensor(np.transpose(seg[:, :, :2], (2, 0, 1)), dtype=torch.float)[None, ...]
        crossfield_batch = torch.tensor(np.transpose(crossfield, (2, 0, 1)), dtype=torch.float)[None, ...]

        out_contours_batch, out_probs_batch = polygonize(seg_batch, crossfield_batch, config)

        polygons = out_contours_batch[0]

        filepath = "demo_poly_acm.pdf"
        plot_utils.save_poly_viz(seg, polygons, filepath, linewidths=0.5, draw_vertices=True, crossfield=crossfield)


if __name__ == '__main__':
    main()
