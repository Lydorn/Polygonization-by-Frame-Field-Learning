import argparse
import fnmatch
import functools
import time
from typing import List

import numpy as np
import skan
import skimage
import skimage.measure
import skimage.morphology
import skimage.io
from tqdm import tqdm
import shapely.geometry
import shapely.ops
import shapely.prepared
import scipy.interpolate

from functools import partial

import torch
import torch_scatter

from frame_field_learning import polygonize_utils, plot_utils, frame_field_utils

from torch_lydorn.torch.nn.functionnal import bilinear_interpolate
from torch_lydorn.torchvision.transforms import Paths, Skeleton, TensorSkeleton, skeletons_to_tensorskeleton, tensorskeleton_to_skeletons
import torch_lydorn.kornia

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
        '--seg_filepath',
        type=str,
        help='Filepath to input segmentation image.')
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


def get_junction_corner_index(tensorskeleton):
    """
    Returns as a tensor the list of 3-tuples each representing a corner of a junction.
    The 3-tuple contains the indices of the 3 vertices making up the corner.

    In the text below, we use the following notation:
        - J: the number of junction nodes
        - Sd: the sum of the degrees of all the junction nodes
        - T: number of tip nodes
    @return: junction_corner_index of shape (Sd*J - T, 3) which is a list of 3-tuples (for each junction corner)
    """
    # --- Compute all junction edges:
    junction_edge_index = torch.empty((2 * tensorskeleton.num_paths, 2), dtype=torch.long, device=tensorskeleton.path_index.device)
    junction_edge_index[:tensorskeleton.num_paths, 0] = tensorskeleton.path_index[tensorskeleton.path_delim[:-1]]
    junction_edge_index[:tensorskeleton.num_paths, 1] = tensorskeleton.path_index[tensorskeleton.path_delim[:-1] + 1]
    junction_edge_index[tensorskeleton.num_paths:, 0] = tensorskeleton.path_index[tensorskeleton.path_delim[1:] - 1]
    junction_edge_index[tensorskeleton.num_paths:, 1] = tensorskeleton.path_index[tensorskeleton.path_delim[1:] - 2]
    # --- Remove tip junctions
    degrees = tensorskeleton.degrees[junction_edge_index[:, 0]]
    junction_edge_index = junction_edge_index[1 < degrees, :]
    # --- Group by junction by sorting
    group_indices = torch.argsort(junction_edge_index[:, 0], dim=0)
    grouped_junction_edge_index = junction_edge_index[group_indices, :]
    # --- Compute angle to vertical axis of each junction edge
    junction_edge = tensorskeleton.pos.detach()[grouped_junction_edge_index, :]
    junction_tangent = junction_edge[:, 1, :] - junction_edge[:, 0, :]
    junction_angle_to_axis = torch.atan2(junction_tangent[:, 1], junction_tangent[:, 0])
    # --- Sort by angle for each junction separately and build junction_corner_index
    unique = torch.unique_consecutive(grouped_junction_edge_index[:, 0])
    count = tensorskeleton.degrees[unique]
    junction_end_index = torch.cumsum(count, dim=0)
    slice_start = 0
    junction_corner_index = torch.empty((grouped_junction_edge_index.shape[0], 3), dtype=torch.long, device=tensorskeleton.path_index.device)
    for slice_end in junction_end_index:
        slice_angle_to_axis = junction_angle_to_axis[slice_start:slice_end]
        slice_junction_edge_index = grouped_junction_edge_index[slice_start:slice_end]
        sort_indices = torch.argsort(slice_angle_to_axis, dim=0)
        slice_junction_edge_index = slice_junction_edge_index[sort_indices]
        junction_corner_index[slice_start:slice_end, 0] = slice_junction_edge_index[:, 1]
        junction_corner_index[slice_start:slice_end, 1] = slice_junction_edge_index[:, 0]
        junction_corner_index[slice_start:slice_end, 2] = slice_junction_edge_index[:, 1].roll(-1, dims=0)
        slice_start = slice_end
    return junction_corner_index


class AlignLoss:
    def __init__(self, tensorskeleton: TensorSkeleton, indicator: torch.Tensor, level: float, c0c2: torch.Tensor, loss_params):
        """
        :param tensorskeleton: skeleton graph in tensor format
        :return:
        """
        self.tensorskeleton = tensorskeleton
        self.indicator = indicator
        self.level = level
        self.c0c2 = c0c2
        # self.uv = frame_field_utils.c0c2_to_uv(c0c2)

        # Prepare junction_corner_index:

        # TODO: junction_corner_index: list
        self.junction_corner_index = get_junction_corner_index(tensorskeleton)

        # Loss coefs
        self.data_coef_interp = scipy.interpolate.interp1d(loss_params["coefs"]["step_thresholds"],
                                                           loss_params["coefs"]["data"])
        self.length_coef_interp = scipy.interpolate.interp1d(loss_params["coefs"]["step_thresholds"],
                                                             loss_params["coefs"]["length"])
        self.crossfield_coef_interp = scipy.interpolate.interp1d(loss_params["coefs"]["step_thresholds"],
                                                                 loss_params["coefs"]["crossfield"])
        self.curvature_coef_interp = scipy.interpolate.interp1d(loss_params["coefs"]["step_thresholds"],
                                                                loss_params["coefs"]["curvature"])
        self.corner_coef_interp = scipy.interpolate.interp1d(loss_params["coefs"]["step_thresholds"],
                                                             loss_params["coefs"]["corner"])
        self.junction_coef_interp = scipy.interpolate.interp1d(loss_params["coefs"]["step_thresholds"],
                                                             loss_params["coefs"]["junction"])

        self.curvature_dissimilarity_threshold = loss_params["curvature_dissimilarity_threshold"]
        self.corner_angles = np.pi * torch.tensor(loss_params["corner_angles"]) / 180  # Convert to radians
        self.corner_angle_threshold = np.pi * loss_params["corner_angle_threshold"] / 180  # Convert to radians
        self.junction_angles = np.pi * torch.tensor(loss_params["junction_angles"]) / 180  # Convert to radians
        self.junction_angle_weights = torch.tensor(loss_params["junction_angle_weights"])
        self.junction_angle_threshold = np.pi * loss_params["junction_angle_threshold"] / 180  # Convert to radians

        # Pre-compute useful pointers
        # edge_index_start = tensorskeleton.path_index[:-1]
        # edge_index_end = tensorskeleton.path_index[1:]
        #
        # self.tensorskeleton.edge_index = edge_index

    def __call__(self, pos: torch.Tensor, iter_num: int):
        # --- Align to frame field loss
        path_pos = pos[self.tensorskeleton.path_index]
        detached_path_pos = path_pos.detach()
        path_batch = self.tensorskeleton.batch[self.tensorskeleton.path_index]
        tangents = path_pos[1:] - path_pos[:-1]
        # Compute edge mask to remove edges that connect two different paths from loss
        edge_mask = torch.ones((tangents.shape[0]), device=tangents.device)
        edge_mask[self.tensorskeleton.path_delim[1:-1] - 1] = 0  # Zero out edges between paths

        midpoints = (path_pos[1:] + path_pos[:-1]) / 2
        midpoints_batch = self.tensorskeleton.batch[self.tensorskeleton.path_index[:-1]]  # Same as start point of edge

        midpoints_int = midpoints.round().long()
        midpoints_int[:, 0] = torch.clamp(midpoints_int[:, 0], 0, self.c0c2.shape[2] - 1)
        midpoints_int[:, 1] = torch.clamp(midpoints_int[:, 1], 0, self.c0c2.shape[3] - 1)
        midpoints_c0 = self.c0c2[midpoints_batch, :2, midpoints_int[:, 0], midpoints_int[:, 1]]
        midpoints_c2 = self.c0c2[midpoints_batch, 2:, midpoints_int[:, 0], midpoints_int[:, 1]]

        norms = torch.norm(tangents, dim=-1)
        edge_mask[norms < 0.1] = 0  # Zero out very small edges
        normed_tangents = tangents / (norms[:, None] + 1e-6)

        align_loss = frame_field_utils.framefield_align_error(midpoints_c0, midpoints_c2, normed_tangents, complex_dim=1)
        align_loss = align_loss * edge_mask
        total_align_loss = torch.sum(align_loss)

        # --- Align to level set of indicator:
        pos_value = bilinear_interpolate(self.indicator[:, None, ...], pos, batch=self.tensorskeleton.batch)
        # TODO: use grid_sample with batch: put batch dim to height dim and make a single big image.
        # TODO: Convert pos accordingly and take care of borders
        # height = self.indicator.shape[1]
        # width = self.indicator.shape[2]
        # normed_xy = tensorskeleton.pos.roll(shifts=1, dims=-1)
        # normed_xy[: 0] /= (width-1)
        # normed_xy[: 1] /= (height-1)
        # centered_xy = 2*normed_xy - 1
        # pos_value = torch.nn.functional.grid_sample(self.indicator[None, None, ...],
        #                                             centered_batch_xy[None, None, ...], align_corners=True).squeeze()
        level_loss = torch.sum(torch.pow(pos_value - self.level, 2))

        # --- Prepare useful tensors for curvature loss:
        prev_pos = detached_path_pos[:-2]
        middle_pos = path_pos[1:-1]
        next_pos = detached_path_pos[2:]
        prev_tangent = middle_pos - prev_pos
        next_tangent = next_pos - middle_pos
        prev_norm = torch.norm(prev_tangent, dim=-1)
        next_norm = torch.norm(next_tangent, dim=-1)

        # --- Apply length penalty with sum of squared norm to penalize uneven edge lengths on selected edges
        prev_length_loss = torch.pow(prev_norm, 2)
        next_length_loss = torch.pow(next_norm, 2)
        prev_length_loss[self.tensorskeleton.path_delim[1:-1] - 1] = 0  # Zero out invalid norms between paths
        prev_length_loss[self.tensorskeleton.path_delim[1:-1] - 2] = 0  # Zero out unwanted contribution to loss
        next_length_loss[self.tensorskeleton.path_delim[1:-1] - 1] = 0  # Zero out unwanted contribution to loss
        next_length_loss[self.tensorskeleton.path_delim[1:-1] - 2] = 0  # Zero out invalid norms between paths
        length_loss = prev_length_loss + next_length_loss
        total_length_loss = torch.sum(length_loss)

        # --- Detect corners:
        with torch.no_grad():
            middle_pos_int = middle_pos.round().long()
            middle_pos_int[:, 0] = torch.clamp(middle_pos_int[:, 0], 0, self.c0c2.shape[2] - 1)
            middle_pos_int[:, 1] = torch.clamp(middle_pos_int[:, 1], 0, self.c0c2.shape[3] - 1)
            middle_batch = path_batch[1:-1]
            middle_c0c2 = self.c0c2[middle_batch, :, middle_pos_int[:, 0], middle_pos_int[:, 1]]
            middle_uv = frame_field_utils.c0c2_to_uv(middle_c0c2)
            prev_tangent_closest_in_uv = frame_field_utils.compute_closest_in_uv(prev_tangent, middle_uv)
            next_tangent_closest_in_uv = frame_field_utils.compute_closest_in_uv(next_tangent, middle_uv)
            is_corner = prev_tangent_closest_in_uv != next_tangent_closest_in_uv
            is_corner[self.tensorskeleton.path_delim[1:-1] - 2] = 0  # Zero out invalid corners between sub-paths
            is_corner[self.tensorskeleton.path_delim[1:-1] - 1] = 0  # Zero out invalid corners between sub-paths
            is_corner_index = torch.nonzero(is_corner)[:, 0] + 1  # Shift due to first vertex not being represented in is_corner
            # TODO: evaluate running time of torch.sort: does it slow down the optimization much?
            sub_path_delim, sub_path_sort_indices = torch.sort(torch.cat([self.tensorskeleton.path_delim, is_corner_index]))
            sub_path_delim_is_corner = self.tensorskeleton.path_delim.shape[0] <= sub_path_sort_indices  # If condition is true, then the delimiter is from is_corner_index

        # --- Compute sub-path dissimilarity in the sense of the Ramer-Douglas-Peucker alg
        # dissimilarity is equal to the max distance of vertices to the straight line connecting the start and end points of the sub-path.
        with torch.no_grad():
            sub_path_start_index = sub_path_delim[:-1]
            sub_path_end_index = sub_path_delim[1:].clone()
            sub_path_end_index[~sub_path_delim_is_corner[1:]] -= 1  # For non-corner delimitators, have to shift
            sub_path_start_pos = path_pos[sub_path_start_index]
            sub_path_end_pos = path_pos[sub_path_end_index]
            sub_path_normal = sub_path_end_pos - sub_path_start_pos
            sub_path_normal = sub_path_normal / (torch.norm(sub_path_normal, dim=1)[:, None] + 1e-6)
            expanded_sub_path_start_pos = torch_scatter.gather_csr(sub_path_start_pos,
                                                                   sub_path_delim)
            expanded_sub_path_normal = torch_scatter.gather_csr(sub_path_normal,
                                                                 sub_path_delim)
            relative_path_pos = path_pos - expanded_sub_path_start_pos
            relative_path_pos_projected_lengh = torch.sum(relative_path_pos * expanded_sub_path_normal, dim=1)
            relative_path_pos_projected = relative_path_pos_projected_lengh[:, None] * expanded_sub_path_normal
            path_pos_distance = torch.norm(relative_path_pos - relative_path_pos_projected, dim=1)
            sub_path_max_distance = torch_scatter.segment_max_csr(path_pos_distance, sub_path_delim)[0]
            sub_path_small_dissimilarity_mask = sub_path_max_distance < self.curvature_dissimilarity_threshold

        # --- Compute curvature loss:
        # print("prev_norm:", prev_norm.min().item(), prev_norm.max().item())
        prev_dir = prev_tangent / (prev_norm[:, None] + 1e-6)
        next_dir = next_tangent / (next_norm[:, None] + 1e-6)
        dot = prev_dir[:, 0] * next_dir[:, 0] + \
              prev_dir[:, 1] * next_dir[:, 1]  # dot product
        det = prev_dir[:, 0] * next_dir[:, 1] - \
              prev_dir[:, 1] * next_dir[:, 0]  # determinant
        vertex_angles = torch.acos(dot) * torch.sign(det)  # TODO: remove acos for speed? Switch everything to signed dot product?
        # Save angles of detected corners:
        corner_angles = vertex_angles[is_corner_index - 1]  # -1 because of the shift of vertex_angles relative to path_pos
        # Compute the mean vertex angle for each sub-path separately:
        vertex_angles[sub_path_delim[1:-1] - 1] = 0  # Zero out invalid angles between paths as well as corner angles
        vertex_angles[self.tensorskeleton.path_delim[1:-1] - 2] = 0  # Zero out invalid angles between paths (caused by the junction points being in all paths of the junction)
        sub_path_vertex_angle_delim = sub_path_delim.clone()
        sub_path_vertex_angle_delim[-1] -= 2
        sub_path_sum_vertex_angle = torch_scatter.segment_sum_csr(vertex_angles, sub_path_vertex_angle_delim)
        sub_path_lengths = sub_path_delim[1:] - sub_path_delim[:-1]
        sub_path_lengths[sub_path_delim_is_corner[1:]] += 1  # Fix length of paths split by corners
        sub_path_valid_angle_count = sub_path_lengths - 2
        # print("sub_path_valid_angle_count:", sub_path_valid_angle_count.min().item(), sub_path_valid_angle_count.max().item())
        sub_path_mean_vertex_angles = sub_path_sum_vertex_angle / sub_path_valid_angle_count
        sub_path_mean_vertex_angles[sub_path_small_dissimilarity_mask] = 0  # Optimize sub-path with a small dissimilarity to have straight edges
        expanded_sub_path_mean_vertex_angles = torch_scatter.gather_csr(sub_path_mean_vertex_angles,
                                                                        sub_path_vertex_angle_delim)
        curvature_loss = torch.pow(vertex_angles - expanded_sub_path_mean_vertex_angles, 2)
        curvature_loss[sub_path_delim[1:-1] - 1] = 0  # Zero out loss for start vertex of inner sub-paths
        curvature_loss[self.tensorskeleton.path_delim[1:-1] - 2] = 0  # Zero out loss for end vertex of inner paths (caused by the junction points being in all paths of the junction)
        total_curvature_loss = torch.sum(curvature_loss)

        # --- Computer corner loss:
        corner_abs_angles = torch.abs(corner_angles)
        self.corner_angles = self.corner_angles.to(corner_abs_angles.device)
        corner_snap_dist = torch.abs(corner_abs_angles[:, None] - self.corner_angles)
        corner_snap_dist_optim_mask = corner_snap_dist < self.corner_angle_threshold
        corner_snap_dist_optim = corner_snap_dist[corner_snap_dist_optim_mask]
        corner_loss = torch.pow(corner_snap_dist_optim, 2)
        total_corner_loss = torch.sum(corner_loss)

        # --- Compute junction corner loss
        junction_corner = pos[self.junction_corner_index, :]
        junction_prev_tangent = junction_corner[:, 1, :] - junction_corner[:, 0, :]
        junction_next_tangent = junction_corner[:, 2, :] - junction_corner[:, 1, :]
        junction_prev_dir = junction_prev_tangent / (torch.norm(junction_prev_tangent, dim=-1)[:, None] + 1e-6)
        junction_next_dir = junction_next_tangent / (torch.norm(junction_next_tangent, dim=-1)[:, None] + 1e-6)
        junction_dot = junction_prev_dir[:, 0] * junction_next_dir[:, 0] + \
              junction_prev_dir[:, 1] * junction_next_dir[:, 1]  # dot product
        junction_abs_angles = torch.acos(junction_dot)
        self.junction_angles = self.junction_angles.to(junction_abs_angles.device)
        self.junction_angle_weights = self.junction_angle_weights.to(junction_abs_angles.device)
        junction_snap_dist = torch.abs(junction_abs_angles[:, None] - self.junction_angles)
        junction_snap_dist_optim_mask = junction_snap_dist < self.junction_angle_threshold
        junction_snap_dist *= self.junction_angle_weights[None, :]  # Apply weights per target angle (as we use the L1 norm, it works applying before the norm)
        junction_snap_dist_optim = junction_snap_dist[junction_snap_dist_optim_mask]
        junction_loss = torch.abs(junction_snap_dist_optim)
        total_junction_loss = torch.sum(junction_loss)

        losses_dict = {
            "align": total_align_loss.item(),
            "level": level_loss.item(),
            "length": total_length_loss.item(),
            "curvature": total_curvature_loss.item(),
            "corner": total_corner_loss.item(),
            "junction": total_junction_loss.item(),
        }
        # Get the loss coefs depending on the current step:
        data_coef = float(self.data_coef_interp(iter_num))
        length_coef = float(self.length_coef_interp(iter_num))
        crossfield_coef = float(self.crossfield_coef_interp(iter_num))
        curvature_coef = float(self.curvature_coef_interp(iter_num))
        corner_coef = float(self.corner_coef_interp(iter_num))
        junction_coef = float(self.junction_coef_interp(iter_num))
        # total_loss = data_coef * level_loss + length_coef * total_length_loss + crossfield_coef * total_align_loss + \
        #              curvature_coef * total_curvature_loss + corner_coef * total_corner_loss + junction_coef * total_junction_loss
        total_loss = data_coef * level_loss + length_coef * total_length_loss + crossfield_coef * total_align_loss + \
                     curvature_coef * total_curvature_loss + corner_coef * total_corner_loss + junction_coef * total_junction_loss

        # print(iter_num)
        # input("<Enter>...")

        return total_loss, losses_dict


class TensorSkeletonOptimizer:
    def __init__(self, config: dict, tensorskeleton: TensorSkeleton, indicator: torch.Tensor, c0c2: torch.Tensor):
        assert len(indicator.shape) == 3, f"indicator should be of shape (N, H, W), not {indicator.shape}"
        assert len(c0c2.shape) == 4 and c0c2.shape[1] == 4, f"c0c2 should be of shape (N, 4, H, W), not {c0c2.shape}"

        self.config = config
        self.tensorskeleton = tensorskeleton

        # Save endpoints that are tips so that they can be reset after each step (tips are not meant to be moved)
        self.is_tip = self.tensorskeleton.degrees == 1
        self.tip_pos = self.tensorskeleton.pos[self.is_tip]

        # Require grads for graph.pos: this is what is optimized
        self.tensorskeleton.pos.requires_grad = True

        level = config["data_level"]
        self.criterion = AlignLoss(self.tensorskeleton, indicator, level, c0c2, config["loss_params"])
        self.optimizer = torch.optim.RMSprop([tensorskeleton.pos], lr=config["lr"], alpha=0.9)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, config["gamma"])

    def step(self, iter_num):
        self.optimizer.zero_grad()

        # tic = time.time()
        loss, losses_dict = self.criterion(self.tensorskeleton.pos, iter_num)

        # toc = time.time()
        # print(f"Forward: {toc - tic}s")

        # print("loss:", loss.item())
        # tic = time.time()
        loss.backward()

        pos_gard_is_nan = torch.isnan(self.tensorskeleton.pos.grad).any().item()
        if pos_gard_is_nan:
            print(f"{iter_num} pos.grad is nan")

        # print(self.tensorskeleton.pos.grad)
        # print(torch.norm(self.tensorskeleton.pos.grad, dim=1).max().item())
        # toc = time.time()
        # print(f"Backward: {toc - tic}s")
        self.optimizer.step()

        # Move tips back:
        with torch.no_grad():
            # TODO: snap to nearest image border
            self.tensorskeleton.pos[self.is_tip] = self.tip_pos

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return loss.item(), losses_dict

    def optimize(self) -> TensorSkeleton:
        if DEBUG:
            optim_iter = tqdm(range(self.config["loss_params"]["coefs"]["step_thresholds"][-1]), desc="Gradient descent", leave=True)
            for iter_num in optim_iter:
                loss, losses_dict = self.step(iter_num)
                optim_iter.set_postfix(loss=loss, **losses_dict)
        else:
            for iter_num in range(self.config["loss_params"]["coefs"]["step_thresholds"][-1]):
                loss, losses_dict = self.step(iter_num)
        # for iter_num in range(self.config["loss_params"]["coefs"]["step_thresholds"][-1]):
        #     loss, losses_dict = self.step(iter_num)
        return self.tensorskeleton


def shapely_postprocess(polylines, np_indicator, tolerance, config):
    if type(tolerance) == list:
        # Use several tolerance values for simplification. return a dict with all results
        out_polygons_dict = {}
        out_probs_dict = {}
        for tol in tolerance:
            out_polygons, out_probs = shapely_postprocess(polylines, np_indicator, tol, config)
            out_polygons_dict["tol_{}".format(tol)] = out_polygons
            out_probs_dict["tol_{}".format(tol)] = out_probs
        return out_polygons_dict, out_probs_dict
    else:
        height = np_indicator.shape[0]
        width = np_indicator.shape[1]

        # Convert to Shapely:
        # tic = time.time()
        line_string_list = [shapely.geometry.LineString(polyline[:, ::-1]) for polyline in polylines]
        line_string_list = [line_string.simplify(tolerance, preserve_topology=True) for line_string in line_string_list]
        # toc = time.time()
        # print(f"simplify: {toc - tic}s")

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

        # tic = time.time()
        multi_line_string = shapely.ops.unary_union(line_string_list)
        # toc = time.time()
        # print(f"shapely.ops.unary_union: {toc - tic}s")

        # debug_print("polygonize_full")

        # Find polygons:
        polygons = shapely.ops.polygonize(multi_line_string)
        polygons = list(polygons)

        # debug_print("Remove small polygons")

        # Remove small polygons
        # tic = time.time()
        polygons = [polygon for polygon in polygons if
                    config["min_area"] < polygon.area]
        # toc = time.time()
        # print(f"Remove small polygons: {toc - tic}s")

        # debug_print("Remove low prob polygons")

        # Remove low prob polygons
        # tic = time.time()

        filtered_polygons = []
        filtered_polygon_probs = []
        for polygon in polygons:
            prob = polygonize_utils.compute_geom_prob(polygon, np_indicator)
            # print("acm:", np_indicator.min(), np_indicator.mean(), np_indicator.max(), prob)
            if config["seg_threshold"] < prob:
                filtered_polygons.append(polygon)
                filtered_polygon_probs.append(prob)

        # toc = time.time()
        # print(f"Remove low prob polygons: {toc - tic}s")

        return filtered_polygons, filtered_polygon_probs


def post_process(polylines, np_indicator, np_crossfield, config):

    # debug_print("Corner-aware simplification")
    # Simplify contours a little to avoid some close-together corner-detection:
    # tic = time.time()
    u, v = math_utils.compute_crossfield_uv(np_crossfield)  # u, v are complex arrays
    corner_masks = frame_field_utils.detect_corners(polylines, u, v)
    polylines = polygonize_utils.split_polylines_corner(polylines, corner_masks)
    # toc = time.time()
    # print(f"Corner detect: {toc - tic}s")

    polygons, probs = shapely_postprocess(polylines, np_indicator, config["tolerance"], config)
    return polygons, probs


def get_skeleton(np_edge_mask, config):
    """

    @param np_edge_mask:
    @param config:
    @return:
    """
    # --- Skeletonize
    # tic = time.time()
    # Pad np_edge_mask first otherwise pixels on the bottom and right are lost after skeletonize:
    pad_width = 2
    np_edge_mask_padded = np.pad(np_edge_mask, pad_width=pad_width, mode="edge")
    skeleton_image = skimage.morphology.skeletonize(np_edge_mask_padded)
    skeleton_image = skeleton_image[pad_width:-pad_width, pad_width:-pad_width]

    # toc = time.time()
    # debug_print(f"skimage.morphology.skeletonize: {toc - tic}s")

    # tic = time.time()

    # if skeleton_image.max() == False:
    #     # There is no polylines to be detected
    #     return [], np.empty((0, 2), dtype=np.bool)

    skeleton = Skeleton()
    if 0 < skeleton_image.sum():
        # skan does not work in some cases (paths of 2 pixels or less, etc) which raises a ValueError, in witch case we continue with an empty skeleton.
        try:
            skeleton = skan.Skeleton(skeleton_image, keep_images=False)
            # skan.skeleton sometimes returns skeleton.coordinates.shape[0] != skeleton.degrees.shape[0] or
            # skeleton.coordinates.shape[0] != skeleton.paths.indices.max() + 1
            # Slice coordinates accordingly
            skeleton.coordinates = skeleton.coordinates[:skeleton.paths.indices.max() + 1]
            if skeleton.coordinates.shape[0] != skeleton.degrees.shape[0]:
                raise ValueError(f"skeleton.coordinates.shape[0] = {skeleton.coordinates.shape[0]} while skeleton.degrees.shape[0] = {skeleton.degrees.shape[0]}. They should be of same size.")
        except ValueError as e:
            if DEBUG:
                print_utils.print_warning(
                    f"WARNING: skan.Skeleton raised a ValueError({e}). skeleton_image has {skeleton_image.sum()} true values. Continuing without detecting skeleton in this image...")
                skimage.io.imsave("np_edge_mask.png", np_edge_mask.astype(np.uint8) * 255)
                skimage.io.imsave("skeleton_image.png", skeleton_image.astype(np.uint8) * 255)

    # toc = time.time()
    #debug_print(f"skan.Skeleton: {toc - tic}s")

    # tic = time.time()

    # # --- For each endpoint, see if it's a tip or not
    # endpoints_src = skeleton.paths.indices[skeleton.paths.indptr[:-1]]
    # endpoints_dst = skeleton.paths.indices[skeleton.paths.indptr[1:] - 1]
    # deg_src = skeleton.degrees[endpoints_src]
    # deg_dst = skeleton.degrees[endpoints_dst]
    # is_tip_array = np.stack([deg_src == 1, deg_dst == 1], axis=1)

    # toc = time.time()
    # debug_print(f"Convert to polylines: {toc - tic}s")

    return skeleton


def get_marching_squares_skeleton(np_int_prob, config):
    """

    @param np_int_prob:
    @param config:
    @return:
    """
    # tic = time.time()
    contours = skimage.measure.find_contours(np_int_prob, config["data_level"], fully_connected='low', positive_orientation='high')
    # Keep contours with more than 3 vertices and large enough area
    contours = [contour for contour in contours if 3 <= contour.shape[0] and
                config["min_area"] < shapely.geometry.Polygon(contour).area]

    # If there are no contours, return empty skeleton
    if len(contours) == 0:
        return Skeleton()

    toc = time.time()
    #debug_print(f"get_skeleton_polylines: {toc - tic}s")
    # Simplify contours a tiny bit:
    # contours = [skimage.measure.approximate_polygon(contour, tolerance=0.001) for contour in contours]

    # Convert into skeleton representation
    coordinates = []
    indices_offset = 0
    indices = []
    indptr = [0]
    degrees = []

    for i, contour in enumerate(contours):
        # Check if it is a closed contour
        is_closed = np.max(np.abs(contour[0] - contour[-1])) < 1e-6
        if is_closed:
            _coordinates = contour[:-1, :]  # Don't include redundant vertex in coordinates
        else:
            _coordinates = contour
        _degrees = 2 * np.ones(_coordinates.shape[0], dtype=np.long)
        if not is_closed:
            _degrees[0] = 1
            _degrees[-1] = 1
        _indices = list(range(indices_offset, indices_offset + _coordinates.shape[0]))
        if is_closed:
            _indices.append(_indices[0])  # Close contour with indices
        coordinates.append(_coordinates)
        degrees.append(_degrees)
        indices.extend(_indices)
        indptr.append(indptr[-1] + len(_indices))
        indices_offset += _coordinates.shape[0]

    coordinates = np.concatenate(coordinates, axis=0)
    degrees = np.concatenate(degrees, axis=0)
    indices = np.array(indices)
    indptr = np.array(indptr)

    paths = Paths(indices, indptr)
    skeleton = Skeleton(coordinates, paths, degrees)

    return skeleton


# @profile
def compute_skeletons(seg_batch, config, spatial_gradient, pool=None) -> List[Skeleton]:
    assert len(seg_batch.shape) == 4 and seg_batch.shape[
        1] <= 3, "seg_batch should be (N, C, H, W) with C <= 3, not {}".format(seg_batch.shape)

    int_prob_batch = seg_batch[:, 0, :, :]
    if config["init_method"] == "marching_squares":
        # Only interior segmentation is available, initialize with marching squares
        np_int_prob_batch = int_prob_batch.cpu().numpy()
        get_marching_squares_skeleton_partial = functools.partial(get_marching_squares_skeleton, config=config)
        if pool is not None:
            skeletons_batch = pool.map(get_marching_squares_skeleton_partial, np_int_prob_batch)
        else:
            skeletons_batch = list(map(get_marching_squares_skeleton_partial, np_int_prob_batch))
    elif config["init_method"] == "skeleton":
        tic_correct = time.time()
        # Edge segmentation is also available, initialize with skan.Squeleton
        corrected_edge_prob_batch = config["data_level"] < int_prob_batch  # Convet to mask
        corrected_edge_prob_batch = corrected_edge_prob_batch[:, None, :, :].float() # Convet to float for spatial grads
        corrected_edge_prob_batch = 2 * spatial_gradient(corrected_edge_prob_batch)[:, 0, :, :]  # (b, 2, h, w), Normalize (kornia normalizes to -0.5, 0.5 for input in [0, 1])
        corrected_edge_prob_batch = corrected_edge_prob_batch.norm(dim=1)  # (b, h, w), take the gradient norm
        # int_contours_mask_batch = compute_contours_mask(int_mask_batch[:, None, :, :])[:, 0, :, :]
        # corrected_edge_prob_batch = int_contours_mask_batch.float()
        if 2 <= seg_batch.shape[1]:
            corrected_edge_prob_batch = torch.clamp(seg_batch[:, 1, :, :] + corrected_edge_prob_batch, 0, 1)
        # Save for viz
        # save_edge_prob_map = (corrected_edge_prob_batch[0].cpu().numpy() * 255).astype(np.uint8)[:, :, None]
        # skimage.io.imsave("corrected_edge_prob_batch.png", save_edge_prob_map)

        toc_correct = time.time()
        #debug_print(f"Correct edge prob map: {toc_correct - tic_correct}s")

        # --- Init skeleton
        corrected_edge_mask_batch = config["data_level"] < corrected_edge_prob_batch
        np_corrected_edge_mask_batch = corrected_edge_mask_batch.cpu().numpy()

        get_skeleton_partial = functools.partial(get_skeleton, config=config)
        # polylines_batch = []
        # is_tip_batch = []
        # for np_corrected_edge_mask in np_corrected_edge_mask_batch:
        #     polylines, is_tip_array = get_skeleton_polylines_partial(np_corrected_edge_mask)
        #     polylines_batch.append(polylines)
        #     is_tip_batch.append(is_tip_array)
        # tic = time.time()
        if pool is not None:
            skeletons_batch = pool.map(get_skeleton_partial, np_corrected_edge_mask_batch)
        else:
            skeletons_batch = list(map(get_skeleton_partial, np_corrected_edge_mask_batch))
        # toc = time.time()
        #debug_print(f"get_skeleton_polylines: {toc - tic}s")
    else:
        raise NotImplementedError(f"init_method '{config['init_method']}' not recognized. Valid init methods are 'skeleton' and 'marching_squares'")

    return skeletons_batch


def skeleton_to_polylines(skeleton: Skeleton) -> List[np.ndarray]:
    polylines = []
    for path_i in range(skeleton.paths.indptr.shape[0] - 1):
        start, stop = skeleton.paths.indptr[path_i:path_i + 2]
        path_indices = skeleton.paths.indices[start:stop]
        path_coordinates = skeleton.coordinates[path_indices]
        polylines.append(path_coordinates)
    return polylines


class PolygonizerASM:
    def __init__(self, config, pool=None):
        self.config = config
        self.pool = pool
        self.spatial_gradient = torch_lydorn.kornia.filters.SpatialGradient(mode="scharr", coord="ij", normalized=True,
                                                                            device=self.config["device"], dtype=torch.float)

    # @profile
    def __call__(self, seg_batch, crossfield_batch, pre_computed=None):
        tic_start = time.time()

        assert len(seg_batch.shape) == 4 and seg_batch.shape[
            1] <= 3, "seg_batch should be (N, C, H, W) with C <= 3, not {}".format(seg_batch.shape)
        assert len(crossfield_batch.shape) == 4 and crossfield_batch.shape[
            1] == 4, "crossfield_batch should be (N, 4, H, W)"
        assert seg_batch.shape[0] == crossfield_batch.shape[0], "Batch size for seg and crossfield should match"


        seg_batch = seg_batch.to(self.config["device"])
        crossfield_batch = crossfield_batch.to(self.config["device"])

        # --- Get initial polylines
        # tic = time.time()
        skeletons_batch = compute_skeletons(seg_batch, self.config, self.spatial_gradient, pool=self.pool)
        # toc = time.time()
        # debug_print(f"Init polylines: {toc - tic}s")

        # # --- Compute distance transform
        # tic = time.time()
        #
        # np_int_mask_batch = int_mask_batch.cpu().numpy()
        # np_dist_batch = np.empty(np_int_mask_batch.shape)
        # for batch_i in range(np_int_mask_batch.shape[0]):
        #     dist_1 = cv.distanceTransform(np_int_mask_batch[batch_i].astype(np.uint8), distanceType=cv.DIST_L2, maskSize=cv.DIST_MASK_5, dstType=cv.CV_64F)
        #     dist_2 = cv.distanceTransform(1 - np_int_mask_batch[batch_i].astype(np.uint8), distanceType=cv.DIST_L2, maskSize=cv.DIST_MASK_5, dstType=cv.CV_64F)
        #     np_dist_batch[0] = dist_1 + dist_2
        # dist_batch = torch.from_numpy(np_dist_batch)
        #
        # toc = time.time()
        # print(f"Distance transform: {toc - tic}s")

        # --- Optimize skeleton:
        tensorskeleton = skeletons_to_tensorskeleton(skeletons_batch, device=self.config["device"])

        # --- Check if tensorskeleton is empty
        if tensorskeleton.num_paths == 0:
            batch_size = seg_batch.shape[0]
            polygons_batch = [[]]*batch_size
            probs_batch = [[]]*batch_size
            return polygons_batch, probs_batch

        int_prob_batch = seg_batch[:, 0, :, :]
        # dist_batch = dist_batch.to(config["device"])
        tensorskeleton_optimizer = TensorSkeletonOptimizer(self.config, tensorskeleton, int_prob_batch,
                                                           crossfield_batch)

        if DEBUG:
            # Animation of optimization
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.autoscale(False)
            ax.axis('equal')
            ax.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Plot without margins

            image = int_prob_batch.cpu().numpy()[0]
            ax.imshow(image, cmap=plt.cm.gray)

            out_skeletons_batch = tensorskeleton_to_skeletons(tensorskeleton)
            polylines_batch = [skeleton_to_polylines(skeleton) for skeleton in out_skeletons_batch]
            out_polylines = [shapely.geometry.LineString(polyline[:, ::-1]) for polyline in polylines_batch[0]]
            artists = plot_utils.plot_geometries(ax, out_polylines, draw_vertices=True, linewidths=1)

            optim_pbar = tqdm(desc="Gradient descent", leave=True, total=self.config["loss_params"]["coefs"]["step_thresholds"][-1])

            def init():  # only required for blitting to give a clean slate.
                for artist, polyline in zip(artists, polylines_batch[0]):
                    artist.set_xdata([np.nan] * polyline.shape[0])
                    artist.set_ydata([np.nan] * polyline.shape[0])
                return artists

            def animate(i):
                loss, losses_dict = tensorskeleton_optimizer.step(i)
                optim_pbar.update(int(2 * i / self.config["loss_params"]["coefs"]["step_thresholds"][-1]))
                optim_pbar.set_postfix(loss=loss, **losses_dict)
                out_skeletons_batch = tensorskeleton_to_skeletons(tensorskeleton)
                polylines_batch = [skeleton_to_polylines(skeleton) for skeleton in out_skeletons_batch]
                for artist, polyline in zip(artists, polylines_batch[0]):
                    artist.set_xdata(polyline[:, 1])
                    artist.set_ydata(polyline[:, 0])
                return artists

            ani = animation.FuncAnimation(
                fig, animate, init_func=init, interval=0, blit=True, frames=self.config["loss_params"]["coefs"]["step_thresholds"][-1], repeat=False)

            # To save the animation, use e.g.
            #
            # ani.save("movie.mp4")
            #
            # or
            #
            # writer = animation.FFMpegWriter(
            #     fps=15, metadata=dict(artist='Me'), bitrate=1800)
            # ani.save("movie.mp4", writer=writer)

            plt.show()
        else:
            tensorskeleton = tensorskeleton_optimizer.optimize()

        out_skeletons_batch = tensorskeleton_to_skeletons(tensorskeleton)

        # --- Convert the skeleton representation into polylines
        polylines_batch = [skeleton_to_polylines(skeleton) for skeleton in out_skeletons_batch]

        # toc = time.time()
        #debug_print(f"Optimize skeleton: {toc - tic}s")

        # --- Post-process:
        # debug_print("Post-process")
        # tic = time.time()

        np_crossfield_batch = np.transpose(crossfield_batch.cpu().numpy(), (0, 2, 3, 1))
        np_int_prob_batch = int_prob_batch.cpu().numpy()
        post_process_partial = partial(post_process, config=self.config)
        if self.pool is not None:
            polygons_probs_batch = self.pool.starmap(post_process_partial,
                                                zip(polylines_batch, np_int_prob_batch, np_crossfield_batch))
        else:
            polygons_probs_batch = map(post_process_partial, polylines_batch, np_int_prob_batch,
                                       np_crossfield_batch)
        polygons_batch, probs_batch = zip(*polygons_probs_batch)

        # toc = time.time()
        #debug_print(f"Post-process: {toc - tic}s")

        toc_end = time.time()
        #debug_print(f"Total: {toc_end - tic_start}s")

        if DEBUG:
            # --- display results
            import matplotlib.pyplot as plt
            image = np_int_prob_batch[0]
            polygons = polygons_batch[0]
            out_polylines = [shapely.geometry.LineString(polyline[:, ::-1]) for polyline in polylines_batch[0]]

            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 16), sharex=True, sharey=True)
            ax = axes.ravel()

            ax[0].imshow(image, cmap=plt.cm.gray)
            plot_utils.plot_geometries(ax[0], out_polylines, draw_vertices=True, linewidths=1)
            ax[0].axis('off')
            ax[0].set_title('original', fontsize=20)

            # ax[1].imshow(skeleton, cmap=plt.cm.gray)
            # ax[1].axis('off')
            # ax[1].set_title('skeleton', fontsize=20)

            fig.tight_layout()
            plt.show()

        return polygons_batch, probs_batch


def polygonize(seg_batch, crossfield_batch, config, pool=None, pre_computed=None):
    polygonizer_asm = PolygonizerASM(config, pool=pool)
    return polygonizer_asm(seg_batch, crossfield_batch, pre_computed=pre_computed)


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
        "init_method": "skeleton",  # Can be either skeleton or marching_squares
        "data_level": 0.5,
        "loss_params": {
            "coefs": {
                "step_thresholds": [0,  100,  200,  300],  # From 0 to 500: gradually go from coefs[0] to coefs[1]
                "data":          [1.0,  0.1,  0.0,    0],
                "crossfield":    [0.0, 0.05,  0.0,    0],
                "length":        [0.1, 0.01,  0.0,    0],
                "curvature":     [0.0,  0.0,  1.0, 1e-6],
                "corner":        [0.0,  0.0,  0.5, 1e-6],
                "junction":      [0.0,  0.0,  0.5, 1e-6],
            },
            "curvature_dissimilarity_threshold": 2,  # In pixels: for each sub-paths, if the dissimilarity (in the same sense as in the Ramer-Douglas-Peucker alg) is lower than curvature_dissimilarity_threshold, then optimize the curve angles to be zero.
            "corner_angles": [45, 90, 135],  # In degrees: target angles for corners.
            "corner_angle_threshold": 22.5,  # If a corner angle is less than this threshold away from any angle in corner_angles, optimize it.
            "junction_angles": [0, 45, 90, 135],  # In degrees: target angles for junction corners.
            "junction_angle_weights": [1, 0.01, 0.1, 0.01],  # Order of decreassing importance: straight, right-angle, then 45° junction corners.
            "junction_angle_threshold": 22.5,  # If a junction corner angle is less than this threshold away from any angle in junction_angles, optimize it.
        },
        "lr": 0.1,
        "gamma": 0.995,
        "device": "cuda",
        "tolerance": 1.0,
        "seg_threshold": 0.5,
        "min_area": 10,
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
        if args.seg_filepath is not None:
            seg = skimage.io.imread(args.seg_filepath) / 255
        else:
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

        # # Add samples to batch to increase batch size for testing
        # batch_size = 4
        # seg_batch = seg_batch.repeat((batch_size, 1, 1, 1))
        # crossfield_batch = crossfield_batch.repeat((batch_size, 1, 1, 1))

        out_contours_batch, out_probs_batch = polygonize(seg_batch, crossfield_batch, config)

        polygons = out_contours_batch[0]

        # Save geojson
        # save_utils.save_geojson(polygons, base_filepath + extra_name, name="poly_asm", image_filepath=args.im_filepath)

        # Save shapefile
        # save_utils.save_shapefile(polygons, base_filepath + extra_name, "poly_asm", args.im_filepath)

        # Save pdf viz
        filepath = base_filepath + extra_name + ".poly_asm.pdf"
        # plot_utils.save_poly_viz(image, polygons, filepath, linewidths=1, draw_vertices=True, color_choices=[[0, 1, 0, 1]])
        plot_utils.save_poly_viz(image, polygons, filepath, markersize=30, linewidths=1, draw_vertices=True)
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
        config = {
            "init_method": "marching_squares",  # Can be either skeleton or marching_squares
            "data_level": 0.5,
            "loss_params": {
                "coefs": {
                    "step_thresholds": [0, 100, 200, 300],  # From 0 to 500: gradually go from coefs[0] to coefs[1]
                    "data": [1.0, 0.1, 0.0, 0.0],
                    "crossfield": [0.0, 0.05, 0.0, 0.0],
                    "length": [0.1, 0.01, 0.0, 0.0],
                    "curvature": [0.0, 0.0, 0.0, 0.0],
                    "corner": [0.0, 0.0, 0.0, 0.0],
                    "junction": [0.0, 0.0, 0.0, 0.0],
                },
                "curvature_dissimilarity_threshold": 2,
                # In pixels: for each sub-paths, if the dissimilarity (in the same sense as in the Ramer-Douglas-Peucker alg) is lower than straightness_threshold, then optimize the curve angles to be zero.
                "corner_angles": [45, 90, 135],  # In degrees: target angles for corners.
                "corner_angle_threshold": 22.5,
                # If a corner angle is less than this threshold away from any angle in corner_angles, optimize it.
                "junction_angles": [0, 45, 90, 135],  # In degrees: target angles for junction corners.
                "junction_angle_weights": [1, 0.01, 0.1, 0.01],
                # Order of decreassing importance: straight, right-angle, then 45° junction corners.
                "junction_angle_threshold": 22.5,
                # If a junction corner angle is less than this threshold away from any angle in junction_angles, optimize it.
            },
            "lr": 0.01,
            "gamma": 0.995,
            "device": "cuda",
            "tolerance": 0.5,
            "seg_threshold": 0.5,
            "min_area": 10,
        }

        seg = np.zeros((6, 8, 1))
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
        u[:4, :4] *= np.exp(1j * np.pi / 4)
        v[:4, :4] *= np.exp(1j * np.pi / 4)
        # Add slope corners
        # u[:2, 4:6] *= np.exp(1j * np.pi / 4)
        # v[4:, :2] *= np.exp(- 1j * np.pi / 4)

        crossfield = math_utils.compute_crossfield_c0c2(u, v)

        seg_batch = torch.tensor(np.transpose(seg[:, :, :2], (2, 0, 1)), dtype=torch.float)[None, ...]
        crossfield_batch = torch.tensor(np.transpose(crossfield, (2, 0, 1)), dtype=torch.float)[None, ...]

        # Add samples to batch to increase batch size
        batch_size = 16
        seg_batch = seg_batch.repeat((batch_size, 1, 1, 1))
        crossfield_batch = crossfield_batch.repeat((batch_size, 1, 1, 1))

        out_contours_batch, out_probs_batch = polygonize(seg_batch, crossfield_batch, config)

        polygons = out_contours_batch[0]

        filepath = "demo_poly_asm.pdf"
        plot_utils.save_poly_viz(seg[:, :, 0], polygons, filepath, linewidths=0.5, draw_vertices=True, crossfield=crossfield)


if __name__ == '__main__':
    main()
