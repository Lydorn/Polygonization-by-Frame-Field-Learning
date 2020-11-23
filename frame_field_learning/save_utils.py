import itertools
import os

import numpy as np
import pycocotools.mask
import shapely.geometry
import torch
import skimage.morphology
import skimage.measure

from lydorn_utils import python_utils, geo_utils, math_utils

from frame_field_learning import plot_utils, local_utils
import tifffile


def get_save_filepath(base_filepath, name=None, ext=""):
    if type(base_filepath) is tuple:
        if name is not None:
            save_filepath = os.path.join(base_filepath[0], name, base_filepath[1] + ext)
        else:
            save_filepath = os.path.join(base_filepath[0], base_filepath[1] + ext)
    elif type(base_filepath) is str:
        if name is not None:
            save_filepath = base_filepath + "." + name + ext
        else:
            save_filepath = base_filepath + ext
    else:
        raise TypeError(f"base_filepath should be either of tuple or str, not {type(base_filepath)}")
    os.makedirs(os.path.dirname(save_filepath), exist_ok=True)
    return save_filepath


def save_outputs(tile_data, config, eval_dirpath, split_name, flag_filepath_format):
    # print("--- save_outputs() ---")
    split_eval_dirpath = os.path.join(eval_dirpath, split_name)
    if not os.path.exists(split_eval_dirpath):
        os.makedirs(split_eval_dirpath)

    base_filepath = os.path.join(split_eval_dirpath, tile_data["name"])
    if "image_relative_filepath" in tile_data:
        image_filepath = os.path.join(config["data_root_dir"], tile_data["image_relative_filepath"])
    elif "image_filepath" in tile_data:
        image_filepath = tile_data["image_filepath"]
    else:
        raise ValueError("Could not get image_filepath from tile_data")

    if config["eval_params"]["save_individual_outputs"]["image"]:
        src_filepath = "/data" + image_filepath  # Because we are executing in Docker, this is a hack!
        filepath = base_filepath + ".image"
        if os.path.islink(filepath):
            os.remove(filepath)
        os.symlink(src_filepath, filepath)
    if config["eval_params"]["save_individual_outputs"]["seg_gt"]:
        save_seg(tile_data["gt_polygons_image"], base_filepath, "seg.gt", image_filepath)
    if config["eval_params"]["save_individual_outputs"]["seg"]:
        save_seg(tile_data["seg"], base_filepath, "seg", image_filepath)
    if config["eval_params"]["save_individual_outputs"]["seg_mask"]:
        save_seg_mask(tile_data["seg_mask"], (split_eval_dirpath, tile_data["name"]), "seg_mask", image_filepath)
    if config["eval_params"]["save_individual_outputs"]["seg_opencities_mask"]:
        save_opencities_mask(tile_data["seg_mask"], base_filepath, "drivendata",
                             image_filepath)
    if config["eval_params"]["save_individual_outputs"]["seg_luxcarta"]:
        save_seg_luxcarta_format(tile_data["seg"], base_filepath, "seg_luxcarta_format",
                                 image_filepath)

    if config["eval_params"]["save_individual_outputs"]["crossfield"]:
        save_crossfield(tile_data["crossfield"], base_filepath, "crossfield")
    if config["eval_params"]["save_individual_outputs"]["uv_angles"]:
        save_uv_angles(tile_data["crossfield"], base_filepath, "uv_angles",
                       image_filepath)

    if config["eval_params"]["save_individual_outputs"]["poly_shapefile"]:
        save_shapefile(tile_data["polygons"], base_filepath, "poly_shapefile", image_filepath)
    if config["eval_params"]["save_individual_outputs"]["poly_geojson"]:
        save_geojson(tile_data["polygons"], base_filepath, "poly_geojson", image_filepath)
    if "poly_viz" in config["eval_params"]["save_individual_outputs"] and config["eval_params"]["save_individual_outputs"]["poly_viz"]:
        save_poly_viz(tile_data["image"], tile_data["polygons"], tile_data["polygon_probs"], base_filepath, "poly_viz")

    if "raw_pred" in config["eval_params"]["save_individual_outputs"] and config["eval_params"]["save_individual_outputs"]["raw_pred"]:
        save_raw_pred(tile_data, base_filepath, "raw_pred")

    # Save a flag file to mark this sample as evaluated
    # pathlib.Path(flag_filepath_format.format(tile_data["name"])).touch()

    # print("Finished saving")


def save_seg(seg, base_filepath, name, image_filepath):
    seg = np.transpose(seg.numpy(), (1, 2, 0))
    # seg = torch_utils.to_numpy_image(seg)
    seg_display = plot_utils.get_seg_display(seg)
    if seg_display.dtype != np.uint8:
        seg_display = (255 * seg_display).astype(np.uint8)
    seg_display_filepath = get_save_filepath(base_filepath, name, ".tif")
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     skimage.io.imsave(seg_display_filepath, seg_display)
    geo_utils.save_image_as_geotiff(seg_display_filepath, seg_display, image_filepath)


def save_seg_mask(seg_mask, base_filepath, name, image_filepath):
    seg_mask = seg_mask.numpy()
    out = (255 * seg_mask).astype(np.uint8)[:, :, None]
    out_filepath = get_save_filepath(base_filepath, name, ".tif")
    geo_utils.save_image_as_geotiff(out_filepath, out, image_filepath)


def save_seg_luxcarta_format(seg, base_filepath, name, image_filepath):
    seg = np.transpose(seg.numpy(), (1, 2, 0))
    seg_luxcarta = np.zeros((seg.shape[0], seg.shape[1], 1), dtype=np.uint8)
    seg_interior = 0.5 < seg[..., 0]
    seg_luxcarta[seg_interior] = 1
    if 2 <= seg.shape[2]:
        seg_edge = 0.5 < seg[..., 1]
        seg_luxcarta[seg_edge] = 2
    seg_luxcarta_filepath = get_save_filepath(base_filepath, name, ".tif")
    geo_utils.save_image_as_geotiff(seg_luxcarta_filepath, seg_luxcarta, image_filepath)


def save_poly_viz(image, polygons, polygon_probs, base_filepath, name):
    if type(image) == torch.Tensor:
        image = np.transpose(image.numpy(), (1, 2, 0))

    if type(polygons) == dict:
        # Means several methods/settings were used
        for key in polygons.keys():
            save_poly_viz(image, polygons[key], polygon_probs[key], base_filepath, name + "." + key)
    elif type(polygons) == list:
        filepath = get_save_filepath(base_filepath, name, ".pdf")
        plot_utils.save_poly_viz(image, polygons, filepath, polygon_probs=polygon_probs)
    else:
        raise TypeError("polygons has unrecognized type {}".format(type(polygons)))


def save_shapefile(polygons, base_filepath, name, image_filepath):
    if type(polygons) == dict:
        # Means several methods/settings were used
        for key, item in polygons.items():
            save_shapefile(item, base_filepath, name + "." + key, image_filepath)
    elif type(polygons) == list:
        filepath = get_save_filepath(base_filepath, name, ".shp")
        if type(polygons[0]) == np.array:
            geo_utils.save_shapefile_from_polygons(polygons, image_filepath, filepath)
        elif type(polygons[0]) == shapely.geometry.polygon.Polygon:
            geo_utils.save_shapefile_from_shapely_polygons(polygons, image_filepath, filepath)
    else:
        raise TypeError("polygons has unrecognized type {}".format(type(polygons)))


def save_geojson(polygons, base_filepath, name=None, image_filepath=None):
    # TODO: add georef and features
    filepath = get_save_filepath(base_filepath, name, ".geojson")
    polygons_geometry_collection = shapely.geometry.collection.GeometryCollection(polygons)
    geojson = shapely.geometry.mapping(polygons_geometry_collection)
    python_utils.save_json(filepath, geojson)


def poly_coco(polygons: list, polygon_probs: list, image_id: int):
    if type(polygons) == dict:
        # Means several methods/settings were used
        annotations_dict = {}
        for key in polygons.keys():
            _polygons = polygons[key]
            _polygon_probs = polygon_probs[key]
            annotations_dict[key] = poly_coco(_polygons, _polygon_probs, image_id)
        return annotations_dict
    elif type(polygons) == list:
        annotations = []
        for polygon, prob in zip(polygons, polygon_probs):
            bbox = np.round([polygon.bounds[0], polygon.bounds[1],
                             polygon.bounds[2] - polygon.bounds[0], polygon.bounds[3] - polygon.bounds[1]], 2)
            exterior = list(np.round(np.array(polygon.exterior.coords).reshape(-1), 2))
            # interiors = [list(np.round(np.array(interior.coords).reshape(-1), 2)) for interior in polygon.interiors]
            # segmentation = [exterior, *interiors]
            segmentation = [exterior]
            score = prob
            annotation = {
                "category_id": 100,  # Building
                "bbox": list(bbox),
                "segmentation": segmentation,
                "score": score,
                "image_id": image_id}
            annotations.append(annotation)
        return annotations
    else:
        raise TypeError("polygons has unrecognized type {}".format(type(polygons)))


def save_poly_coco(annotations: list, base_filepath: str):
    """

    @param annotations: Either [[annotation1 of im1, annotation2 of im1, ...], ...] or [{"method1": [annotation1 of im1, ...]}, ...]
    @param base_filepath:
    @return:
    """
    # seg_coco_list is either a list of annotations or a list of dictionaries for each method (and sub-methods) used
    if type(annotations[0]) == dict:
        # Means several methods/settings were used
        # Transform list of dicts to a dict of lists:
        dictionary = local_utils.list_of_dicts_to_dict_of_lists(annotations)

        dictionary = local_utils.flatten_dict(dictionary)

        for key, _annotations in dictionary.items():
            out_filepath = base_filepath + "." + key + ".json"
            python_utils.save_json(out_filepath, _annotations)
    elif type(annotations[0]) == list:
        # Concatenate all lists
        flattened_annotations = list(itertools.chain(*annotations))
        out_filepath = get_save_filepath(base_filepath, None, ".json")
        python_utils.save_json(out_filepath, flattened_annotations)
    else:
        raise TypeError("annotations has unrecognized type {}".format(type(annotations)))


def seg_coco(sample):
    annotations = []
    # Have to convert binary mask to a list of annotations
    seg_interior = sample["seg"][0, :, :].numpy()
    seg_mask = sample["seg_mask"].numpy()
    labels = skimage.morphology.label(seg_mask)
    properties = skimage.measure.regionprops(labels, cache=True)
    for i, contour_props in enumerate(properties):
        skimage_bbox = contour_props["bbox"]
        obj_mask = seg_mask[skimage_bbox[0]:skimage_bbox[2], skimage_bbox[1]:skimage_bbox[3]]
        obj_seg_interior = seg_interior[skimage_bbox[0]:skimage_bbox[2], skimage_bbox[1]:skimage_bbox[3]]
        score = float(np.mean(obj_seg_interior * obj_mask))

        coco_bbox = [skimage_bbox[1], skimage_bbox[0],
                     skimage_bbox[3] - skimage_bbox[1], skimage_bbox[2] - skimage_bbox[0]]

        image_mask = labels == (i + 1)  # The mask has to span the whole image
        rle = pycocotools.mask.encode(np.asfortranarray(image_mask))
        rle["counts"] = rle["counts"].decode("utf-8")
        image_id = sample["image_id"].item()
        annotation = {
            "category_id": 100,  # Building
            "bbox": coco_bbox,
            "segmentation": rle,
            "score": score,
            "image_id": image_id}
        annotations.append(annotation)
    return annotations


def save_seg_coco(sample, base_filepath, name):
    filepath = get_save_filepath(base_filepath, name, ".json")
    annotations = seg_coco(sample)
    python_utils.save_json(filepath, annotations)


def save_crossfield(crossfield, base_filepath, name):
    # TODO: optimize crossfield disk space
    # Save raw crossfield
    crossfield = np.transpose(crossfield.numpy(), (1, 2, 0))
    crossfield_filepath =get_save_filepath(base_filepath, name, ".npy")
    np.save(crossfield_filepath, crossfield)


def save_uv_angles(crossfield, base_filepath, name, image_filepath):
    crossfield = np.transpose(crossfield.numpy(), (1, 2, 0))
    u, v = math_utils.compute_crossfield_uv(crossfield)  # u, v are complex arrays
    u_angle, v_angle = np.angle(u), np.angle(v)
    u_angle, v_angle = np.mod(u_angle, np.pi), np.mod(v_angle, np.pi)
    uv_angles = np.stack([u_angle, v_angle], axis=-1)
    uv_angles_as_image = np.round(uv_angles * 255 / np.pi).astype(np.uint8)
    save_filepath = get_save_filepath(base_filepath, name, ".tif")
    geo_utils.save_image_as_geotiff(save_filepath, uv_angles_as_image, image_filepath)


def save_raw_pred(sample, base_filepath, name):
    save_filepath =get_save_filepath(base_filepath, name, ".pt")
    torch.save(sample, save_filepath)


def save_opencities_mask(seg_mask, base_filepath, name, image_filepath):
    seg_mask = seg_mask.numpy()
    out_filepath = get_save_filepath(base_filepath, name, ".tif")
    tifffile.imwrite(out_filepath, np.logical_not((np.array(seg_mask).astype(np.bool))))
