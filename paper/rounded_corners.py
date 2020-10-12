import numpy as np
import matplotlib.pyplot as plt

import shapely.geometry

import torch_lydorn.torchvision
from numpy.core._multiarray_umath import ndarray
from scipy import ndimage

import vectorization_ambiguities


def create_polygons():
    polygons = [
        np.array([
            [25, 25],
            [75, 25],
            [75, 50],
            [50, 50],
            [50, 75],
            [25, 75],
            [25, 25],
        ])
    ]
    return polygons


def displace_polygons(polygons, max_global, max_polygon, max_vertex, max_rot_deg):
    new_polygons = []
    global_disp = np.random.uniform(-1, 1, 2) * max_global
    for polygon in polygons:
        polygon_disp = np.random.uniform(-1, 1, 2) * max_polygon
        vertex_disp = np.random.uniform(-1, 1, polygon.shape) * max_vertex
        new_polygon = polygon + global_disp + polygon_disp + vertex_disp

        # Rotation
        geom = shapely.geometry.Polygon(new_polygon)
        angle = np.random.uniform(-1, 1, 1) * max_rot_deg
        geom = shapely.affinity.rotate(geom, angle, origin='center', use_radians=False)
        new_polygon = geom.exterior.coords[:]

        new_polygons.append(new_polygon)
    return new_polygons


def rasterize(image, polygons):
    polygons = [shapely.geometry.Polygon(polygon) for polygon in polygons]
    raster = torch_lydorn.torchvision.transforms.Rasterize(fill=True, edges=False, vertices=False, line_width=4, antialiasing=True)(image, polygons)
    raster = raster[:, :, 0]
    raster = ndimage.gaussian_filter(raster, sigma=1)  # Simulates blurriness of overhead image, which leads to blurriness of segmentation
    return raster


def plot(image, out_filepath, dpi=300):
    height = image.shape[0]
    width = image.shape[1]
    f, axis = plt.subplots(1, 1, figsize=(width, height), dpi=dpi)

    # Plot image
    axis.imshow(image, cmap="gray")

    axis.autoscale(False)
    axis.axis('equal')
    axis.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Plot without margins
    plt.savefig(out_filepath, transparent=True, dpi=dpi)
    plt.close()


def main():
    shape = (100, 100)
    samples = 10
    all_rasters: ndarray = np.empty((*shape, samples))
    methods = [
        "marching_squares",
        "border_following",
        "rasterio"
    ]

    for s in range(samples):
        polygons = create_polygons()
        polygons = displace_polygons(polygons, max_global=0, max_polygon=3, max_vertex=0.5, max_rot_deg=1)  # Simulates imperfect ground truth annotations
        raster = rasterize(np.zeros(shape), polygons) / 255
        all_rasters[:, :, s] = raster
        plot(raster, f"rounded_corners_sample_{s:02d}.png", dpi=1)

        mean_raster_s = np.mean(all_rasters[:, :, :(s+1)], axis=-1)  # Simulates training to reduce average loss over all (noisy) ground truth
        plot(mean_raster_s, f"rounded_corners_avg_{s:02d}.png", dpi=1)

    for m in methods:
        contours = vectorization_ambiguities.detect_contours(mean_raster_s, method=m)
        vectorization_ambiguities.plot(mean_raster_s, contours, f"rounded_corners_avg_contour_{m}.pdf", linewidth=60, dpi=1, grid=False)


if __name__ == "__main__":
    main()
