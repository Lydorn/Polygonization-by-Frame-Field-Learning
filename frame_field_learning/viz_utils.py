import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry

from frame_field_learning import plot_utils


def save_poly_viz(image, polygons, out_filepath, linewidths=2, markersize=20, alpha=0.2, draw_vertices=True, corners=None, crossfield=None, polygon_probs=None, seg=None, color_choices=None, dpi=10):
    assert isinstance(polygons, list), f"polygons should be of type list, not {type(polygons)}"
    if len(polygons):
        assert (type(polygons[0]) == np.ndarray or type(polygons[0]) == shapely.geometry.Polygon), f"Item of the polygons list should be of type ndarray or shapely Polygon, not {type(polygons[0])}"
    if polygon_probs is not None:
        assert type(polygon_probs) == list
        assert len(polygons) == len(polygon_probs), "len(polygons)={} should be equal to len(polygon_probs)={}".format(len(polygons), len(polygon_probs))
    # Setup plot
    height = image.shape[0]
    width = image.shape[1]
    f, axis = plt.subplots(1, 1, figsize=(width / 10, height / 10), dpi=10)

    axis.imshow(image)

    if seg is not None:
        seg *= 0.9
        axis.imshow(seg)

    if crossfield is not None:
        plot_utils.plot_crossfield(axis, crossfield, crossfield_stride=1, alpha=0.5, width=0.1, add_scale=1.1, invert_y=False)

    plot_utils.plot_polygons(axis, polygons, polygon_probs=polygon_probs, draw_vertices=draw_vertices, linewidths=linewidths, markersize=markersize, alpha=alpha, color_choices=color_choices)

    if corners is not None and len(corners):
        assert len(corners[0].shape) == 2
        for corner_array in corners:
            plt.plot(corner_array[:, 0], corner_array[:, 1], marker="o", linewidth=0, markersize=20, color="red")

    axis.autoscale(False)
    axis.axis('equal')
    axis.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Plot without margins
    plt.savefig(out_filepath, transparent=True, dpi=dpi)
    plt.close()
