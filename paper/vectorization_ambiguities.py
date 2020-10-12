import numpy as np
import matplotlib.pyplot as plt
import skimage.measure
import cv2


def create_seg():
    seg = np.zeros((6, 8))
    # Triangle:
    seg[1, 4] = 1
    seg[2, 3:5] = 1
    seg[3, 2:5] = 1
    seg[4, 1:5] = 1
    # L extension:
    seg[3:5, 5:7] = 1
    return seg


def detect_contours(seg, method):
    if method == "marching_squares":
        contours = skimage.measure.find_contours(seg, 0.5, fully_connected='low', positive_orientation='high')
    elif method == "border_following":
        u, contours, _ = cv2.findContours((0.5 < seg).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = [contour[:, 0, ::-1] for contour in contours]
        contours = [np.concatenate((contour, contour[0:1, :]), axis=0) for contour in contours]
    elif method == "rasterio":
        import rasterio.features
        shapes = rasterio.features.shapes((0.5 < seg).astype(np.uint8))
        contours = []
        for shape in shapes:
            for coords in shape[0]["coordinates"][1:]:
                contours.append(np.array(coords)[:, ::-1] - 0.5)
    else:
        raise ValueError(f"Method {method} not recognized!")
    return contours


def plot(image, contours, out_filepath, linewidth=6, dpi=300, grid=True):
    height = image.shape[0]
    width = image.shape[1]
    f, axis = plt.subplots(1, 1, figsize=(width, height), dpi=dpi)

    # Plot image
    axis.imshow(image, cmap="gray")

    # Grid lines
    if grid:
        for p in range(image.shape[1]):
            plt.axvline(p + 0.5, color=[0.5]*3, linewidth=0.5)
        for p in range(image.shape[0]):
            plt.axhline(p + 0.5, color=[0.5]*3, linewidth=0.5)

    # Plot contours
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=linewidth)

    axis.autoscale(False)
    axis.axis('equal')
    axis.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Plot without margins
    plt.savefig(out_filepath, transparent=True, dpi=dpi)
    plt.close()


def main():
    seg = create_seg()

    methods = [
        "marching_squares",
        "border_following",
        "rasterio"
    ]

    for m in methods:
        contours = detect_contours(seg, method=m)
        plot(seg, contours, f"vectorization_ambiguities_{m}.pdf", dpi=300)


if __name__ == "__main__":
    main()
