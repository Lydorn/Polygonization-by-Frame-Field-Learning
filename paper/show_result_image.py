import matplotlib.pyplot as plt

from pycocotools.coco import COCO
import skimage.io as io
import os


def plot_result(output_filename_format, im, image_id, coco):
    print("Plotting image" + output_filename_format.format(image_id))
    annotation_ids = coco.getAnnIds(imgIds=image_id)
    annotations = coco.loadAnns(annotation_ids)
    dpi = 100
    f, axis = plt.subplots(1, 1, figsize=(im.shape[1] / dpi, im.shape[0] / dpi), dpi=dpi)
    axis.imshow(im)
    coco.showAnns(annotations)
    axis.autoscale(False)
    axis.axis('equal')
    axis.axis('off')
    axis.set_xlim([0, im.shape[1]])
    axis.set_ylim([im.shape[0], 0])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Plot without margins
    plt.savefig(output_filename_format.format(image_id))
    plt.show()


def show_result_image(val_annotations_filepath, val_images_dirpath, info_list, image_id_list):
    coco_gt = COCO(val_annotations_filepath)
    image_ids = coco_gt.getImgIds(catIds=coco_gt.getCatIds())

    for info in info_list:
        if info["annotations_filepath"] == "gt_annotations":
            coco_dt = coco_gt
        else:
            coco_dt = coco_gt.loadRes(info["annotations_filepath"])

        for image_id in image_id_list:
            assert image_id in image_ids, "image_id is invalid"
            img_info = coco_gt.loadImgs(image_id)[0]
            image_path = os.path.join(val_images_dirpath, img_info["file_name"])
            im = io.imread(image_path)

            plot_result(info["output_filename_format"], im, image_id, coco_dt)


def main():
    val_annotations_filepath = "/data/data/mapping_challenge_dataset/raw/val/annotation.json"
    val_images_dirpath = "/data/data/mapping_challenge_dataset/raw/val/images"

    # Mapping challenge:
    info_list = [
        {
            "output_filename_format": "crowdai_gt_{:012d}.pdf",
            "annotations_filepath": "gt_annotations"
        },
        # {
        #     "output_filename_format": "crowdai_unet_resnet101_{:012d}.poly_viz.acm.tol_1.pdf",
        #     "annotations_filepath": "/data/data/mapping_challenge_dataset/eval_runs/mapping_dataset.unet_resnet101_pretrained.train_val | 2020-05-21 08:32:48/test.annotation.poly.acm.tol_1.json"
        # },
        # {
        #     "output_filename_format": "crowdai_polymapper_{:012d}.poly_viz.pdf",
        #     "annotations_filepath": "/data/data/mapping_challenge_dataset/eval_runs/mapping_dataset.polymapper | 0000-00-00 00:00:00/test.annotation.poly.json"
        # },
        # {
        #     "output_filename_format": "crowdai_li_{:012d}.poly_viz.pdf",
        #     "annotations_filepath": "/data/data/mapping_challenge_dataset/eval_runs/mapping_dataset.mu | 0000-00-00 00:00:00/test.annotation.poly.json"
        # },

        {
            "output_filename_format": "crowdai_unet_resnet101_full_{:012d}.poly_viz.acm.tol_0.125.pdf",
            "annotations_filepath": "/data/data/mapping_challenge_dataset/eval_runs/mapping_dataset.unet_resnet101_pretrained.train_val | 2020-05-21 08:32:48/test.annotation.poly.acm.tol_0.125.json"
        },
        {
            "output_filename_format": "crowdai_unet_resnet101_full_{:012d}.poly_viz.simple.tol_0.125.pdf",
            "annotations_filepath": "/data/data/mapping_challenge_dataset/eval_runs/mapping_dataset.unet_resnet101_pretrained.train_val | 2020-05-21 08:32:48/test.annotation.poly.simple.tol_0.125.json"
        },
        {
            "output_filename_format": "crowdai_unet_resnet101_no_field_{:012d}.poly_viz.simple.tol_0.125.pdf",
            "annotations_filepath": "/data/data/mapping_challenge_dataset/eval_runs/mapping_dataset.unet_resnet101_pretrained.field_off.train_val | 2020-05-21 08:33:20/test.annotation.poly.simple.tol_0.125.json"
        },
        {
            "output_filename_format": "crowdai_unet16_no_coupling_losses_{:012d}.poly_viz.simple.tol_0.125.pdf",
            "annotations_filepath": "/data/data/mapping_challenge_dataset/eval_runs/mapping_dataset.unet16.coupling_losses_off.train_val | 2020-03-01 13:27:45/test.annotation.poly.simple.tol_0.125.json"
        },
        {
            "output_filename_format": "crowdai_unet16_no_coupling_losses_{:012d}.poly_viz.acm.tol_0.125.pdf",
            "annotations_filepath": "/data/data/mapping_challenge_dataset/eval_runs/mapping_dataset.unet16.coupling_losses_off.train_val | 2020-03-01 13:27:45/test.annotation.poly.acm.tol_0.125.json"
        },
    ]
    image_id_list = [21219, 443, 371, 265, 18205, 1, 2, 3, 4]
    # image_id_list = [443]

    show_result_image(val_annotations_filepath, val_images_dirpath, info_list, image_id_list)


if __name__ == '__main__':
    main()
