from . import polygonize_utils
from . import polygonize_acm
from . import polygonize_simple

from lydorn_utils import print_utils


class Polygonizer():
    def __init__(self, polygonize_params, pool=None):
        self.pool = pool
        self.polygonizer_asm = None

    def __call__(self, polygonize_params, seg_batch, crossfield_batch=None, pre_computed=None):
        """

        :param polygonize_params:
        :param seg_batch: (N, C, H, W)
        :param crossfield_batch: (N, 4, H, W)
        :param pre_computed: None o a Dictionary of pre-computed values used for various methods
        :return:
        """
        assert len(seg_batch.shape) == 4, "seg_batch should be (N, C, H, W)"
        assert pre_computed is None or isinstance(pre_computed, dict), "pre_computed should be either None or a dict"
        batch_size = seg_batch.shape[0]

        # Check if polygonize_params["method"] is a list or a string:
        if type(polygonize_params["method"]) == list:
            # --- For speed up, pre-compute anything that is used by multiple methods:
            if pre_computed is None:
                pre_computed = {}
            if ("simple" in polygonize_params["method"] or "acm" in polygonize_params["method"]) and "init_contours_batch" not in pre_computed:
                indicator_batch = seg_batch[:, 0, :, :]
                np_indicator_batch = indicator_batch.cpu().numpy()
                init_contours_batch = polygonize_utils.compute_init_contours_batch(np_indicator_batch,
                                                                                   polygonize_params["common_params"][
                                                                                       "init_data_level"],
                                                                                   pool=self.pool)
                pre_computed["init_contours_batch"] = init_contours_batch
            # ---
            # Run one method after the other:
            out_polygons_dict_batch = [{} for _ in range(batch_size)]
            out_probs_dict_batch = [{} for _ in range(batch_size)]
            for method_name in polygonize_params["method"]:
                new_polygonize_params = polygonize_params.copy()
                new_polygonize_params["method"] = method_name
                polygons_batch, probs_batch = self(new_polygonize_params, seg_batch,
                                                   crossfield_batch=crossfield_batch, pre_computed=pre_computed)
                if polygons_batch is not None:
                    for i, (polygons, probs) in enumerate(zip(polygons_batch, probs_batch)):
                        out_polygons_dict_batch[i][method_name] = polygons
                        out_probs_dict_batch[i][method_name] = probs
            return out_polygons_dict_batch, out_probs_dict_batch

        # --- Else: run the one method
        if polygonize_params["method"] == "acm":
            if crossfield_batch is None:
                # Cannot run the ACM method
                return None, None
            polygons_batch, probs_batch = polygonize_acm.polygonize(seg_batch, crossfield_batch,
                                                                    polygonize_params["acm_method"], pool=self.pool,
                                                                    pre_computed=pre_computed)
        elif polygonize_params["method"] == "asm":
            from . import polygonize_asm
            if crossfield_batch is None:
                # Cannot run the ASM method
                return None, None
            if self.polygonizer_asm is None:
                self.polygonizer_asm = polygonize_asm.PolygonizerASM(polygonize_params["asm_method"], pool=self.pool)
            polygons_batch, probs_batch = self.polygonizer_asm(seg_batch, crossfield_batch, pre_computed=pre_computed)
        elif polygonize_params["method"] == "simple":
            polygons_batch, probs_batch = polygonize_simple.polygonize(seg_batch, polygonize_params["simple_method"],
                                                                       pool=self.pool, pre_computed=pre_computed)
        else:
            print_utils.print_error("ERROR: polygonize method {} not recognized!".format(polygonize_params["method"]))
            raise NotImplementedError

        return polygons_batch, probs_batch


def polygonize(polygonize_params, seg_batch, crossfield_batch=None, pool=None, pre_computed=None):
    polygonizer = Polygonizer(polygonize_params, pool=pool)
    return polygonizer(polygonize_params, seg_batch, crossfield_batch=crossfield_batch, pre_computed=pre_computed)
