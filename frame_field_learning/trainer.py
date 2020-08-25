import os

import torch_lydorn.torchvision
from tqdm import tqdm

import torch
import torch.distributed

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from torch.utils.tensorboard import SummaryWriter

# from pytorch_memlab import profile, profile_every

from . import measures, plot_utils
from . import local_utils

from lydorn_utils import run_utils
from lydorn_utils import python_utils
from lydorn_utils import math_utils

try:
    from apex import amp

    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False


def humanbytes(B):
    'Return the given bytes as a human friendly KB, MB, GB, or TB string'
    B = float(B)
    KB = float(1024)
    MB = float(KB ** 2)  # 1,048,576
    GB = float(KB ** 3)  # 1,073,741,824
    TB = float(KB ** 4)  # 1,099,511,627,776

    if B < KB:
        return '{0} {1}'.format(B, 'Bytes' if 0 == B > 1 else 'Byte')
    elif KB <= B < MB:
        return '{0:.2f} KB'.format(B / KB)
    elif MB <= B < GB:
        return '{0:.2f} MB'.format(B / MB)
    elif GB <= B < TB:
        return '{0:.2f} GB'.format(B / GB)
    elif TB <= B:
        return '{0:.2f} TB'.format(B / TB)


class Trainer:
    def __init__(self, rank, gpu, config, model, optimizer, loss_func,
                 run_dirpath, init_checkpoints_dirpath=None, lr_scheduler=None):
        self.rank = rank
        self.gpu = gpu
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.loss_func = loss_func

        self.init_checkpoints_dirpath = init_checkpoints_dirpath
        logs_dirpath = run_utils.setup_run_subdir(run_dirpath, config["optim_params"]["logs_dirname"])
        self.checkpoints_dirpath = run_utils.setup_run_subdir(run_dirpath, config["optim_params"]["checkpoints_dirname"])
        if self.rank == 0:
            self.logs_dirpath = logs_dirpath
            train_logs_dirpath = os.path.join(self.logs_dirpath, "train")
            val_logs_dirpath = os.path.join(self.logs_dirpath, "val")
            self.train_writer = SummaryWriter(train_logs_dirpath)
            self.val_writer = SummaryWriter(val_logs_dirpath)
        else:
            self.logs_dirpath = self.train_writer = self.val_writer = None

    def log_weights(self, module, module_name, step):
        weight_list = module.parameters()
        for i, weight in enumerate(weight_list):
            if len(weight.shape) == 4:
                weight_type = "4d"
            elif len(weight.shape) == 1:
                weight_type = "1d"
            elif len(weight.shape) == 2:
                weight_type = "2d"
            else:
                weight_type = ""
            self.train_writer.add_histogram('{}/{}/{}/hist'.format(module_name, i, weight_type), weight, step)
            # self.writer.add_scalar('{}/{}/mean'.format(module_name, i), mean, step)
            # self.writer.add_scalar('{}/{}/max'.format(module_name, i), maxi, step)

    # def log_pr_curve(self, name, pred, batch, iter_step):
    #     num_thresholds = 100
    #     thresholds = torch.linspace(0, 2 * self.config["max_disp_global"] + self.config["max_disp_poly"], steps=num_thresholds)
    #     dists = measures.pos_dists(pred, batch).cpu()
    #     tiled_dists = dists.repeat(num_thresholds, 1)
    #     tiled_thresholds = thresholds.repeat(dists.shape[0], 1).t()
    #     true_positives = tiled_dists < tiled_thresholds
    #     true_positive_counts = torch.sum(true_positives, dim=1)
    #     recall = true_positive_counts.float() / true_positives.shape[1]
    #
    #     precision = 1 - thresholds / (2 * self.config["max_disp_global"] + self.config["max_disp_poly"])
    #
    #     false_positive_counts = true_positives.shape[1] - true_positive_counts
    #     true_negative_counts = torch.zeros(num_thresholds)
    #     false_negative_counts = torch.zeros(num_thresholds)
    #     self.writer.add_pr_curve_raw(name, true_positive_counts,
    #                                  false_positive_counts,
    #                                  true_negative_counts,
    #                                  false_negative_counts,
    #                                  precision,
    #                                  recall,
    #                                  global_step=iter_step,
    #                                  num_thresholds=num_thresholds)

    def sync_outputs(self, loss, individual_metrics_dict):
        # Reduce to rank 0:
        torch.distributed.reduce(loss, dst=0)
        for key in individual_metrics_dict.keys():
            torch.distributed.reduce(individual_metrics_dict[key], dst=0)
        # Average on rank 0:
        if self.rank == 0:
            loss /= self.config["world_size"]
            for key in individual_metrics_dict.keys():
                individual_metrics_dict[key] /= self.config["world_size"]

    # from pytorch_memlab import profile
    # @profile
    def loss_batch(self, batch, opt=None, epoch=None):
        # print("Forward pass:")
        # t0 = time.time()
        pred, batch = self.model(batch)
        # print(f"{time.time() - t0}s")

        # print("Loss computation:")
        # t0 = time.time()
        loss, individual_metrics_dict, extra_dict = self.loss_func(pred, batch, epoch=epoch)
        # print(f"{time.time() - t0}s")

        # Compute IoUs at different thresholds
        if "seg" in pred:
            y_pred = pred["seg"][:, 0, ...]
            y_true = batch["gt_polygons_image"][:, 0, ...]
            iou_thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]
            for iou_threshold in iou_thresholds:
                iou = measures.iou(y_pred.reshape(y_pred.shape[0], -1), y_true.reshape(y_true.shape[0], -1), threshold=iou_threshold)
                mean_iou = torch.mean(iou)
                individual_metrics_dict[f"IoU_{iou_threshold}"] = mean_iou

        # print("Backward pass:")
        # t0 = time.time()
        if opt is not None:
            # Detect if loss is nan
            # contains_nan = bool(torch.sum(torch.isnan(loss)).item())
            # if contains_nan:
            #     raise ValueError("NaN values detected, aborting...")
            if self.config["use_amp"] and APEX_AVAILABLE:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # all_grads = []
            # for param in self.model.parameters():
            #     # print("shape: {}".format(param.shape))
            #     if param.grad is not None:
            #         all_grads.append(param.grad.view(-1))
            # all_grads = torch.cat(all_grads)
            # all_grads_abs = torch.abs(all_grads)

            opt.step()
            opt.zero_grad()
        # print(f"{time.time() - t0}s")

        # Synchronize losses/accuracies to GPU 0 so that they can be logged
        self.sync_outputs(loss, individual_metrics_dict)

        for key in individual_metrics_dict:
            individual_metrics_dict[key] = individual_metrics_dict[key].item()

        # Log IoU if exists
        log_iou = None
        iou_name = f"IoU_{0.5}"  # Progress bars will show that IoU and it will be saved in checkpoints
        if iou_name in individual_metrics_dict:
            log_iou = individual_metrics_dict[iou_name]

        return pred, batch, loss.item(), individual_metrics_dict, extra_dict, log_iou, batch["image"].shape[0]

    def run_epoch(self, split_name, dl, epoch, log_steps=None, opt=None, iter_step=None):
        assert split_name in ["train", "val"]
        if split_name == "train":
            writer = self.train_writer
        elif split_name == "val":
            writer = self.val_writer
            assert iter_step is not None
        else:
            writer = None

        running_loss_meter = math_utils.AverageMeter("running_loss")
        running_losses_meter_dict = {loss_func.name: math_utils.AverageMeter(loss_func.name) for loss_func in
                                     self.loss_func.loss_funcs}
        total_running_loss_meter = math_utils.AverageMeter("total_running_loss")
        running_iou_meter = math_utils.AverageMeter("running_iou")
        total_running_iou_meter = math_utils.AverageMeter("total_running_iou")

        # batch_index_offset = 0
        epoch_iterator = dl
        if self.gpu == 0:
            epoch_iterator = tqdm(epoch_iterator, desc="{}: ".format(split_name), leave=False)
        for i, batch in enumerate(epoch_iterator):
            # Send batch to device
            batch = local_utils.batch_to_cuda(batch)

            # with torch.autograd.detect_anomaly():  # TODO: comment when not debugging
            pred, batch, total_loss, metrics_dict, loss_extra_dict, log_iou, nums = self.loss_batch(batch, opt=opt, epoch=epoch)
            # with torch.autograd.profiler.profile(use_cuda=True) as prof:
            #     loss, nums = self.loss_batch(batch, opt=opt)
            # print(prof.key_averages().table(sort_by="cuda_time_total"))

            running_loss_meter.update(total_loss, nums)
            for name, loss in metrics_dict.items():
                if name not in running_losses_meter_dict:  # Init
                    running_losses_meter_dict[name] = math_utils.AverageMeter(name)
                running_losses_meter_dict[name].update(loss, nums)
            total_running_loss_meter.update(total_loss, nums)
            if log_iou is not None:
                running_iou_meter.update(log_iou, nums)
                total_running_iou_meter.update(log_iou, nums)

            # Log values
            # batch_index = i + batch_index_offset
            if split_name == "train":
                iter_step = epoch * len(epoch_iterator) + i
            if split_name == "train" and (iter_step % log_steps == 0) or \
                    split_name == "val" and i == (len(epoch_iterator) - 1):
                # if iter_step % log_steps == 0:
                if self.gpu == 0:
                    epoch_iterator.set_postfix(loss="{:.4f}".format(running_loss_meter.get_avg()),
                                               iou="{:.4f}".format(running_iou_meter.get_avg()))

                # Logs
                if self.rank == 0:
                    writer.add_scalar("Metrics/Loss", running_loss_meter.get_avg(), iter_step)
                    for key, meter in running_losses_meter_dict.items():
                        writer.add_scalar(f"Metrics/{key}", meter.get_avg(), iter_step)

                    image_display = torch_lydorn.torchvision.transforms.functional.batch_denormalize(batch["image"],
                                                                                                     batch[
                                                                                                         "image_mean"],
                                                                                                     batch["image_std"])
                    # # Save image overlaid with gt_seg to tensorboard:
                    # image_gt_seg_display = plot_utils.get_tensorboard_image_seg_display(image_display, batch["gt_polygons_image"])
                    # writer.add_images('gt_seg', image_gt_seg_display, iter_step)

                    # Save image overlaid with seg to tensorboard:
                    if "seg" in pred:
                        crossfield = pred["crossfield"] if "crossfield" in pred else None
                        image_seg_display = plot_utils.get_tensorboard_image_seg_display(image_display, pred["seg"], crossfield=crossfield)
                        writer.add_images('seg', image_seg_display, iter_step)

                    # self.log_pr_curve("PR curve/{}".format(name), pred, batch, iter_step)

                    # self.log_weights(self.model.module.backbone, "backbone", iter_step)
                    # if hasattr(self.model.module, "seg_module"):
                    #     self.log_weights(self.model.module.seg_module, "seg_module", iter_step)
                    # if hasattr(self.model.module, "crossfield_module"):
                    #     self.log_weights(self.model.module.crossfield_module, "crossfield_module", iter_step)

                    # self.writer.flush()
                    # im = batch["image"][0]
                    # self.writer.add_image('image', im)
                running_loss_meter.reset()
                for key, meter in running_losses_meter_dict.items():
                    meter.reset()
                running_iou_meter.reset()

        return total_running_loss_meter.get_avg(), total_running_iou_meter.get_avg(), iter_step

    def compute_loss_norms(self, dl, total_batches):
        self.loss_func.reset_norm()

        t = None
        if self.gpu == 0:
            t = tqdm(total=total_batches, desc="Init loss norms", leave=True)  # Initialise

        batch_i = 0
        while batch_i < total_batches:
            for batch in dl:
                # Update loss norms
                batch = local_utils.batch_to_cuda(batch)
                pred, batch = self.model(batch)
                self.loss_func.update_norm(pred, batch, batch["image"].shape[0])
                if t is not None:
                    t.update(1)
                batch_i += 1
                if not batch_i < total_batches:
                    break

        # Now sync loss norms across GPUs:
        self.loss_func.sync(self.config["world_size"])

    def fit(self, train_dl, val_dl=None, init_dl=None):
        # Try loading previous model
        checkpoint = self.load_checkpoint(self.checkpoints_dirpath)  # Try last checkpoint
        if checkpoint is None and self.init_checkpoints_dirpath is not None:
            # Try with init_checkpoints_dirpath:
            checkpoint = self.load_checkpoint(self.init_checkpoints_dirpath)
            checkpoint["epoch"] = 0  # Re-start from 0
        if checkpoint is None:
            checkpoint = {
                "epoch": 0,
            }
            if init_dl is not None:
                # --- Compute norms of losses on several epochs:
                self.model.train()  # Important for batchnorm and dropout, even in computing loss norms
                with torch.no_grad():
                    loss_norm_batches_min = self.config["loss_params"]["multiloss"]["normalization_params"]["min_samples"] // (2 * self.config["optim_params"]["batch_size"]) + 1
                    loss_norm_batches_max = self.config["loss_params"]["multiloss"]["normalization_params"]["max_samples"] // (2 * self.config["optim_params"]["batch_size"]) + 1
                    loss_norm_batches = max(loss_norm_batches_min, min(loss_norm_batches_max, len(init_dl)))
                    self.compute_loss_norms(init_dl, loss_norm_batches)

        if self.gpu == 0:
            # Prints loss norms:
            print(self.loss_func)

        start_epoch = checkpoint["epoch"]  # Start at next epoch

        fit_iterator = range(start_epoch, self.config["optim_params"]["max_epoch"])
        if self.gpu == 0:
            fit_iterator = tqdm(fit_iterator, desc="Fitting: ", initial=start_epoch,
                                total=self.config["optim_params"]["max_epoch"])

        train_loss = None
        val_loss = None
        train_iou = None
        epoch = None
        for epoch in fit_iterator:

            self.model.train()
            train_loss, train_iou, iter_step = self.run_epoch("train", train_dl, epoch, self.config["optim_params"]["log_steps"],
                                                              opt=self.optimizer)

            if val_dl is not None:
                self.model.eval()
                with torch.no_grad():
                    val_loss, val_iou, _ = self.run_epoch("val", val_dl, epoch, self.config["optim_params"]["log_steps"], iter_step=iter_step)
            else:
                val_loss = None
                val_iou = None

            if val_loss is not None:
                self.lr_scheduler.step()
            else:
                self.lr_scheduler.step()

            if self.gpu == 0:
                postfix_args = {"t_loss": "{:.4f}".format(train_loss), "t_iou": "{:.4f}".format(train_iou)}
                if val_loss is not None:
                    postfix_args["v_loss"] = "{:.4f}".format(val_loss)
                if val_loss is not None:
                    postfix_args["v_iou"] = "{:.4f}".format(val_iou)
                fit_iterator.set_postfix(**postfix_args)
            if self.rank == 0:
                if (epoch + 1) % self.config["optim_params"]["checkpoint_epoch"] == 0:
                    self.save_last_checkpoint(epoch + 1, train_loss, val_loss, train_iou,
                                              val_iou)  # Save the last completed epoch, hence the "+1"
                    self.delete_old_checkpoint(epoch + 1)
                if val_loss is not None:
                    self.save_best_val_checkpoint(epoch + 1, train_loss, val_loss, train_iou, val_iou)
        if self.rank == 0 and epoch is not None:
            self.save_last_checkpoint(epoch + 1, train_loss, val_loss, train_iou,
                                      val_iou)  # Save the last completed epoch, hence the "+1"

    def load_checkpoint(self, checkpoints_dirpath):
        """
        Loads last checkpoint in checkpoints_dirpath
        :param checkpoints_dirpath:
        :return:
        """
        try:
            filepaths = python_utils.get_filepaths(checkpoints_dirpath, endswith_str=".tar",
                                                   startswith_str="checkpoint.")
            if len(filepaths) == 0:
                return None

            filepaths = sorted(filepaths)
            filepath = filepaths[-1]  # Last checkpoint

            checkpoint = torch.load(filepath, map_location="cuda:{}".format(
                self.gpu))  # map_location is used to load on current device

            self.model.module.load_state_dict(checkpoint['model_state_dict'])

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            self.loss_func.load_state_dict(checkpoint['loss_func_state_dict'])
            epoch = checkpoint['epoch']

            return {
                "epoch": epoch,
            }
        except NotADirectoryError:
            return None

    def save_checkpoint(self, filepath, epoch, train_loss, val_loss, train_acc, val_acc):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict(),  # model is a DistributedDataParallel module
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'loss_func_state_dict': self.loss_func.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
        }, filepath)

    def save_last_checkpoint(self, epoch, train_loss, val_loss, train_acc, val_acc):
        filename_format = "checkpoint.epoch_{:06d}.tar"
        filepath = os.path.join(self.checkpoints_dirpath, filename_format.format(epoch))
        self.save_checkpoint(filepath, epoch, train_loss, val_loss, train_acc, val_acc)

    def delete_old_checkpoint(self, current_epoch):
        filename_format = "checkpoint.epoch_{:06d}.tar"
        to_delete_epoch = current_epoch - self.config["optim_params"]["checkpoints_to_keep"] * self.config["optim_params"]["checkpoint_epoch"]
        filepath = os.path.join(self.checkpoints_dirpath, filename_format.format(to_delete_epoch))
        if os.path.exists(filepath):
            os.remove(filepath)

    def save_best_val_checkpoint(self, epoch, train_loss, val_loss, train_acc, val_acc):
        filepath = os.path.join(self.checkpoints_dirpath, "checkpoint.best_val.epoch_{:06d}.tar".format(epoch))

        # Search for a prev best val checkpoint:
        prev_filepaths = python_utils.get_filepaths(self.checkpoints_dirpath, startswith_str="checkpoint.best_val.",
                                                    endswith_str=".tar")

        if len(prev_filepaths):
            prev_filepaths = sorted(prev_filepaths)
            prev_filepath = prev_filepaths[-1]  # Last best val checkpoint filepath in case there is more than one

            prev_best_val_checkpoint = torch.load(prev_filepath)
            prev_best_loss = prev_best_val_checkpoint["val_loss"]
            if val_loss < prev_best_loss:
                self.save_checkpoint(filepath, epoch, train_loss, val_loss, train_acc, val_acc)
                # Delete prev best val
                [os.remove(prev_filepath) for prev_filepath in prev_filepaths]
        else:
            self.save_checkpoint(filepath, epoch, train_loss, val_loss, train_acc, val_acc)
