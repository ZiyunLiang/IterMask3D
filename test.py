import torch
import random
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from argumentlib import args
from itertools import chain
from Nets.unet3d import UNET3D
from scipy.optimize import curve_fit
from monai.metrics import DiceMetric, ConfusionMatrixMetric, SurfaceDistanceMetric, MeanIoU
from dataloader import get_data_loader_test
from sklearn.metrics import precision_recall_curve, auc, roc_curve, average_precision_score, roc_auc_score



def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def tangent_model_shifted(t, a, b):
    return a * np.tan(b * t)

def tangent_model_derivative(t, a, b):
    return a * b * (1 / np.cos(b * t)) ** 2

def normalize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img

def sensitivity(pred: torch.Tensor, target: torch.Tensor, epsilon=1e-6) -> torch.Tensor:
    """
    Computes sensitivity (recall) for 3D binary volumes.

    Args:
        pred (torch.Tensor): Binary predicted mask [N, 1, W, H, D]
        target (torch.Tensor): Binary ground truth mask [N, 1, W, H, D]

    Returns:
        torch.Tensor: Sensitivity per volume [N]
    """
    TP = (pred * target).sum(dim=(2, 3, 4))  # shape [N, 1]
    FN = ((1 - pred) * target).sum(dim=(2, 3, 4))
    return ((TP + epsilon) / (TP + FN + epsilon)).squeeze(1)  # shape [N]


def precision(pred: torch.Tensor, target: torch.Tensor, epsilon=1e-6) -> torch.Tensor:
    """
    Computes precision for 3D binary volumes.

    Args:
        pred (torch.Tensor): Binary predicted mask [N, 1, W, H, D]
        target (torch.Tensor): Binary ground truth mask [N, 1, W, H, D]

    Returns:
        torch.Tensor: Precision per volume [N]
    """
    TP = (pred * target).sum(dim=(2, 3, 4))
    FP = (pred * (1 - target)).sum(dim=(2, 3, 4))
    return ((TP + epsilon) / (TP + FP + epsilon)).squeeze(1)


def psnr(recon: torch.Tensor, original: torch.Tensor, mask: torch.Tensor, max_val=1.0, epsilon=1e-8) -> torch.Tensor:
    """
    Computes PSNR over a masked region (e.g., healthy area) for 3D volumes.

    Args:
        recon (torch.Tensor): Reconstructed image, shape [N, 1, W, H, D]
        original (torch.Tensor): Ground truth image, shape [N, 1, W, H, D]
        mask (torch.Tensor): Binary mask (1 = include voxel), shape [N, 1, W, H, D]
        max_val (float): Maximum possible voxel intensity (e.g., 1.0 if normalized)
        epsilon (float): Small constant to avoid division by zero

    Returns:
        torch.Tensor: PSNR per volume [N]
    """
    # Compute masked MSE
    mse = (((recon - original) ** 2) * mask).sum(dim=(2, 3, 4)) / (mask.sum(dim=(2, 3, 4)) + epsilon)

    # Compute PSNR
    psnr_val = 20 * torch.log10(max_val / (torch.sqrt(mse + epsilon)))
    return psnr_val.squeeze(1)  # shape: [N]


def iou(pred: torch.Tensor, target: torch.Tensor, epsilon=1e-6) -> torch.Tensor:
    """
    Computes IoU (Jaccard Index) for 3D binary segmentation.

    Args:
        pred (torch.Tensor): Predicted binary mask [N, 1, W, H, D]
        target (torch.Tensor): Ground truth binary mask [N, 1, W, H, D]
        epsilon (float): Small constant to avoid division by zero

    Returns:
        torch.Tensor: IoU per volume [N]
    """
    TP = (pred * target).sum(dim=(2, 3, 4))
    FP = (pred * (1 - target)).sum(dim=(2, 3, 4))
    FN = ((1 - pred) * target).sum(dim=(2, 3, 4))

    iou_score = (TP + epsilon) / (TP + FP + FN + epsilon)
    return iou_score.squeeze(1)  # shape [N]


def auc_score(error_map: torch.Tensor, target: torch.Tensor) -> list:
    """
    Computes AUC-ROC for 3D volumes based on voxel-wise anomaly scores.

    Args:
        error_map (torch.Tensor): Continuous anomaly scores, shape [N, 1, W, H, D]
        target (torch.Tensor): Binary ground truth labels, shape [N, 1, W, H, D]

    Returns:
        List[float]: AUC per volume (length N)
    """
    assert error_map.shape == target.shape, "Shape mismatch"

    N = error_map.shape[0]
    aucs = []

    for i in range(N):
        score_flat = error_map[i].flatten().cpu().numpy()
        target_flat = target[i].flatten().cpu().numpy()

        if (target_flat == 0).all() or (target_flat == 1).all():
            # AUC is undefined if only one class is present
            print('auc requires attention for sample', N)
            aucs.append(float('nan'))
        else:
            auc = roc_auc_score(target_flat, score_flat)
            aucs.append(auc)

    return torch.tensor(aucs, dtype=torch.float32)


def assd_batch(pred: torch.Tensor, target: torch.Tensor, spacing=(1.0, 1.0, 1.0)) -> torch.Tensor:
    """
    Computes ASSD (Average Symmetric Surface Distance) for a batch of 3D volumes.

    Args:
        pred (torch.Tensor): Predicted binary masks, shape [N, 1, W, H, D]
        target (torch.Tensor): Ground truth binary masks, shape [N, 1, W, H, D]
        spacing (tuple): Voxel spacing in mm (e.g., (1.0, 1.0, 1.0))

    Returns:
        torch.Tensor: ASSD per volume, shape [N], dtype=torch.float32
    """
    assert pred.shape == target.shape, "Shape mismatch"
    N = pred.shape[0]
    assd_values = []

    for i in range(N):
        pred_np = pred[i, 0].cpu().numpy().astype(np.uint8)
        target_np = target[i, 0].cpu().numpy().astype(np.uint8)

        pred_itk = sitk.GetImageFromArray(pred_np)
        target_itk = sitk.GetImageFromArray(target_np)

        pred_itk.SetSpacing(spacing)
        target_itk.SetSpacing(spacing)

        hd_filter = sitk.HausdorffDistanceImageFilter()
        try:
            hd_filter.Execute(pred_itk, target_itk)
            assd_val = hd_filter.GetAverageHausdorffDistance()
        except RuntimeError:
            # ASSD undefined if one of the surfaces is empty
            assd_val = float('nan')

        assd_values.append(assd_val)

    return torch.tensor(assd_values, dtype=torch.float32)


if __name__ == "__main__":


    workers = 8
    fit_funtion_y_limit = args.fit_funtion_y_limit

    print("Workers: ", workers)

    input_mod_test = args.test_modality

    print("Running on GPU:" + str(args.gpu_id))
    print("Running for epochs:" + str(args.epochs))

    print("Running on local machine")
    cuda_id = "cuda:" + str(args.gpu_id)
    device = torch.device(cuda_id)
    torch.cuda.set_device(cuda_id)

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    sensitivity_metric = ConfusionMatrixMetric(include_background=True, metric_name='sensitivity', reduction="mean",
                                               get_not_nans=False)
    precision_metric = ConfusionMatrixMetric(include_background=True, metric_name='precision', reduction="mean",
                                             get_not_nans=False)
    assd_metric = SurfaceDistanceMetric(include_background=True, reduction="mean", get_not_nans=False, symmetric=True)
    IOU_metric = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)

    in_channels = 2
    model = UNET3D(in_channels=in_channels).to(device)

    load_model_path = args.load_model_path
    model_state_dict = torch.load(load_model_path, map_location="cuda")
    model.load_state_dict(model_state_dict)
    print("LOADING MODEL: ", load_model_path)

    gamma = args.gamma
    task = args.testing_task
    shrinking_start_mask = args.shrinking_start_mask
    detection_score = args.detection_score
    images_test = []
    segs_test = []

    data_path = args.test_data_path
    train_batch_size_test = args.test_batch_size
    file_name_txt= args.test_file_name_txt
    file_name_txt2 = args.test_file_name_txt2

    if task == 'detection':
        test_loader = get_data_loader_test(input_mod_test, data_path, file_txt=file_name_txt,
                                           batch_size=train_batch_size_test, num_workers=workers, test_label=False)
        synthetic_anomaly = args.synthetic_anomaly
        synthetic_artifact_type = args.synthetic_anomaly_type
        if synthetic_anomaly is False:
            data_path2 = args.test_data_path2
            test_loader2 = get_data_loader_test(input_mod_test, data_path2, file_txt=file_name_txt2, batch_size=train_batch_size_test, num_workers=workers, test_label = False)
        else:
            test_loader2 = get_data_loader_test(input_mod_test, data_path, file_txt=file_name_txt, batch_size=train_batch_size_test, num_workers=workers, test_label = False, synthetic_anomaly=synthetic_anomaly, synthetic_artifact_type=synthetic_artifact_type)
        val_outputs_all = torch.zeros((len(test_loader.dataset) + len(test_loader2.dataset), 1, 192, 192, 192))
        ground_truth_all = torch.zeros((len(test_loader.dataset) + len(test_loader2.dataset), 1, 192, 192, 192))
        brain_mask_all = torch.zeros((len(test_loader.dataset) + len(test_loader2.dataset), 1, 192, 192, 192))
        final_mask_all = torch.zeros((len(test_loader.dataset) + len(test_loader2.dataset), 1, 192, 192, 192))
        label_anomaly_all = torch.zeros((len(test_loader.dataset)) + len(test_loader2.dataset))
        pred_size_mask = torch.zeros((len(test_loader.dataset) + len(test_loader2.dataset)))
    elif task == 'segmentation':
        test_loader = get_data_loader_test(input_mod_test, data_path, file_txt=file_name_txt,
                                           batch_size=train_batch_size_test, num_workers=workers, test_label=True)
        val_outputs_all = torch.zeros((len(test_loader.dataset), 1, 192, 192, 192))
        ground_truth_all = torch.zeros((len(test_loader.dataset), 1, 192, 192, 192))
        label_all = torch.zeros((len(test_loader.dataset), 1, 192, 192, 192))
        brain_mask_all = torch.zeros((len(test_loader.dataset), 1, 192, 192, 192))
        final_mask_all = torch.zeros((len(test_loader.dataset), 1, 192, 192, 192))
    model.eval()


    seg_channel = 0
    val_images = None
    val_labels = None
    val_outputs = None
    metric = {}
    dice_metric.reset()
    sensitivity_metric.reset()
    precision_metric.reset()
    assd_metric.reset()
    IOU_metric.reset()

    num_sample = 0

    instance_detection = 0
    sample_index = 0
    dice_after_all = 0

    ### for anomaly segmentation ###
    interaction = False
    if task == 'detection':
        test_loader = chain(
            ((data, 1) for data in test_loader2),
            ((data, 0) for data in test_loader)
        )

    else:
        test_loader = test_loader

    for test_data in test_loader:
        if task == 'detection':
            test_data_combine = test_data[0]
            ground_truth = test_data_combine[:, 0:1, :, :, :].cuda()
            highfreq_data = test_data_combine[:, 1:2, :, :, :].cuda()
            brain_mask = test_data_combine[:, 2:3, :, :, :].cuda()
            brain_mask_numpy = brain_mask[0, 0].cpu().numpy()
            label_anomaly = test_data[1]
            label_anomaly_all[num_sample:(num_sample + ground_truth.shape[0])] = label_anomaly
            ground_truth_all[num_sample:(num_sample + ground_truth.shape[0])] = ground_truth
            brain_mask_all[num_sample:(num_sample + ground_truth.shape[0])] = brain_mask

        elif task == 'segmentation':
            test_data_combine = test_data
            ground_truth = test_data_combine[:, 0:1, :, :, :].cuda()
            highfreq_data = test_data_combine[:, 1:2, :, :, :].cuda()
            brain_mask = test_data_combine[:, 2:3, :, :, :].cuda()
            label_gt = test_data_combine[:, 3:4, :, :, :].to(device)
            ground_truth_all[num_sample:(num_sample + ground_truth.shape[0])] = ground_truth
            brain_mask_all[num_sample:(num_sample + ground_truth.shape[0])] = brain_mask
            label_all[num_sample:(num_sample + ground_truth.shape[0])] = label_gt

        mask_update = torch.ones_like(brain_mask).cuda()
        model_confidence_map = torch.zeros_like(ground_truth)
        best_dice = 0
        thres_all = torch.zeros(50)




        mask_inpaint_input = brain_mask

        sw_batch_size = 1
        flag = torch.ones([ground_truth.shape[0]]).cuda()
        i = 0
        thres_all = torch.ones((100, brain_mask.shape[0])).cuda()
        thres = torch.ones(brain_mask.shape[0]).cuda()
        find_thres = True
        mask_inpaint_all_current_sample = torch.zeros([brain_mask.shape[0], 100, brain_mask.shape[1], 192, 192, 192]).cuda()


        while flag.sum()!=0:
            if i == 0:
                ratio_list = []
                mask_inpaint_new = brain_mask
                ratio = 1
                update_flag = torch.tensor(False)

            else:
                if find_thres is True:
                    for b in range(brain_mask.shape[0]):
                        # Total number of voxels in the sample
                        total_voxels = brain_mask[b].numel()
                        foreground_voxels = brain_mask[b].sum()

                        kth_num = total_voxels - (foreground_voxels * (1 - 0.01 * i))
                        kth_num = torch.clamp(kth_num, min=1)  # in case it's zero

                        # Flatten the error map for that sample
                        flattened_error = error_map[b].flatten()

                        thres_cur = torch.kthvalue(flattened_error, kth_num.int()).values
                        thres[b] = thres_cur
                    thres_all[i-1] = thres


                    mask_inpaint_new = torch.where(error_map > thres.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4), 1.0, 0.0)
                    mask_inpaint_all_current_sample[:, i - 1] = mask_inpaint_new


                    if i == 100:
                        y_data = thres_all
                        initial_guess = torch.tensor([1.0, 0.01],
                                                     dtype=torch.float32)
                        x_data = torch.arange(1, 101, dtype=torch.float32)
                        thres_final = torch.zeros(brain_mask.shape[0])
                        df_values = torch.zeros((brain_mask.shape[0], 100))
                        for b in range(brain_mask.shape[0]):
                            thres_b = thres_all[:,b]
                            y_b = y_data[:,b]
                            indices = torch.nonzero(y_b > fit_funtion_y_limit, as_tuple=True)[0]
                            if indices.numel() > 0:
                                index = indices[0]
                            else:
                                index=100
                            params, _ = curve_fit(tangent_model_shifted,
                                                  torch.arange(1, index+1, dtype=torch.float32).numpy(), y_data[:index,b].cpu().numpy(),
                                                  p0=initial_guess.cpu().numpy(),
                                                  maxfev=500000)
                            x,y = params

                            df_values[b] = tangent_model_derivative(x_data, x, y)

                            # Calculate R-squared to evaluate fit quality (using first 80 steps only)
                            y_pred = tangent_model_shifted(torch.arange(1, 81, dtype=torch.float32).numpy(), x, y)
                            y_true = y_data[:80,b].cpu().numpy()
                            ss_res = np.sum((y_true - y_pred) ** 2)
                            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                            r_squared = 1 - (ss_res / ss_tot)

                            print(f'Sample {b}: R² = {r_squared:.4f}')

                            if r_squared < 0.85:
                                df_values[b,0] = thres_b[0] - 0  # First element
                                df_values[b,1:] = thres_b[1:] - thres_b[:-1]
                                print('fit function failed')

                                window_size = 5
                                df_values_smooth = torch.zeros_like(df_values[b,:])
                                # Ensure df_values is 1D for smoothing
                                df_values_1d = df_values.squeeze() if df_values.dim() > 1 else df_values
                                for i in range(len(df_values_1d)):
                                    start = max(0, i - window_size // 2)
                                    end = min(len(df_values_1d), i + window_size // 2 + 1)
                                    df_values_smooth[i] = df_values_1d[start:end].mean()
                                df_values[b,:] = df_values_smooth


                            ### plotting code ###
                            if args.fit_function_plot == True:
                                t_fit = torch.linspace(torch.min(x_data), torch.max(x_data), 1000, device=device)
                                plt.plot(t_fit.cpu().numpy(),
                                         tangent_model_shifted(t_fit.cpu(), *map(torch.tensor, params)).cpu().numpy(),
                                         label="Fitted Curve", color='red')
                                plt.scatter(x_data.cpu().numpy(), thres_all[:, b].cpu().numpy(),
                                            label="Actual Thres Data", color='blue')
                                plt.xlabel("Iter Number")
                                plt.ylabel("Threshold")
                                plt.title(f"Sample {sample_index}: Tangent Model Fit")
                                plt.legend()
                                plt.grid(True)
                                plt.ylim(-1, 2)
                                plt.show()
                            sample_index += 1

                        mask_df = df_values > gamma
                        stop_k = torch.argmax(mask_df.float(), dim=1)
                        thres_final = thres_b[stop_k]

                        if shrinking_start_mask == 'brain_mask':
                            mask_inpaint_new = brain_mask
                        elif shrinking_start_mask == 'best_threshold_mask':
                            mask_inpaint_new = mask_inpaint_all_current_sample[torch.arange(brain_mask.shape[0]), stop_k.long()]
                        find_thres = False

                else:
                    mask_inpaint_new = torch.where(error_map > thres_final.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4), 1.0, 0.0)
                    ratio = (mask_inpaint_input.sum(dim=(1, 2, 3, 4)) - mask_inpaint_new.sum(
                        dim=(1, 2, 3, 4))) / mask_inpaint_input.sum(dim=(1, 2, 3, 4))
                    ratio = torch.where(torch.isnan(ratio), -1, ratio)


            mask_inpaint_input = mask_inpaint_new
            noise = torch.randn_like(ground_truth)
            if not find_thres:
                update_flag = (ratio < 0.01)*(flag==1)

            flag = flag * (~update_flag).int()
            if i == 0:
                flag = torch.ones([ground_truth.shape[0]]).cuda()
            if i > 0:
                index = torch.where((update_flag == 1).int())
                if len(index[0]) != 0:
                    val_outputs_all[num_sample+index[0]] = val_outputs[index].cpu()
                    if task == 'detection':
                        pred_size_mask[num_sample+index[0]] = mask_inpaint_new.sum(dim=(1,2,3,4))[index].cpu().float()


            if flag.sum() == 0:
                num_sample += ground_truth.shape[0]
                print('sample ', num_sample, ' finished',pred_size_mask.sum())
                break

            with torch.no_grad():
                masked_data = (1 - mask_inpaint_input) * ground_truth + mask_inpaint_input * noise
                input_cat = torch.cat((masked_data, highfreq_data), dim=1)
                val_outputs = model(input_cat.float())

            error_map = ((ground_truth - val_outputs) ** 2) * brain_mask
            i += 1

    print('iterative process finished')
    print('instance_detection', instance_detection)
    error_map = normalize(((ground_truth_all - val_outputs_all) ** 2) * brain_mask_all)

    ### compute the metrics ###
    if task == 'detection':
        if detection_score == 'final_mask_size':
            ### when using final mask size as anomaly score ###
            y_scores = np.array(pred_size_mask)
            y_true = np.array(label_anomaly_all)
        elif detection_score == 'mean_error':
            ### when using mean error map as anomaly score ###
            mean_error_map = error_map.mean(dim=(1, 2, 3, 4))
            y_scores = np.array(mean_error_map.cpu().numpy())
            y_true = np.array(label_anomaly_all)


        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        auroc = roc_auc_score(y_true, y_scores)
        auprc = average_precision_score(y_true, y_scores)
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auroc_manual = auc(fpr, tpr)
        print('auprc score:', auprc)
        print('auroc score:', auroc)
        target_tprs = [0.80, 0.90]
        fpr_values = {}
        fnr_values = {}
        target_fnrs = [0.80, 0.90]

        for target_tpr in target_tprs:
            idx = np.argmax(tpr >= target_tpr)  # Find first index where TPR meets/exceeds target
            fpr_values[f'FPR{int(target_tpr * 100)}'] = fpr[idx] if idx < len(fpr) else None
        print('fpr', fpr_values)
        for target_fnr in target_fnrs:
            target_tpr = 1 - target_fnr  # Convert FNR to equivalent TPR
            idx = np.argmax(tpr >= target_tpr)
            fnr_values[f'FNR{int(target_fnr * 100)}'] = 1 - tpr[idx] if idx < len(tpr) else None
        print('fnr', fnr_values)

    if task == 'segmentation':
        label_all = torch.round(label_all).to(torch.uint8)
        dice = torch.zeros(200)
        for thres in range(200):
            thres = thres
            mask_inpaint_input = torch.where(thres / 10000 < error_map, 1.0, 0.0)
            dice[thres] += dice_metric(y_pred=mask_inpaint_input, y=label_all).mean()

        max_dice_index = np.argmax(dice)
        max_dice = dice[max_dice_index]

        print('dice:', max_dice, 'max_dice_index', max_dice_index)

        predicted_mask = torch.where(max_dice_index / 10000 < error_map, 1.0, 0.0)
        sens = sensitivity(predicted_mask, label_all).mean()
        prec = precision(predicted_mask, label_all).mean()
        jaccard = iou(predicted_mask, label_all).mean()
        max_value = (ground_truth_all * (1 - label_all) * brain_mask_all).flatten(1).max()
        psnr_value = psnr(val_outputs_all, ground_truth_all, (1 - label_all) * brain_mask_all, max_val=max_value).mean()
        auc = auc_score(error_map, label_all).mean()
        assd = assd_batch(predicted_mask, label_all).mean()

        print('sensitivity:', sens, ' precision:', prec, ' jaccard:', jaccard, ' auc:', auc, ' assd:', assd,
              ' psnr_value:', psnr_value)




