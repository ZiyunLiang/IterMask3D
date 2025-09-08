import torch
import time
import random
import numpy as np
from argumentlib import args
from Nets.unet3d import UNET3D
from torch.utils.tensorboard import SummaryWriter
from dataloader import get_data_loader

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


if __name__ == "__main__":

    save_name = args.save_name
    workers = args.num_workers
    train_batch_size = args.train_batch_size
    lr = args.lr

    # learning rate drop
    drop_learning_rate = True
    drop_learning_rate_epoch = args.drop_learning_rate_epoch  # epoch at which to decrease the learning rate
    drop_learning_rate_value = 1e-4

    # parameter for data
    print("lr: ", lr)
    print("Workers: ", workers)
    print("Batch size: ", train_batch_size)

    val_loaders = []
    img_path = {}
    filename_img = {}

    input_mod = args.train_modality
    file_name_txt = args.train_file_name_txt

    data_path_train = args.train_data_path
    train_loader = get_data_loader(input_mod, data_path_train, file_txt=file_name_txt, batch_size=train_batch_size, num_workers=workers)

    print("Running on GPU:" + str(args.gpu_id))
    print("Running for epochs:" + str(args.epochs))

    print("Running on local machine")
    cuda_id = "cuda:" + str(args.gpu_id)
    device = torch.device(cuda_id)
    torch.cuda.set_device(cuda_id)

    print("batch size = ", train_batch_size)

    in_channels = 2
    model = UNET3D(in_channels=in_channels).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoched = 0

    best_metric={}
    best_metric = -1

    best_metric_epoch = -1
    metric_values = list()

    # create save names for tensorboard logging
    log_save = "./output/" + args.save_name
    model_save_path = "./output/" + args.save_name
    writer = SummaryWriter(log_dir=log_save)
    data_size = len(train_loader)*train_batch_size
    scaler = torch.cuda.amp.GradScaler()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for epoch in range(epoched,args.epochs):
        t1 = time.time()
        print(f"epoch {epoch + 1}/{args.epochs}")
        model.train()
        epoch_loss = 0
        step = 0

        # drop learning rate if desired
        if drop_learning_rate and epoch >= drop_learning_rate_epoch:
            for g in optimizer.param_groups:
                g['lr'] = drop_learning_rate_value

        for batch_data in train_loader:
            step += 1

            highfreq_data = batch_data[:,1:2,:,:,:].cuda()
            ground_truth = batch_data[:,0:1,:,:,:].cuda()
            brainmask_data = batch_data[:,2:3,:,:,:].cuda()
            mask_data = batch_data[:, 3:4, :, :, :].cuda()

            mask = (1 - mask_data) * brainmask_data
            noise = torch.randn_like(ground_truth)
            input_masked = (1 - mask) * ground_truth + mask * noise
            input_data = torch.cat((input_masked, highfreq_data), dim=1)

            out = model(input_data.float())
            loss = mean_flat((ground_truth - out) ** 2).mean()

            optimizer.zero_grad()

            ### mixed precision ###
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_len = data_size  // train_batch_size
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
            writer.add_scalar("learning_rate", (optimizer.param_groups)[0]['lr'], epoch_len * epoch + step)
        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        t2 = time.time()
        print('time', t2 - t1)


        if (epoch + 1) % args.save_epoch == 0:
            model_save_name = model_save_path + "_Epoch_" + str(epoch) + ".pth"
            opt_save_name = model_save_path + "_checkpoint_Epoch_" + str(epoch) + ".pt"
            torch.save(model.state_dict(), model_save_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, opt_save_name)
            print("Saved Model")


