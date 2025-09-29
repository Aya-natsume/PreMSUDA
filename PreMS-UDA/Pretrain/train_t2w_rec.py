import itertools
from tqdm import tqdm
from timm.utils import ModelEmaV2
from Ours_v1_pretrain.networks.discriminator import *
from Ours_v1_pretrain.networks.generator import *
from Ours_v1_pretrain.networks.segmenter import *
from Ours_v1_pretrain.utils.metrics import *
from Ours_v1_pretrain.utils.display import *
from picai_dataset import *

# PICAI: source_modality = "t2w", target_modality = "adc"
# 超参数设置
torch.manual_seed(99)
device = torch.device("cuda:1")
data_root = "../../experiment_datas/picai"
source_modality = "t2w"
target_modality = "adc"
image_shape = (33, 192, 192)
epochs = 10
input_slices = 1
class_nums = 3
sample_nums = 3
scale = 2 ** sample_nums
H, W = image_shape[1], image_shape[2]
h, w = image_shape[1] // scale, image_shape[2] // scale
learning_rate = 0.0001
cpu_nums = 0
n_channels = 32
batch_size = 1
decay = 0.9999
n_embed = 1024
embed_dims = 4  # !!!!!!!!!!
lambda_dis = 1
lambda_gq = 1
lambda_rec = 1
lambda_adv = 0.1
lambda_seg = 1
norm_type = "gn"
act_type = "swish"
act_type_D = "leaky"

# 创建discriminator
ID = ImageDiscriminator(
    input_slices, n_channels, sample_nums, norm_type=norm_type, act_type=act_type_D
).to(device)
print("ID:", ID)

# 创建generator
# encoder
GE = GeneratorEncoder(
    input_slices, n_channels, sample_nums, norm_type=norm_type, act_type=act_type
).to(device)
GE_EMA = ModelEmaV2(GE, decay=decay, device=device)
print("GE_EMA:", GE_EMA)
# group quantizer
GQ = GroupQuantizer(n_embed, embed_dims, beta=0.25, legacy=False).to(device)
GQ_EMA = ModelEmaV2(GQ, decay=decay, device=device)
print("GQ_EMA:", GQ_EMA)
# decoder
GD = GeneratorDecoder(
    input_slices, n_channels * scale, sample_nums,
    norm_type=norm_type, act_type=act_type
).to(device)
GD_EMA = ModelEmaV2(GD, decay=decay, device=device)
print("GD_EMA:", GD_EMA)

# 创建segmenter !!!!!!!!!!
# encoder
SE = SegmenterEncoder(
    input_slices, n_channels, sample_nums,
    norm_type=norm_type, act_type=act_type
).to(device)
SE_EMA = ModelEmaV2(model=SE, decay=decay, device=device)
print("SE_EMA:", SE_EMA)
# decoder
SD = SegmenterDecoder(
    input_slices, n_channels * scale, class_nums,
    sample_nums, norm_type=norm_type, act_type=act_type
).to(device)
SD_EMA = ModelEmaV2(SD, decay=decay, device=device)
print("SD_EMA:", SD_EMA)

# 损失函数
criterion_MSE = nn.MSELoss().to(device)
criterion_L1 = nn.L1Loss().to(device)
criterion_CE = nn.CrossEntropyLoss().to(device)

# 定义优化函数
optimizer_D = torch.optim.AdamW(ID.parameters(), lr=learning_rate)
optimizer_G = torch.optim.AdamW(
    itertools.chain(GE.parameters(), GQ.parameters(), GD.parameters()), lr=learning_rate
)
optimizer_S = torch.optim.AdamW(itertools.chain(SE.parameters(), SD.parameters()), lr=learning_rate)

# train dataloader
train_dataloader = DataLoader(
    dataset=PICAITrainDataset(
        data_root=os.path.join(data_root, "train"), modality=source_modality, n_slices=input_slices
    ), batch_size=batch_size, shuffle=True, num_workers=cpu_nums
)

# valid dataloader
valid_dataloader = DataLoader(
    dataset=PICAIValidDataset(
        data_root=os.path.join(data_root, "valid"), source_modality=source_modality, target_modality=target_modality
    ), batch_size=1, shuffle=False, num_workers=cpu_nums
)

# test dataloader
test_dataloader = DataLoader(
    dataset=PICAITestDataset(
        data_root=os.path.join(data_root, "valid"), source_modality=source_modality, target_modality=target_modality
    ), batch_size=1, shuffle=False, num_workers=cpu_nums
)


# train and valid
def main():
    # valid指标
    valid_best_PSNR = 0
    valid_best_DSC = 0
    # test指标
    test_best_PSNR = 0
    test_best_DSC = 0

    # 开始训练
    ID.train()
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(epochs):
        GE.train()
        GQ.train()
        GD.train()
        SE.train()
        SD.train()

        train_tqdm = tqdm(train_dataloader, total=len(train_dataloader), ncols=200)
        train_tqdm.set_description_str("[Train]-[Epoch:%d/%d]" % (epoch + 1, epochs))
        for batch_index, input_batch in enumerate(train_tqdm):
            real_slice = input_batch[source_modality].to(device)
            slice_label = input_batch["seg"].to(device)

            real_label = torch.ones(slice_label.shape[0], 1, h, w).to(device)
            fake_label = torch.zeros(slice_label.shape[0], 1, h, w).to(device)

            # train discriminator
            with torch.no_grad():
                real_quant, _ = GQ(GE(real_slice))
                fake_slice = GD(real_quant)
            # loss discriminate
            loss_dis = criterion_MSE(ID(real_slice), real_label) + criterion_MSE(ID(fake_slice), fake_label)

            # total loss
            loss_D = lambda_dis * loss_dis
            optimizer_D.zero_grad()  # 在反向传播之前，先将梯度归0
            loss_D.backward()  # 将误差反向传播
            optimizer_D.step()  # 更新参数

            # 将loss值写进txt文件里面
            with open("%s_discriminator_loss.txt" % source_modality, "a") as file_D:
                loss_d = "loss_D: {:.5f}\t\t".format(loss_D)
                loss_dis_ = "loss_dis: {:.5f}\n".format(loss_dis)
                file_D.write(loss_d)
                file_D.write(loss_dis_)
            file_D.close()

            # train segmentor !!!!!!!!!!
            with torch.no_grad():
                real_quant, _ = GQ(GE(real_slice))
                fake_slice = GD(real_quant)
            fake_feature = SE(fake_slice)
            real_pred = SD(fake_feature)
            # loss of seg
            loss_seg = criterion_CE(real_pred, slice_label)

            # total loss
            loss_S = lambda_seg * loss_seg
            optimizer_S.zero_grad()  # 在反向传播之前，先将梯度归0
            loss_S.backward()  # 将误差反向传播
            optimizer_S.step()  # 更新参数
            # EMA update
            SE_EMA.update(SE)
            SD_EMA.update(SD)

            # 将loss值写进txt文件里面
            with open("%s_segmentor_loss.txt" % source_modality, "a") as file_S:
                loss_s = "loss_S: {:.5f}\t\t".format(loss_S)
                loss_seg_ = "loss_seg: {:.5f}\n".format(loss_seg)
                file_S.write(loss_s)
                file_S.write(loss_seg_)
            file_S.close()

            # train generator
            # loss group quantization
            real_quant, loss_gq = GQ(GE(real_slice))
            # loss reconstruction
            fake_slice = GD(real_quant)
            loss_rec = criterion_L1(fake_slice, real_slice)
            # loss adversarial
            loss_adv = criterion_MSE(ID(fake_slice), real_label)

            # total loss
            loss_G = lambda_gq * loss_gq + lambda_rec * loss_rec + lambda_adv * loss_adv
            optimizer_G.zero_grad()  # 在反向传播之前，先将梯度归0
            loss_G.backward()  # 将误差反向传播
            optimizer_G.step()  # 更新参数
            # EMA update
            GE_EMA.update(GE)
            GQ_EMA.update(GQ)
            GD_EMA.update(GD)

            # 将loss值写进txt文件里面
            with open("%s_generator_loss.txt" % source_modality, "a") as file_G:
                loss_g = "loss_G: {:.5f}\t\t".format(loss_G)
                loss_gq_ = "loss_gq: {:.5f}\t\t".format(loss_gq)
                loss_rec_ = "loss_rec: {:.5f}\t\t".format(loss_rec)
                loss_adv_ = "loss_adv: {:.5f}\n".format(loss_adv)
                file_G.write(loss_g)
                file_G.write(loss_gq_)
                file_G.write(loss_rec_)
                file_G.write(loss_adv_)
            file_G.close()

            # Print log
            train_tqdm.set_postfix_str("loss_D:%.5f, loss_G:%.5f, loss_S:%.5f" % (loss_D, loss_G, loss_S))

        # 开始验证
        GE_EMA.eval()
        GQ_EMA.eval()
        GD_EMA.eval()
        SE_EMA.eval()
        SD_EMA.eval()

        # 创建存储指标的列表
        valid_PSNR_list = []
        valid_DSC_list = []
        test_PSNR_list = []
        test_DSC_list = []

        valid_tqdm = tqdm(valid_dataloader, total=len(valid_dataloader), ncols=200)
        valid_tqdm.set_description_str("[Valid]-[Epoch:%d/%d]" % (epoch + 1, epochs))
        for batch_index, input_batch in enumerate(valid_tqdm):
            real_image = input_batch[source_modality].to(device)
            image_label = input_batch["seg"].to(device)

            fake_image = torch.zeros_like(real_image).to(device)
            real_preds = torch.zeros(image_label.shape[0], class_nums, image_shape[0], H, W).to(device)
            count = torch.zeros(image_shape[0]).to(device)

            for slice_index in range(image_shape[0] - input_slices + 1):
                start_index = slice_index
                end_index = start_index + input_slices
                count[start_index:end_index] += 1

                real_slice = real_image[:, start_index:end_index]

                with torch.no_grad():
                    real_quant, _ = GQ_EMA(GE_EMA(real_slice))
                    real_feature = SE_EMA(real_slice)

                with torch.no_grad():
                    fake_slice = GD_EMA(real_quant)
                    real_pred = SD_EMA(real_feature)

                    fake_image[:, start_index:end_index] += fake_slice
                    real_preds[:, :, start_index:end_index] += real_pred

            for slice_index in range(image_shape[0]):
                fake_image[:, slice_index] /= count[slice_index]
                real_preds[:, :, slice_index] /= count[slice_index]

            real_seg = torch.argmax(real_preds, dim=1)

            # 计算PSNR
            PSNR_value = psnr(fake_image.squeeze(0), real_image.squeeze(0))
            valid_PSNR_list.append(PSNR_value)

            # 计算DSC
            DSC_value = calc_multi_dice(real_seg.squeeze(0), image_label.squeeze(0), num_cls=class_nums)
            valid_DSC_list.append(DSC_value)

            # Print log
            valid_tqdm.set_postfix_str("source_PSNR:%.3f, source_DSC:%.3f%%" %
                                       (PSNR_value, torch.tensor(DSC_value).mean() * 100))

            # 保存图像
            if epoch == 0:
                for i in range(image_shape[0]):
                    save_image(real_image[:, i:i + 1].cpu(), class_nums=class_nums, is_seg=False,
                               save_path="display_images/image/real_%s/patient_%d-slice_%d.png" %
                                         (source_modality, batch_index + 1, i + 1))
            if PSNR_value >= valid_best_PSNR:
                for i in range(image_shape[0]):
                    save_image(fake_image[:, i:i + 1].cpu(), class_nums=class_nums, is_seg=False,
                               save_path="display_images/image/fake_%s/patient_%d-slice_%d.png" %
                                         (source_modality, batch_index + 1, i + 1))

        test_tqdm = tqdm(test_dataloader, total=len(test_dataloader), ncols=200)
        test_tqdm.set_description_str("[Test]-[Epoch:%d/%d]" % (epoch + 1, epochs))
        for batch_index, input_batch in enumerate(test_tqdm):
            real_image = input_batch[source_modality].to(device)
            image_label = input_batch["seg"].to(device)

            fake_image = torch.zeros_like(real_image).to(device)
            real_preds = torch.zeros(image_label.shape[0], class_nums, image_shape[0], H, W).to(device)
            count = torch.zeros(image_shape[0]).to(device)

            for slice_index in range(image_shape[0] - input_slices + 1):
                start_index = slice_index
                end_index = start_index + input_slices
                count[start_index:end_index] += 1

                real_slice = real_image[:, start_index:end_index]

                with torch.no_grad():
                    real_quant, _ = GQ_EMA(GE_EMA(real_slice))
                    real_feature = SE_EMA(real_slice)

                with torch.no_grad():
                    fake_slice = GD_EMA(real_quant)
                    real_pred = SD_EMA(real_feature)

                    fake_image[:, start_index:end_index] += fake_slice
                    real_preds[:, :, start_index:end_index] += real_pred

            for slice_index in range(image_shape[0]):
                fake_image[:, slice_index] /= count[slice_index]
                real_preds[:, :, slice_index] /= count[slice_index]

            real_seg = torch.argmax(real_preds, dim=1)

            # 计算PSNR
            PSNR_value = psnr(fake_image.squeeze(0), real_image.squeeze(0))
            test_PSNR_list.append(PSNR_value)

            # 计算DSC
            DSC_value = calc_multi_dice(real_seg.squeeze(0), image_label.squeeze(0), num_cls=class_nums)
            test_DSC_list.append(DSC_value)

            # Print log
            test_tqdm.set_postfix_str("source_PSNR:%.3f, source_DSC:%.3f%%" %
                                      (PSNR_value, torch.tensor(DSC_value).mean() * 100))

        # 根据指标保存模型
        valid_PSNR_tensor = torch.tensor(valid_PSNR_list)
        valid_PSNR_mean = valid_PSNR_tensor.mean()
        valid_PSNR_std = valid_PSNR_tensor.std()
        test_PSNR_mean = torch.tensor(test_PSNR_list).mean()
        if valid_PSNR_mean >= valid_best_PSNR and test_PSNR_mean >= test_best_PSNR:
            valid_best_PSNR = valid_PSNR_mean
            test_best_PSNR = test_PSNR_mean
            # 原始版本
            torch.save(GE.state_dict(), "saved_models/original_version/PSNR/GE_%s_best.pth" % source_modality)
            torch.save(GQ.state_dict(), "saved_models/original_version/PSNR/GQ_%s_best.pth" % source_modality)
            torch.save(GD.state_dict(), "saved_models/original_version/PSNR/GD_%s_best.pth" % source_modality)
            torch.save(SE.state_dict(), "saved_models/original_version/PSNR/SE_%s_best.pth" % source_modality)
            torch.save(SD.state_dict(), "saved_models/original_version/PSNR/SD_%s_best.pth" % source_modality)
            # EMA版本
            torch.save(GE_EMA.state_dict(), "saved_models/ema_version/PSNR/GE_%s_best.pth" % source_modality)
            torch.save(GQ_EMA.state_dict(), "saved_models/ema_version/PSNR/GQ_%s_best.pth" % source_modality)
            torch.save(GD_EMA.state_dict(), "saved_models/ema_version/PSNR/GD_%s_best.pth" % source_modality)
            torch.save(SE_EMA.state_dict(), "saved_models/ema_version/PSNR/SE_%s_best.pth" % source_modality)
            torch.save(SD_EMA.state_dict(), "saved_models/ema_version/PSNR/SD_%s_best.pth" % source_modality)
            print("Save my models with the best PSNR of source modality(Valid:%.3f-MAX, Test:%.3f-MAX)!" %
                  (valid_best_PSNR, test_best_PSNR))
        elif valid_PSNR_mean < valid_best_PSNR and test_PSNR_mean >= test_best_PSNR:
            test_best_PSNR = test_PSNR_mean
            # 原始版本
            torch.save(GE.state_dict(), "saved_models/original_version/PSNR/GE_%s_best.pth" % source_modality)
            torch.save(GQ.state_dict(), "saved_models/original_version/PSNR/GQ_%s_best.pth" % source_modality)
            torch.save(GD.state_dict(), "saved_models/original_version/PSNR/GD_%s_best.pth" % source_modality)
            torch.save(SE.state_dict(), "saved_models/original_version/PSNR/SE_%s_best.pth" % source_modality)
            torch.save(SD.state_dict(), "saved_models/original_version/PSNR/SD_%s_best.pth" % source_modality)
            # EMA版本
            torch.save(GE_EMA.state_dict(), "saved_models/ema_version/PSNR/GE_%s_best.pth" % source_modality)
            torch.save(GQ_EMA.state_dict(), "saved_models/ema_version/PSNR/GQ_%s_best.pth" % source_modality)
            torch.save(GD_EMA.state_dict(), "saved_models/ema_version/PSNR/GD_%s_best.pth" % source_modality)
            torch.save(SE_EMA.state_dict(), "saved_models/ema_version/PSNR/SE_%s_best.pth" % source_modality)
            torch.save(SD_EMA.state_dict(), "saved_models/ema_version/PSNR/SD_%s_best.pth" % source_modality)
            print("Save my models with the best PSNR of source modality(Valid:%.3f, Test:%.3f-MAX)!" %
                  (valid_PSNR_mean, test_best_PSNR))
        elif valid_PSNR_mean >= valid_best_PSNR and test_PSNR_mean < test_best_PSNR:
            valid_best_PSNR = valid_PSNR_mean
            print("Didn't save my models with the best PSNR of source modality(Valid:%.3f-MAX, Test:%.3f)!" %
                  (valid_best_PSNR, test_PSNR_mean))
        else:
            print("Didn't save my models with the PSNR of source modality(Valid:%.3f, Test:%.3f)!" %
                  (valid_PSNR_mean, test_PSNR_mean))

        valid_DSC_tensor = torch.tensor(valid_DSC_list)
        valid_DSC_mean = valid_DSC_tensor.mean()
        valid_DSC_std = valid_DSC_tensor.std()
        test_DSC_mean = torch.tensor(test_DSC_list).mean()
        if valid_DSC_mean >= valid_best_DSC and test_DSC_mean >= test_best_DSC:
            valid_best_DSC = valid_DSC_mean
            test_best_DSC = test_DSC_mean
            # 原始版本
            torch.save(GE.state_dict(), "saved_models/original_version/DSC/GE_%s_best.pth" % source_modality)
            torch.save(GQ.state_dict(), "saved_models/original_version/DSC/GQ_%s_best.pth" % source_modality)
            torch.save(GD.state_dict(), "saved_models/original_version/DSC/GD_%s_best.pth" % source_modality)
            torch.save(SE.state_dict(), "saved_models/original_version/DSC/SE_%s_best.pth" % source_modality)
            torch.save(SD.state_dict(), "saved_models/original_version/DSC/SD_%s_best.pth" % source_modality)
            # EMA版本
            torch.save(GE_EMA.state_dict(), "saved_models/ema_version/DSC/GE_%s_best.pth" % source_modality)
            torch.save(GQ_EMA.state_dict(), "saved_models/ema_version/DSC/GQ_%s_best.pth" % source_modality)
            torch.save(GD_EMA.state_dict(), "saved_models/ema_version/DSC/GD_%s_best.pth" % source_modality)
            torch.save(SE_EMA.state_dict(), "saved_models/ema_version/DSC/SE_%s_best.pth" % source_modality)
            torch.save(SD_EMA.state_dict(), "saved_models/ema_version/DSC/SD_%s_best.pth" % source_modality)
            print("Save my models with the best DSC of source modality(Valid:%.3f%%-MAX, Test:%.3f%%-MAX)!" %
                  (valid_best_DSC * 100, test_best_DSC * 100))
        elif valid_DSC_mean < valid_best_DSC and test_DSC_mean >= test_best_DSC:
            test_best_DSC = test_DSC_mean
            # 原始版本
            torch.save(GE.state_dict(), "saved_models/original_version/DSC/GE_%s_best.pth" % source_modality)
            torch.save(GQ.state_dict(), "saved_models/original_version/DSC/GQ_%s_best.pth" % source_modality)
            torch.save(GD.state_dict(), "saved_models/original_version/DSC/GD_%s_best.pth" % source_modality)
            torch.save(SE.state_dict(), "saved_models/original_version/DSC/SE_%s_best.pth" % source_modality)
            torch.save(SD.state_dict(), "saved_models/original_version/DSC/SD_%s_best.pth" % source_modality)
            # EMA版本
            torch.save(GE_EMA.state_dict(), "saved_models/ema_version/DSC/GE_%s_best.pth" % source_modality)
            torch.save(GQ_EMA.state_dict(), "saved_models/ema_version/DSC/GQ_%s_best.pth" % source_modality)
            torch.save(GD_EMA.state_dict(), "saved_models/ema_version/DSC/GD_%s_best.pth" % source_modality)
            torch.save(SE_EMA.state_dict(), "saved_models/ema_version/DSC/SE_%s_best.pth" % source_modality)
            torch.save(SD_EMA.state_dict(), "saved_models/ema_version/DSC/SD_%s_best.pth" % source_modality)
            print("Save my models with the best DSC of source modality(Valid:%.3f%%, Test:%.3f%%-MAX)!" %
                  (valid_DSC_mean * 100, test_best_DSC * 100))
        elif valid_DSC_mean >= valid_best_DSC and test_DSC_mean < test_best_DSC:
            valid_best_DSC = valid_DSC_mean
            print("Didn't save my models with the best DSC of source modality(Valid:%.3f%%-MAX, Test:%.3f%%)!" %
                  (valid_best_DSC * 100, test_DSC_mean * 100))
        else:
            print("Didn't save my models with the DSC of source modality(Valid:%.3f%%, Test:%.3f%%)!" %
                  (valid_DSC_mean * 100, test_DSC_mean * 100))

        with open("%s_psnr_metrics.txt" % source_modality, "a") as file_valid:
            file_valid.write("source_PSNR:")
            for z in valid_PSNR_list:
                file_valid.write("{:.3f}\t".format(z))
            file_valid.write("\nsource_PSNR_mean:%.3f" % valid_PSNR_mean)
            file_valid.write("\nsource_PSNR_std:%.3f" % valid_PSNR_std)
            file_valid.write("\n\n")
        file_valid.close()

        organ_DSC_mean = []
        organ_DSC_std = []
        for organ in range(class_nums - 1):
            organ_DSC_mean.append(valid_DSC_tensor[:, organ].mean())
            organ_DSC_std.append(valid_DSC_tensor[:, organ].std())

        with open("%s_dsc_metrics.txt" % source_modality, "a") as file_valid:
            file_valid.write("source_DSC:\n")
            for z in valid_DSC_list:
                file_valid.write("{}\n".format(z))
            file_valid.write("\nsource_organ_DSC_mean(%):")
            for z in organ_DSC_mean:
                file_valid.write("{:.3f}\t".format(z * 100))
            file_valid.write("\nsource_organ_DSC_std:")
            for z in organ_DSC_std:
                file_valid.write("{:.3f}\t".format(z))
            file_valid.write("\nsource_DSC_mean(%%):%.3f" % (valid_DSC_mean * 100))
            file_valid.write("\nsource_DSC_std:%.3f" % valid_DSC_std)
            file_valid.write("\n\n\n")
        file_valid.close()

    # 保存最后的模型
    # 原始版本
    torch.save(GE.state_dict(), "final_models/original_version/GE_%s_final.pth" % source_modality)
    torch.save(GQ.state_dict(), "final_models/original_version/GQ_%s_final.pth" % source_modality)
    torch.save(GD.state_dict(), "final_models/original_version/GD_%s_final.pth" % source_modality)
    torch.save(SE.state_dict(), "final_models/original_version/SE_%s_final.pth" % source_modality)
    torch.save(SD.state_dict(), "final_models/original_version/SD_%s_final.pth" % source_modality)
    # EMA版本
    torch.save(GE_EMA.state_dict(), "final_models/ema_version/GE_%s_final.pth" % source_modality)
    torch.save(GQ_EMA.state_dict(), "final_models/ema_version/GQ_%s_final.pth" % source_modality)
    torch.save(GD_EMA.state_dict(), "final_models/ema_version/GD_%s_final.pth" % source_modality)
    torch.save(SE_EMA.state_dict(), "final_models/ema_version/SE_%s_final.pth" % source_modality)
    torch.save(SD_EMA.state_dict(), "final_models/ema_version/SD_%s_final.pth" % source_modality)
    print("训练结束")


if __name__ == "__main__":
    main()
