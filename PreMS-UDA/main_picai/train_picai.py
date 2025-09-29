import itertools
from tqdm import tqdm
from timm.utils import ModelEmaV2
from Ours_v1_new.networks.our_module import *
from Ours_v1_new.networks.discriminators import *
from Ours_v1_new.networks.segmenter import *
from Ours_v1_new.networks.generator import *
from Ours_v1_new.utils.metrics import *
from Ours_v1_new.utils.display import *
from picai_dataset import *

os.makedirs("saved_models/DSC/original_version/", exist_ok=True)
os.makedirs("saved_models/ASSD/original_version/", exist_ok=True)
os.makedirs("saved_models/DSC/ema_version/", exist_ok=True)
os.makedirs("saved_models/ASSD/ema_version/", exist_ok=True)

# PICAI: source_modality = "t2w", target_modality = "adc"
# 超参数设置
torch.manual_seed(99)
device = torch.device("cuda:1")
data_root = "/home/chenxu/liudongliang/experiment_datas/picai"
# source_modality = "t2w"
# target_modality = "adc"
#
source_modality = "adc"
target_modality = "t2w"

image_shape = (33, 192, 192)
epochs = 10  # ..........
input_slices = 1
class_nums = 3
sample_nums = 3
scale = 2 ** sample_nums
H, W = image_shape[1], image_shape[2]
h, w = image_shape[1] // scale, image_shape[2] // scale
learning_rate = 0.0001
cpu_nums = 0
modality_nums = 2
batch_size = 1  # ..........
decay = 0.9999
n_channels = 32
n_embed = 1024
embed_dims = 8
lambda_I_dis = 1
lambda_F_dis = 1
lambda_S_seg = 1
lambda_S2T_seg = 0.5
lambda_F_adv = 0.01
lambda_I_adv = 0.1
lambda_fc = 0.1  # **********
lambda_fdc = 0.01  # **********
n_slices = 10  # **********
kernel_list = [2, 4, 8, 16]  # **********
norm_type = "gn"
act_type = "swish"
act_type_D = "leaky"

# 加载预训练模型
# source
# encoder
GE_S = ModelEmaV2(
    model=GeneratorEncoder(
        input_slices, n_channels, sample_nums, norm_type=norm_type, act_type=act_type
    ), decay=decay, device=device
)
GE_S.load_state_dict(torch.load("pretrained_models/ema_version/GE_%s_best.pth" % source_modality))
for params in GE_S.parameters():
    params.requires_grad = False
print("GE_S:", GE_S)
# quantizer
GQ_S = ModelEmaV2(
    model=GroupQuantizer(n_embed, embed_dims, beta=0.25, legacy=False),
    decay=decay, device=device
)
GQ_S.load_state_dict(torch.load("pretrained_models/ema_version/GQ_%s_best.pth" % source_modality))
for params in GQ_S.parameters():
    params.requires_grad = False
print("GQ_S:", GQ_S)
# target
# encoder
GE_T = ModelEmaV2(
    model=GeneratorEncoder(
        input_slices, n_channels, sample_nums, norm_type=norm_type, act_type=act_type
    ), decay=decay, device=device
)
GE_T.load_state_dict(torch.load("pretrained_models/ema_version/GE_%s_best.pth" % target_modality))
for params in GE_T.parameters():
    params.requires_grad = False
print("GE_T:", GE_T)
# quantizer
GQ_T = ModelEmaV2(
    model=GroupQuantizer(n_embed, embed_dims, beta=0.25, legacy=False),
    decay=decay, device=device
)
GQ_T.load_state_dict(torch.load("pretrained_models/ema_version/GQ_%s_best.pth" % target_modality))
for params in GQ_T.parameters():
    params.requires_grad = False
print("GQ_T:", GQ_T)

# 创建our modules !!!!!!!!!!
# for generators
# source
MSFF_S = MultiScaleFeatureFusion(
    n_channels * scale, n_slices, kernel_list, norm_type=norm_type, act_type=act_type
).to(device)
MSFF_S_EMA = ModelEmaV2(model=MSFF_S, decay=decay, device=device)
print("MSFF_S_EMA:", MSFF_S_EMA)
# target
MSFF_T = MultiScaleFeatureFusion(
    n_channels * scale, n_slices, kernel_list, norm_type=norm_type, act_type=act_type
).to(device)
MSFF_T_EMA = ModelEmaV2(model=MSFF_T, decay=decay, device=device)
print("MSFF_T_EMA:", MSFF_T_EMA)
# for segmenter
# source
MSFF_S_ = MultiScaleFeatureFusion(
    n_channels * scale, n_slices, kernel_list, norm_type=norm_type, act_type=act_type
).to(device)
MSFF_S_EMA_ = ModelEmaV2(model=MSFF_S_, decay=decay, device=device)
print("MSFF_S_EMA_:", MSFF_S_EMA_)
# target
MSFF_T_ = MultiScaleFeatureFusion(
    n_channels * scale, n_slices, kernel_list, norm_type=norm_type, act_type=act_type
).to(device)
MSFF_T_EMA_ = ModelEmaV2(model=MSFF_T_, decay=decay, device=device)
print("MSFF_T_EMA_:", MSFF_T_EMA_)

# 创建discriminators
# image
# source
ID_S = ImageDiscriminator(
    input_slices, n_channels, sample_nums, norm_type=norm_type, act_type=act_type_D
).to(device)
print("ID_S:", ID_S)
# target
ID_T = ImageDiscriminator(
    input_slices, n_channels, sample_nums, norm_type=norm_type, act_type=act_type_D
).to(device)
print("ID_T:", ID_T)
# feature(source and target)
FD = FeatureDiscriminator(n_channels * scale, norm_type=norm_type, act_type=act_type_D).to(device)
print("FD:", FD)

# 创建segmenter(source and target)
# encoder
SE = SegmenterEncoder(
    input_slices + modality_nums, n_channels, sample_nums,
    norm_type=norm_type, act_type=act_type
).to(device)
SE_EMA = ModelEmaV2(model=SE, decay=decay, device=device)
print("SE_EMA:", SE_EMA)
# decoder
SD = SegmenterDecoder(
    input_slices, n_channels * scale, class_nums, sample_nums,
    norm_type=norm_type, act_type=act_type
).to(device)
SD_EMA = ModelEmaV2(model=SD, decay=decay, device=device)
print("SD_EMA:", SD_EMA)

# 创建generators
# source to target
GD_S2T = GeneratorDecoder(
    input_slices, n_channels * scale, sample_nums,
    norm_type=norm_type, act_type=act_type
).to(device)
GD_S2T_EMA = ModelEmaV2(model=GD_S2T, decay=decay, device=device)
print("GD_S2T_EMA:", GD_S2T_EMA)
# target to source
GD_T2S = GeneratorDecoder(
    input_slices, n_channels * scale, sample_nums,
    norm_type=norm_type, act_type=act_type
).to(device)
GD_T2S_EMA = ModelEmaV2(model=GD_T2S, decay=decay, device=device)
print("GD_T2S_EMA:", GD_T2S_EMA)

# 损失函数
criterion_MSE = nn.MSELoss().to(device)
criterion_CE = nn.CrossEntropyLoss().to(device)

# 定义优化函数 !!!!!!!!!!
optimizer_D = torch.optim.AdamW(
    itertools.chain(ID_S.parameters(), ID_T.parameters(), FD.parameters()), lr=learning_rate
)
optimizer_S = torch.optim.AdamW(
    itertools.chain(
        SE.parameters(), MSFF_S_.parameters(), MSFF_T_.parameters(), SD.parameters()
    ), lr=learning_rate
)
optimizer_G = torch.optim.AdamW(
    itertools.chain(
        MSFF_S.parameters(), MSFF_T.parameters(), GD_S2T.parameters(), GD_T2S.parameters()
    ), lr=learning_rate
)

# train dataloader
train_dataloader_S = DataLoader(
    dataset=PICAITrainDataset(data_root=os.path.join(data_root, "train"), n_slices=n_slices, modality=source_modality),
    batch_size=batch_size, shuffle=True, num_workers=cpu_nums
)
train_dataloader_T = DataLoader(
    dataset=PICAITrainDataset(data_root=os.path.join(data_root, "train"), n_slices=n_slices, modality=target_modality),
    batch_size=batch_size, shuffle=True, num_workers=cpu_nums
)

# valid dataloader
valid_dataloader = DataLoader(
    dataset=PICAIValidDataset(
        data_root=os.path.join(data_root, "valid"), source_modality=source_modality, target_modality=target_modality
    ), batch_size=1, shuffle=False, num_workers=cpu_nums
)


# train and valid
def main():
    assert batch_size * n_slices >= 2
    # valid指标
    # best_DSC_S = 0
    best_DSC_T = 0
    # best_ASSD_S = 100000000
    best_ASSD_T = 100000000
    # 数据集长度(train和valid)
    train_length_S, train_length_T = len(train_dataloader_S), len(train_dataloader_T)
    train_length = train_length_S if train_length_S <= train_length_T else train_length_T

    # 开始训练
    GE_S.eval()
    GQ_S.eval()
    GE_T.eval()
    GQ_T.eval()
    ID_S.train()
    ID_T.train()
    FD.train()
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(epochs):
        SE.train()
        MSFF_S_.train()  # !!!!!!!!!!
        MSFF_T_.train()  # !!!!!!!!!!
        SD.train()
        MSFF_S.train()  # !!!!!!!!!!
        MSFF_T.train()  # !!!!!!!!!!
        GD_S2T.train()
        GD_T2S.train()

        train_tqdm = tqdm(zip(train_dataloader_S, train_dataloader_T), total=train_length, ncols=200)
        train_tqdm.set_description_str("[Train]-[Epoch:%d/%d]" % (epoch + 1, epochs))
        for batch_index, (source_batch, target_batch) in enumerate(train_tqdm):
            real_image_S = source_batch[source_modality].to(device)
            S_image_label = source_batch["seg"].to(device)
            real_image_T = target_batch[target_modality].to(device)
            S_slice_label = S_image_label.reshape(-1, S_image_label.shape[2], S_image_label.shape[3]).unsqueeze(1)

            real_label = torch.ones(S_slice_label.shape[0], 1, h, w).to(device)
            fake_label = torch.zeros(S_slice_label.shape[0], 1, h, w).to(device)
            adv_label = -torch.ones(S_slice_label.shape[0], 1, h, w).to(device)

            real_slice_S = real_image_S.reshape(-1, real_image_S.shape[2], real_image_S.shape[3]).unsqueeze(1)
            real_slice_T = real_image_T.reshape(-1, real_image_T.shape[2], real_image_T.shape[3]).unsqueeze(1)

            train_mask_S = torch.zeros(real_slice_S.shape[0], modality_nums, H, W).to(device)
            train_mask_S[:, 0, :, :] = 1  # 1-0
            train_mask_T = torch.zeros(real_slice_T.shape[0], modality_nums, H, W).to(device)
            train_mask_T[:, 1, :, :] = 1  # 0-1

            real_slice_S_ = torch.cat([real_slice_S, train_mask_S], dim=1)
            real_slice_T_ = torch.cat([real_slice_T, train_mask_T], dim=1)

            with torch.no_grad():
                real_quant_S = GQ_S(GE_S(real_slice_S))
                real_quant_T = GQ_T(GE_T(real_slice_T))

            # train discriminators
            # loss image discriminate
            with torch.no_grad():
                fake_slice_S = GD_T2S(real_quant_T + MSFF_T(real_quant_T))  # !!!!!!!!!!
                fake_slice_T = GD_S2T(real_quant_S + MSFF_S(real_quant_S))  # !!!!!!!!!!
            loss_I_dis = criterion_MSE(ID_S(real_slice_S), real_label) + \
                         criterion_MSE(ID_S(fake_slice_S), fake_label) + \
                         criterion_MSE(ID_T(real_slice_T), real_label) + \
                         criterion_MSE(ID_T(fake_slice_T), fake_label)
            # loss feature discriminate
            with torch.no_grad():
                real_feature_S_ = SE(real_slice_S_)
                real_feature_T_ = SE(real_slice_T_)
            loss_F_dis = criterion_MSE(FD(real_feature_S_), real_label) + criterion_MSE(FD(real_feature_T_), adv_label)

            # total loss
            loss_D = lambda_I_dis * loss_I_dis + lambda_F_dis * loss_F_dis
            optimizer_D.zero_grad()  # 在反向传播之前，先将梯度归0
            loss_D.backward()  # 将误差反向传播
            optimizer_D.step()  # 更新参数

            # 将loss值写进txt文件里面
            with open("discriminator_loss.txt", "a") as file_D:
                loss_d = "loss_D: {:.5f}\t\t".format(loss_D)
                loss_i_dis = "loss_I_dis: {:.5f}\t\t".format(loss_I_dis)
                loss_f_dis = "loss_F_dis: {:.5f}\n".format(loss_F_dis)
                file_D.write(loss_d)
                file_D.write(loss_i_dis)
                file_D.write(loss_f_dis)
            file_D.close()

            # train segmentor
            # loss source segment
            real_feature_S_ = SE(real_slice_S_)
            real_pred_S = SD(real_feature_S_ + MSFF_S_(real_feature_S_))  # !!!!!!!!!!
            loss_S_seg = criterion_CE(real_pred_S, S_slice_label)
            # loss source to target segment
            with torch.no_grad():
                fake_slice_T = GD_S2T(real_quant_S + MSFF_S(real_quant_S))  # !!!!!!!!!!
            fake_slice_T_ = torch.cat([fake_slice_T, train_mask_T], dim=1)
            fake_feature_T_ = SE(fake_slice_T_)
            fake_pred_T = SD(fake_feature_T_ + MSFF_T_(fake_feature_T_))  # !!!!!!!!!!
            loss_S2T_seg = criterion_CE(fake_pred_T, S_slice_label)
            # loss feature adversarial
            real_feature_T_ = SE(real_slice_T_)
            loss_F_adv = criterion_MSE(FD(real_feature_T_), fake_label)

            # total loss
            loss_S = lambda_S_seg * loss_S_seg + lambda_S2T_seg * loss_S2T_seg + lambda_F_adv * loss_F_adv
            optimizer_S.zero_grad()  # 在反向传播之前，先将梯度归0
            loss_S.backward()  # 将误差反向传播
            optimizer_S.step()  # 更新参数
            # EMA update
            SE_EMA.update(SE)
            MSFF_S_EMA_.update(MSFF_S_)  # !!!!!!!!!!
            MSFF_T_EMA_.update(MSFF_T_)  # !!!!!!!!!!
            SD_EMA.update(SD)

            # 将loss值写进txt文件里面
            with open("segmenter_loss.txt", "a") as file_S:
                loss_s = "loss_S: {:.5f}\t\t".format(loss_S)
                loss_s_seg = "loss_S_seg: {:.5f}\t\t".format(loss_S_seg)
                loss_s2t_seg = "loss_S2T_seg: {:.5f}\t\t".format(loss_S2T_seg)
                loss_f_adv = "loss_F_adv: {:.5f}\n".format(loss_F_adv)
                file_S.write(loss_s)
                file_S.write(loss_s_seg)
                file_S.write(loss_s2t_seg)
                file_S.write(loss_f_adv)
            file_S.close()

            # train generators
            # loss adversarial
            fake_slice_S = GD_T2S(real_quant_T + MSFF_T(real_quant_T))  # !!!!!!!!!!
            fake_slice_T = GD_S2T(real_quant_S + MSFF_S(real_quant_S))  # !!!!!!!!!!
            loss_I_adv = criterion_MSE(ID_S(fake_slice_S), real_label) + criterion_MSE(ID_T(fake_slice_T), real_label)
            # loss feature consistent
            fake_slice_S_ = torch.cat([fake_slice_S, train_mask_S], dim=1)
            fake_slice_T_ = torch.cat([fake_slice_T, train_mask_T], dim=1)
            fake_feature_S_ = SE(fake_slice_S_)
            fake_feature_T_ = SE(fake_slice_T_)
            with torch.no_grad():
                real_feature_S_ = SE(real_slice_S_)
                real_feature_T_ = SE(real_slice_T_)
            loss_fc = criterion_MSE(fake_feature_T_, real_feature_S_) + criterion_MSE(fake_feature_S_, real_feature_T_)
            # loss feature difference consistent !!!!!!!!!!
            b_n = real_feature_T_.shape[0]
            b = b_n // n_slices
            i = n_slices * (b // 2) if b != 1 else (n_slices * b) // 2
            j = b_n - i
            if i == 0:
                loss_fdc = criterion_MSE((real_feature_S_ - fake_feature_S_), (fake_feature_T_ - real_feature_T_))
            else:
                loss_fdc = \
                    criterion_MSE(
                        (fake_feature_S_[:i] - fake_feature_S_[j:]), (real_feature_T_[:i] - real_feature_T_[j:])
                    ) + \
                    criterion_MSE(
                        (fake_feature_T_[:i] - fake_feature_T_[j:]), (real_feature_S_[:i] - real_feature_S_[j:])
                    ) + \
                    criterion_MSE((real_feature_S_ - fake_feature_S_), (fake_feature_T_ - real_feature_T_))

            # total loss
            loss_G = lambda_I_adv * loss_I_adv + lambda_fc * loss_fc + lambda_fdc * loss_fdc
            optimizer_G.zero_grad()  # 在反向传播之前，先将梯度归0
            loss_G.backward()  # 将误差反向传播
            optimizer_G.step()  # 更新参数
            # EMA update
            MSFF_S_EMA.update(MSFF_S)  # !!!!!!!!!!
            MSFF_T_EMA.update(MSFF_T)  # !!!!!!!!!!
            GD_S2T_EMA.update(GD_S2T)
            GD_T2S_EMA.update(GD_T2S)

            # 将loss值写进txt文件里面
            with open("generator_loss.txt", "a") as file_G:
                loss_g = "loss_G: {:.5f}\t\t".format(loss_G)
                loss_i_adv = "loss_I_adv: {:.5f}\t\t".format(loss_I_adv)
                loss_fc_ = "loss_fc: {:.5f}\t\t".format(loss_fc)
                loss_fdc_ = "loss_fdc: {:.5f}\n".format(loss_fdc)
                file_G.write(loss_g)
                file_G.write(loss_i_adv)
                file_G.write(loss_fc_)
                file_G.write(loss_fdc_)
            file_G.close()

            # Print log
            train_tqdm.set_postfix_str("loss_D:%.5f, loss_S:%.5f, loss_G:%.5f" % (loss_D, loss_S, loss_G))

        # 开始验证
        # MSFF_S_EMA.eval()  # !!!!!!!!!!
        MSFF_T_EMA.eval()  # !!!!!!!!!!
        # GD_S2T_EMA.eval()
        GD_T2S_EMA.eval()
        SE_EMA.eval()
        # MSFF_S_EMA_.eval() # !!!!!!!!!!
        MSFF_T_EMA_.eval()  # !!!!!!!!!!
        SD_EMA.eval()

        # 创建存储指标的列表
        # DSC_list_S = []
        DSC_list_T = []
        # ASSD_list_S = []
        ASSD_list_T = []
        # PSNR_list_S = []
        # PSNR_list_T = []

        valid_tqdm = tqdm(valid_dataloader, total=len(valid_dataloader), ncols=200)
        valid_tqdm.set_description_str("[Valid]-[Epoch:%d/%d]" % (epoch + 1, epochs))
        for batch_index, input_batch in enumerate(valid_tqdm):
            # real_image_S = input_batch[source_modality].to(device)
            shared_label = input_batch["seg"].to(device)
            real_image_T = input_batch[target_modality].to(device)
            count = torch.zeros(image_shape[0]).to(device)

            # fake_image_T = torch.zeros(image_shape[0], input_slices, image_shape[1], image_shape[2]).to(device)
            # real_preds_S = torch.zeros(image_shape[0], class_nums, input_slices, H, W).to(device)
            fake_image_S = torch.zeros(image_shape[0], input_slices, image_shape[1], image_shape[2]).to(device)
            real_preds_T = torch.zeros(image_shape[0], class_nums, input_slices, H, W).to(device)

            for slice_index in range(image_shape[0] - n_slices + 1):
                start_index = slice_index
                end_index = start_index + n_slices
                count[start_index:end_index] += 1

                # real_image_S_ = real_image_S[:, start_index:end_index]
                real_image_T_ = real_image_T[:, start_index:end_index]

                # real_slice_S = real_image_S_.reshape(-1, real_image_S_.shape[2], real_image_S_.shape[3]).unsqueeze(1)
                real_slice_T = real_image_T_.reshape(-1, real_image_T_.shape[2], real_image_T_.shape[3]).unsqueeze(1)

                # valid_mask_S = torch.zeros(real_slice_S.shape[0], modality_nums, H, W).to(device)
                # valid_mask_S[:, 0, :, :] = 1  # 1-0
                valid_mask_T = torch.zeros(real_slice_T.shape[0], modality_nums, H, W).to(device)
                valid_mask_T[:, 1, :, :] = 1  # 0-1

                # real_slice_S_ = torch.cat([real_slice_S, valid_mask_S], dim=1)
                real_slice_T_ = torch.cat([real_slice_T, valid_mask_T], dim=1)

                with torch.no_grad():
                    # real_quant_S = GQ_S(GE_S(real_slice_S))
                    # real_feature_S_ = SE_EMA(real_slice_S_)
                    real_quant_T = GQ_T(GE_T(real_slice_T))
                    real_feature_T_ = SE_EMA(real_slice_T_)

                with torch.no_grad():
                    # fake_slice_T = GD_S2T_EMA(real_quant_S + MSFF_S_EMA(real_quant_S))  # !!!!!!!!!!
                    # real_pred_S = SD_EMA(real_feature_S_ + MSFF_S_EMA_(real_feature_S_))  # !!!!!!!!!!
                    fake_slice_S = GD_T2S_EMA(real_quant_T + MSFF_T_EMA(real_quant_T))  # !!!!!!!!!!
                    real_pred_T = SD_EMA(real_feature_T_ + MSFF_T_EMA_(real_feature_T_))  # !!!!!!!!!!

                    # fake_image_T[start_index:end_index] += fake_slice_T
                    # real_preds_S[start_index:end_index] += real_pred_S
                    fake_image_S[start_index:end_index] += fake_slice_S
                    real_preds_T[start_index:end_index] += real_pred_T

            for slice_index in range(image_shape[0]):
                # fake_image_T[slice_index] /= count[slice_index]
                # real_preds_S[slice_index] /= count[slice_index]
                fake_image_S[slice_index] /= count[slice_index]
                real_preds_T[slice_index] /= count[slice_index]

            # real_seg_S = torch.argmax(real_preds_S, dim=1)
            real_seg_T = torch.argmax(real_preds_T, dim=1)

            # 计算DSC
            # source_DSC = calc_multi_dice(real_seg_S.squeeze(1), shared_label.squeeze(0), num_cls=class_nums)
            # DSC_list_S.append(source_DSC)
            target_DSC = calc_multi_dice(real_seg_T.squeeze(1), shared_label.squeeze(0), num_cls=class_nums)
            DSC_list_T.append(target_DSC)

            # 计算ASSD
            # source_ASSD = calc_multi_assd(real_seg_S.squeeze(1), shared_label.squeeze(0), num_cls=class_nums)
            # ASSD_list_S.append(source_ASSD)
            target_ASSD = calc_multi_assd(real_seg_T.squeeze(1), shared_label.squeeze(0), num_cls=class_nums)
            ASSD_list_T.append(target_ASSD)

            # # 计算PSNR
            # source_PSNR = psnr(fake_image_S.squeeze(1), real_image_S.squeeze(0))
            # PSNR_list_S.append(source_PSNR)
            # target_PSNR = psnr(fake_image_T.squeeze(1), real_image_T.squeeze(0))
            # PSNR_list_T.append(target_PSNR)

            # # Print log
            # valid_tqdm.set_postfix_str("source_DSC:%.3f%%, target_DSC:%.3f%%, source_ASSD:%.3f, target_ASSD:%.3f"
            #                            "source_PSNR:%.3f, target_PSNR:%.3f" %
            #                            (torch.tensor(source_DSC).mean() * 100, torch.tensor(target_DSC).mean() * 100,
            #                             torch.tensor(source_ASSD).mean(), torch.tensor(target_ASSD).mean(),
            #                             source_PSNR, target_PSNR))
            valid_tqdm.set_postfix_str("target_DSC:%.3f%%, target_ASSD:%.3f" %
                                       (torch.tensor(target_DSC).mean() * 100, torch.tensor(target_ASSD).mean()))

            # # 保存图像
            # if epoch == 0:
            #     for i in range(image_shape[0]):
            #         # save_image(real_image_S[:, i:i + 1].cpu(), class_nums=class_nums, is_seg=False,
            #         #            save_path="display_images/image/real_%s/patient_%d-slice_%d.png" %
            #         #                      (source_modality, batch_index + 1, i + 1))
            #         save_image(real_image_T[:, i:i + 1].cpu(), class_nums=class_nums, is_seg=False,
            #                    save_path="display_images/image/real_%s/patient_%d-slice_%d.png" %
            #                              (target_modality, batch_index + 1, i + 1))
            #         save_image(shared_label[:, i:i + 1].cpu(), class_nums=class_nums, is_seg=True,
            #                    save_path="display_images/shared_label/patient_%d-slice_%d.png" %
            #                              (batch_index + 1, i + 1))
            # if torch.tensor(source_DSC).mean() >= best_DSC_S:
            #     for i in range(image_shape[0]):
            #         save_image(fake_image_T[i:i + 1, :].cpu(), class_nums=class_nums, is_seg=False,
            #                    save_path="display_images/image/fake_%s/patient_%d-slice_%d.png" %
            #                              (target_modality, batch_index + 1, i + 1))
            #         save_image(real_seg_S[i:i + 1, :].cpu(), class_nums=class_nums, is_seg=True,
            #                    save_path="display_images/%s_seg/patient_%d-slice_%d.png" %
            #                              (source_modality, batch_index + 1, i + 1))
            # if torch.tensor(target_DSC).mean() >= best_DSC_T:
            #     for i in range(image_shape[0]):
            #         save_image(fake_image_S[i:i + 1, :].cpu(), class_nums=class_nums, is_seg=False,
            #                    save_path="display_images/image/fake_%s/patient_%d-slice_%d.png" %
            #                              (source_modality, batch_index + 1, i + 1))
            #         save_image(real_seg_T[i:i + 1, :].cpu(), class_nums=class_nums, is_seg=True,
            #                    save_path="display_images/%s_seg/patient_%d-slice_%d.png" %
            #                              (target_modality, batch_index + 1, i + 1))

        # DSC_tensor_S = torch.tensor(DSC_list_S)
        # DSC_mean_S = DSC_tensor_S.mean()
        # DSC_std_S = DSC_tensor_S.std()
        # if DSC_mean_S >= best_DSC_S:
        #     best_DSC_S = DSC_mean_S
        #     print("Save my models with the best DSC of source modality(Mean:%.3f%%-MAX)!" % (best_DSC_S * 100))
        # else:
        #     print("Didn't save my models with the DSC of source modality(Mean:%.3f%%)!" % (DSC_mean_S * 100))

        DSC_tensor_T = torch.tensor(DSC_list_T)
        DSC_mean_T = DSC_tensor_T.mean()
        DSC_std_T = DSC_tensor_T.std()
        if DSC_mean_T >= best_DSC_T:
            best_DSC_T = DSC_mean_T
            # 原始版本
            torch.save(MSFF_S.state_dict(), "saved_models/DSC/original_version/MSFF_S_best.pth")  # !!!!!!!!!!
            torch.save(MSFF_T.state_dict(), "saved_models/DSC/original_version/MSFF_T_best.pth")  # !!!!!!!!!!
            torch.save(GD_S2T.state_dict(), "saved_models/DSC/original_version/GD_S2T_best.pth")
            torch.save(GD_T2S.state_dict(), "saved_models/DSC/original_version/GD_T2S_best.pth")
            torch.save(SE.state_dict(), "saved_models/DSC/original_version/SE_best.pth")
            torch.save(MSFF_S_.state_dict(), "saved_models/DSC/original_version/MSFF_S_best_.pth")  # !!!!!!!!!!
            torch.save(MSFF_T_.state_dict(), "saved_models/DSC/original_version/MSFF_T_best_.pth")  # !!!!!!!!!!
            torch.save(SD.state_dict(), "saved_models/DSC/original_version/SD_best.pth")
            # EMA版本
            torch.save(MSFF_S_EMA.state_dict(), "saved_models/DSC/ema_version/MSFF_S_best.pth")  # !!!!!!!!!!
            torch.save(MSFF_T_EMA.state_dict(), "saved_models/DSC/ema_version/MSFF_T_best.pth")  # !!!!!!!!!!
            torch.save(GD_S2T_EMA.state_dict(), "saved_models/DSC/ema_version/GD_S2T_best.pth")
            torch.save(GD_T2S_EMA.state_dict(), "saved_models/DSC/ema_version/GD_T2S_best.pth")
            torch.save(SE_EMA.state_dict(), "saved_models/DSC/ema_version/SE_best.pth")
            torch.save(MSFF_S_EMA_.state_dict(), "saved_models/DSC/ema_version/MSFF_S_best_.pth")  # !!!!!!!!!!
            torch.save(MSFF_T_EMA_.state_dict(), "saved_models/DSC/ema_version/MSFF_T_best_.pth")  # !!!!!!!!!!
            torch.save(SD_EMA.state_dict(), "saved_models/DSC/ema_version/SD_best.pth")
            print("Save my models with the best DSC of target modality(Mean:%.3f%%-MAX)!" % (best_DSC_T * 100))
        else:
            print("Didn't save my models with the DSC of target modality(Mean:%.3f%%)!" % (DSC_mean_T * 100))

        # ASSD_tensor_S = torch.tensor(ASSD_list_S)
        # ASSD_mean_S = ASSD_tensor_S.mean()
        # ASSD_std_S = ASSD_tensor_S.std()
        # if ASSD_mean_S <= best_ASSD_S:
        #     best_ASSD_S = ASSD_mean_S
        #     print("Save my models with the best ASSD of source modality(Mean:%.3f-MIN)!" % best_ASSD_S)
        # else:
        #     print("Didn't save my models with the ASSD of source modality(Mean:%.3f)!" % ASSD_mean_S)

        ASSD_tensor_T = torch.tensor(ASSD_list_T)
        ASSD_mean_T = ASSD_tensor_T.mean()
        ASSD_std_T = ASSD_tensor_T.std()
        if ASSD_mean_T <= best_ASSD_T:
            best_ASSD_T = ASSD_mean_T
            # 原始版本
            torch.save(MSFF_S.state_dict(), "saved_models/ASSD/original_version/MSFF_S_best.pth")  # !!!!!!!!!!
            torch.save(MSFF_T.state_dict(), "saved_models/ASSD/original_version/MSFF_T_best.pth")  # !!!!!!!!!!
            torch.save(GD_S2T.state_dict(), "saved_models/ASSD/original_version/GD_S2T_best.pth")
            torch.save(GD_T2S.state_dict(), "saved_models/ASSD/original_version/GD_T2S_best.pth")
            torch.save(SE.state_dict(), "saved_models/ASSD/original_version/SE_best.pth")
            torch.save(MSFF_S_.state_dict(), "saved_models/ASSD/original_version/MSFF_S_best_.pth")  # !!!!!!!!!!
            torch.save(MSFF_T_.state_dict(), "saved_models/ASSD/original_version/MSFF_T_best_.pth")  # !!!!!!!!!!
            torch.save(SD.state_dict(), "saved_models/ASSD/original_version/SD_best.pth")
            # EMA版本
            torch.save(MSFF_S_EMA.state_dict(), "saved_models/ASSD/ema_version/MSFF_S_best.pth")  # !!!!!!!!!!
            torch.save(MSFF_T_EMA.state_dict(), "saved_models/ASSD/ema_version/MSFF_T_best.pth")  # !!!!!!!!!!
            torch.save(GD_S2T_EMA.state_dict(), "saved_models/ASSD/ema_version/GD_S2T_best.pth")
            torch.save(GD_T2S_EMA.state_dict(), "saved_models/ASSD/ema_version/GD_T2S_best.pth")
            torch.save(SE_EMA.state_dict(), "saved_models/ASSD/ema_version/SE_best.pth")
            torch.save(MSFF_S_EMA_.state_dict(), "saved_models/ASSD/ema_version/MSFF_S_best_.pth")  # !!!!!!!!!!
            torch.save(MSFF_T_EMA_.state_dict(), "saved_models/ASSD/ema_version/MSFF_T_best_.pth")  # !!!!!!!!!!
            torch.save(SD_EMA.state_dict(), "saved_models/ASSD/ema_version/SD_best.pth")
            print("Save my models with the best ASSD of target modality(Mean:%.3f-MIN)!" % best_ASSD_T)
        else:
            print("Didn't save my models with the ASSD of target modality(Mean:%.3f)!" % ASSD_mean_T)

        # organ_DSC_mean_S = []
        # organ_DSC_std_S = []
        organ_DSC_mean_T = []
        organ_DSC_std_T = []
        # organ_ASSD_mean_S = []
        # organ_ASSD_std_S = []
        organ_ASSD_mean_T = []
        organ_ASSD_std_T = []
        for organ in range(class_nums - 1):
            # organ_DSC_mean_S.append(DSC_tensor_S[:, organ].mean())
            # organ_DSC_std_S.append(DSC_tensor_S[:, organ].std())
            organ_DSC_mean_T.append(DSC_tensor_T[:, organ].mean())
            organ_DSC_std_T.append(DSC_tensor_T[:, organ].std())
            # organ_ASSD_mean_S.append(ASSD_tensor_S[:, organ].mean())
            # organ_ASSD_std_S.append(ASSD_tensor_S[:, organ].std())
            organ_ASSD_mean_T.append(ASSD_tensor_T[:, organ].mean())
            organ_ASSD_std_T.append(ASSD_tensor_T[:, organ].std())

        # PSNR_tensor_S = torch.tensor(PSNR_list_S)
        # PSNR_tensor_T = torch.tensor(PSNR_list_T)
        with open("valid_metrics.txt", "a") as file_valid:
            # file_valid.write("source_DSC:\n")
            # for z in DSC_list_S:
            #     file_valid.write("{}\n".format(z))
            # file_valid.write("source_organ_DSC_mean(%):")
            # for z in organ_DSC_mean_S:
            #     file_valid.write("{:.3f}\t".format(z * 100))
            # file_valid.write("\nsource_organ_DSC_std:")
            # for z in organ_DSC_std_S:
            #     file_valid.write("{:.3f}\t".format(z))
            # file_valid.write("\nsource_DSC_mean(%%):%.3f" % (DSC_mean_S * 100))
            # file_valid.write("\nsource_DSC_std:%.3f" % DSC_std_S)
            # file_valid.write("\n\n")

            file_valid.write("target_DSC:\n")
            for z in DSC_list_T:
                file_valid.write("{}\n".format(z))
            file_valid.write("target_organ_DSC_mean(%):")
            for z in organ_DSC_mean_T:
                file_valid.write("{:.3f}\t".format(z * 100))
            file_valid.write("\ntarget_organ_DSC_std:")
            for z in organ_DSC_std_T:
                file_valid.write("{:.3f}\t".format(z))
            file_valid.write("\ntarget_DSC_mean(%%):%.3f" % (DSC_mean_T * 100))
            file_valid.write("\ntarget_DSC_std:%.3f" % DSC_std_T)
            file_valid.write("\n\n\n")

            # file_valid.write("source_ASSD:\n")
            # for z in ASSD_list_S:
            #     file_valid.write("{}\n".format(z))
            # file_valid.write("source_organ_ASSD_mean:")
            # for z in organ_ASSD_mean_S:
            #     file_valid.write("{:.3f}\t".format(z))
            # file_valid.write("\nsource_organ_ASSD_std:")
            # for z in organ_ASSD_std_S:
            #     file_valid.write("{:.3f}\t".format(z))
            # file_valid.write("\nsource_ASSD_mean:%.3f" % ASSD_mean_S)
            # file_valid.write("\nsource_ASSD_std:%.3f" % ASSD_std_S)
            # file_valid.write("\n\n")

            file_valid.write("target_ASSD:\n")
            for z in ASSD_list_T:
                file_valid.write("{}\n".format(z))
            file_valid.write("target_organ_ASSD_mean:")
            for z in organ_ASSD_mean_T:
                file_valid.write("{:.3f}\t".format(z))
            file_valid.write("\ntarget_organ_ASSD_std:")
            for z in organ_ASSD_std_T:
                file_valid.write("{:.3f}\t".format(z))
            file_valid.write("\ntarget_ASSD_mean:%.3f" % ASSD_mean_T)
            file_valid.write("\ntarget_ASSD_std:%.3f" % ASSD_std_T)
            file_valid.write("\n\n\n")

            # file_valid.write("source_PSNR:")
            # for z in PSNR_list_S:
            #     file_valid.write("{:.3f}\t".format(z))
            # file_valid.write("\nsource_PSNR_mean:%.3f" % PSNR_tensor_S.mean())
            # file_valid.write("\nsource_PSNR_std:%.3f" % PSNR_tensor_S.std())
            # file_valid.write("\n\n")

            # file_valid.write("target_PSNR:")
            # for z in PSNR_list_T:
            #     file_valid.write("{:.3f}\t".format(z))
            # file_valid.write("\ntarget_PSNR_mean:%.3f" % PSNR_tensor_T.mean())
            # file_valid.write("\ntarget_PSNR_std:%.3f" % PSNR_tensor_T.std())
            # file_valid.write("\n\n\n")
            file_valid.write("\n")
        file_valid.close()

    # 训练结束后
    # 原始版本
    torch.save(MSFF_S.state_dict(), "final_models/original_version/MSFF_S_final.pth")  # !!!!!!!!!!
    torch.save(MSFF_T.state_dict(), "final_models/original_version/MSFF_T_final.pth")  # !!!!!!!!!!
    torch.save(GD_S2T.state_dict(), "final_models/original_version/GD_S2T_final.pth")
    torch.save(GD_T2S.state_dict(), "final_models/original_version/GD_T2S_final.pth")
    torch.save(SE.state_dict(), "final_models/original_version/SE_final.pth")
    torch.save(MSFF_S_.state_dict(), "final_models/original_version/MSFF_S_final_.pth")  # !!!!!!!!!!
    torch.save(MSFF_T_.state_dict(), "final_models/original_version/MSFF_T_final_.pth")  # !!!!!!!!!!
    torch.save(SD.state_dict(), "final_models/original_version/SD_final.pth")
    # EMA版本
    torch.save(MSFF_S_EMA.state_dict(), "final_models/ema_version/MSFF_S_final.pth")  # !!!!!!!!!!
    torch.save(MSFF_T_EMA.state_dict(), "final_models/ema_version/MSFF_T_final.pth")  # !!!!!!!!!!
    torch.save(GD_S2T_EMA.state_dict(), "final_models/ema_version/GD_S2T_final.pth")
    torch.save(GD_T2S_EMA.state_dict(), "final_models/ema_version/GD_T2S_final.pth")
    torch.save(SE_EMA.state_dict(), "final_models/ema_version/SE_final.pth")
    torch.save(MSFF_S_EMA_.state_dict(), "final_models/ema_version/MSFF_S_final_.pth")  # !!!!!!!!!!
    torch.save(MSFF_T_EMA_.state_dict(), "final_models/ema_version/MSFF_T_final_.pth")  # !!!!!!!!!!
    torch.save(SD_EMA.state_dict(), "final_models/ema_version/SD_final.pth")
    print("训练结束")


if __name__ == "__main__":
    main()
