import pdb
import numpy as np
import torchvision.transforms as transforms

label_colors = (
    (211, 44, 31),  # 1：d32c1f
    (205, 140, 149),  # 2：CD8C95
    (67, 107, 173),  # 3：436bad
    (205, 173, 0),  # 4：CDAD00
    (4, 244, 137),  # 5：04f489
    (254, 1, 154),  # 6：fe019a
    (6, 71, 12),  # 7：06470c
    (97, 222, 42),  # 8：61de2a
    (203, 248, 95),  # 9：cbf85f
    (255, 187, 255),  # 10：FFBBFF
    (127, 255, 212),  # 11：7FFFD4
    (0, 0, 255),  # 12：0000FF
    (2, 204, 254),  # 13：02ccfe
    (153, 0, 250),  # 14：9900fa
    (93, 20, 81),  # 15：5d1451
)


def fill_buf(buf, i, image, shape):
    # n = buf.shape[0] / shape[1]
    m = buf.shape[1] / shape[1]
    sx = int(i % m) * shape[1]
    sy = int(i / m) * shape[0]

    try:
        buf[sy:sy + shape[0], sx:sx + shape[1], :] = image
    except:
        pdb.set_trace()


def visual(X, bgr2rgb=False, need_restore=True):
    assert len(X.shape) == 4

    if need_restore:
        X = np.clip((X + 1.0) * (255.0 / 2.0), 0, 255).astype(np.uint8)  # restore(X)

    n = int(np.ceil(np.sqrt(X.shape[0])))
    buff = np.zeros((n * X.shape[1], n * X.shape[2], X.shape[3]), dtype=np.uint8)

    for i, image in enumerate(X):
        fill_buf(buff, i, image, X.shape[1:])

    if bgr2rgb:
        pdb.set_trace()
        buff = buff[:, :, ::-1]
    return buff


def display_image(image, class_nums, is_seg=False):
    assert image.ndim == 4

    # image = image[:, :, ::-1, :].transpose((2, 3, 1, 0))
    image = image.permute(2, 3, 1, 0)
    image = image.rot90(k=2, dims=(0, 1))
    image = image.reshape((image.shape[0], image.shape[1], image.shape[2] * image.shape[3]))
    # image = image.transpose((2, 0, 1))
    image = image.permute(2, 0, 1)
    image = np.expand_dims(image, -1)

    if is_seg:
        image = visual(image, need_restore=False)
    else:
        image = visual(image)
    image = np.expand_dims(image, 0)

    if is_seg:
        new_im = np.tile(image, (1, 1, 1, 3))
        for c in range(1, class_nums):
            new_im[:, :, :, 0:1][image == c] = label_colors[c - 1][0]
            new_im[:, :, :, 1:2][image == c] = label_colors[c - 1][1]
            new_im[:, :, :, 2:3][image == c] = label_colors[c - 1][2]
        image = new_im.astype(np.uint8)

    return image[0]


def save_image(image, class_nums, save_path, is_seg=False):
    # 创建一个转换器，将张量转换为PIL图像
    to_PIL_image = transforms.ToPILImage()
    # 保存图像
    process_image = display_image(image, class_nums=class_nums, is_seg=is_seg)
    process_image = to_PIL_image(process_image)
    process_image.save(save_path)
