from random import randint
from PIL import Image
import numpy as np
from my_py_tools.image.img_format import get_img
import copy


# === Resize image with zero-padding ===


# Resize to specific width and height with zero-padding
# step 1. If the width or height is larger than target, remain the proportion,
#         and make the width and height <= target width and height.
# step 2. Fill zero (black) in unfilled region,
#         and make original region in the center.
#
# Parameters:
#     pil_img_mode:
#         - 'none'
#     input_mode & img:
#         See the function "get_img(...)" in "img_format.py" file for more information.
#     target_size: tuple (target_width, target_height):
#         The size you want your image be resized.
#     return_mode:
#         - 'array'
#         - 'PIL_Image'
def resize_with_zero_padding(img,
                             # pil_img_mode: str,
                             input_mode='file_path',
                             target_size=(100, 100),
                             return_mode='array'):
    # Get img
    img = get_img(img=img, input_mode=input_mode, return_mode='PIL_Image')

    # Fit width
    w, h = img.size
    target_w, target_h = target_size
    if w > target_w:
        img = img.resize((target_w, int(h * (target_w / w))))

    # Fit height
    w, h = img.size
    if h > target_h:
        img = img.resize((int(w * (target_h / h)), target_h))

    # Zero padding
    w, h = img.size
    w_padding = target_w - w
    h_padding = target_h - h
    img = img.crop(
        [-(w_padding // 2), -(h_padding // 2), w + w_padding - (w_padding // 2), h + h_padding - (h_padding // 2)])
    # The other way to add zero:
    # img_arr = np.array(img)
    # img_arr = np.concatenate([np.zeros((target_height, total_padding//2, 3), dtype="uint8"), \
    # img_arr, np.zeros((target_height, total_padding-total_padding//2, 3))], 1)
    # img_arr = img_arr.astype('uint8')
    # img = Image.fromarray(img_arr)

    # Return in specific format
    if return_mode == 'array':
        return np.array(img)
    elif return_mode == 'PIL_Image':
        return img
    else:
        raise ValueError('Invalid return format.')


# === Randomly Crop ===


# Parameters:
#     img & input_mode: See "get_img(...)" function in "img_format.py" for more information.
#     bbox_size: tuple (bbox_width, bbox_height)
#     min_size: tuple (min_width, min_height).
#         If w or h of img is smaller bbox_size's, the img will be resized in the same proportion.
#         Both w and h of min_size must be equivalent or larger than bbox_size's.
#     all_possible:
#         Determine if return all possible crop.
#         If this parameter is True, it'll return a list of object; otherwise, a single object.
#         What is the object's format depends on the parameter "return_mode".
#     return_mode: 'PIL_Image' or 'array'
#         Determine if the return is/are PIL Image(s) or numpy array(s).
#
# P.S.
#     Suggest to turn bbox_size smaller than min_size to get different crop every time.
def random_crop(img,
                bbox_size,
                input_mode='file_path',
                min_size='bbox_size',
                all_possible=False,
                return_mode='PIL_Image'
                ):
    # Get img
    img = get_img(img, input_mode=input_mode, return_mode='PIL_Image')

    # Check if the two size are in valid format and get width and height
    if type(bbox_size) is not tuple or (min_size != 'bbox_size' and type(min_size) is not tuple):
        raise ValueError('Parameter Error')
    (bbox_w, bbox_h) = bbox_size
    (min_w, min_h) = min_size if min_size != 'bbox_size' else bbox_size

    # Check if min_size < bbox_size
    if min_w < bbox_w or min_h < bbox_h:
        raise Exception("Height and width of min_size should be both larger than bbox_size's.")

    # Adjust size to at least width and height equivalent with min_size's
    w, h = img.size
    if w < min_w:
        img = img.resize((min_w, int(h * (min_w / w))))
    if h < min_h:
        img = img.resize((int(w * (min_h / h)), min_h))

    # Randomly crop
    w, h = img.size
    w_rand_len = w - bbox_w
    h_rand_len = h - bbox_h

    # random_tile = img.crop([left, upper, right, lower])
    # 0 - - - - - - > x
    # |
    # | Image.crop([start_x, start_y, end_x, end_y])
    # V
    # y
    if all_possible:
        all_possible_crop = []
        for h_num in range(h_rand_len + 1):
            img_cropped = None
            for w_num in range(w_rand_len + 1):
                left = w_num
                right = w_num + bbox_w
                upper = h_num
                lower = h_num + bbox_h
                img_cropped = img.crop([left, upper, right, lower])
            if return_mode == 'PIL_Image':
                all_possible_crop.append(img_cropped)
            elif return_mode == 'array':
                all_possible_crop.append(np.array(img_cropped))
        return all_possible_crop
    else:
        left = randint(0, w_rand_len)
        right = left + bbox_w
        upper = randint(0, h_rand_len)
        lower = upper + bbox_h
        img_cropped = img.crop([left, upper, right, lower])
        return img_cropped if return_mode == 'PIL_Image' else np.array(img_cropped)


# === Make image be square with zero-padding ===


# Parameters:
#     img & input_mode: See "get_img(...)" function in "img_format.py" for more information.
#     return_mode: 'array' or 'PIL_Image'
#         If return mode is PIL Image, it will be turned into a RGB image.
#     pil_mode: If the return mode is 'PIL_Image', you should give PIL mode.
#         Default is 'RGB'.
# P.S.
#     Image will be turned to RGB.
def make_square_image(img, input_mode='file_path', return_mode='PIL_Image'):
    img = get_img(img=img, input_mode=input_mode, return_mode='PIL_Image')
    w, h = img.size
    target_size = (w, w) if w > h else (h, h)
    return resize_with_zero_padding(img, input_mode='PIL_Image', target_size=target_size, return_mode=return_mode)


"""
Example:
make_square_image(input_mode='file_path', img='data/images/100489.jpg', return_mode='PIL_Image').show()
"""
