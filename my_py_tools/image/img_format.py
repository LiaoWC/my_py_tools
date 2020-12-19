from PIL import Image
import numpy as np

# === Get image information ===


VALID_GET_IMG_INPUT_MODE = ['file_path', 'array', 'PIL_Image']
VALID_GET_IMG_RETURN_MODE = ['array', 'PIL_Image']

# Parameters:
#     input_mode:
#         - 'file_path': Get img information by opening the img file.
#         - 'array': Input the image in numpy array format.
#         - 'PIL_Image': Input the image in PIL Image format.
#     return_mode:
#         - 'array': Return the image in numpy array format.
#         - 'PIL_Image': Return the image in PIL Image format.
#     img:
#         Whether they should be given depends on the mode you choose.
#         e.g. If input mode is 'array', img should be a numpy array which represents the image.
def get_img(img,
            input_mode='file_path',
            return_mode='array'
            ):
    # Check if the two mode valid
    if input_mode not in VALID_GET_IMG_INPUT_MODE or return_mode not in VALID_GET_IMG_RETURN_MODE:
        raise ValueError('Invalid input mode or invalid return_mode.')
    # Get image
    img = img
    if input_mode == 'file_path':
        img = Image.open(img)
    elif input_mode == 'array':
        img = Image.fromarray(img)
    elif input_mode == 'PIL_Image':
        pass
    # Return image
    if return_mode == 'array':
        return np.array(img)
    elif return_mode == 'PIL_Image':
        return img
