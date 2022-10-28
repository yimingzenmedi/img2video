import cv2
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity


def image_align(deblurred, gt):
    # this function is based on kohler evaluation code
    z = deblurred
    x = gt

    zs = (np.sum(x * z) / np.sum(z * z)) * z  # simple intensity matching

    warp_mode = cv2.MOTION_HOMOGRAPHY
    warp_matrix = np.eye(3, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 100

    termination_eps = 0

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.cvtColor(zs, cv2.COLOR_RGB2GRAY),
                                             warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=5)

    target_shape = x.shape
    shift = warp_matrix

    zr = cv2.warpPerspective(
        zs,
        warp_matrix,
        (target_shape[1], target_shape[0]),
        flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT)

    cr = cv2.warpPerspective(
        np.ones_like(zs, dtype='float32'),
        warp_matrix,
        (target_shape[1], target_shape[0]),
        flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0)

    zr = zr * cr
    xr = x * cr

    return zr, xr, cr, shift


def compute_ssim(tar_img, prd_img):
    prd_img, tar_img, cr1, shift = image_align(prd_img, tar_img)
    ssim_pre, ssim_map = structural_similarity(tar_img, prd_img, multichannel=True, gaussian_weights=True,
                                               use_sample_covariance=False, data_range=1.0, full=True)
    ssim_map = ssim_map * cr1
    r = int(3.5 * 1.5 + 0.5)  # radius as in ndimage
    win_size = 2 * r + 1
    pad = (win_size - 1) // 2
    ssim = ssim_map[pad:-pad, pad:-pad, :]
    crop_cr1 = cr1[pad:-pad, pad:-pad, :]
    ssim = ssim.sum(axis=0).sum(axis=0) / crop_cr1.sum(axis=0).sum(axis=0)
    ssim = np.mean(ssim)
    return ssim


def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions:', img1.shape, img2.shape)
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    # print(">> img1.ndim:", img1.ndim)
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        # print(">> img1.shape:", img1.shape)
        if img1.shape[0] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[0] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    # print(">>> SSIM")

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.cpu().numpy().astype(np.float64)
    img2 = img2.cpu().numpy().astype(np.float64)

    # print(f"img1: {img1}")
    kernel = cv2.getGaussianKernel(11, 1.5)
    # print("kernel:", kernel)
    window = np.outer(kernel, kernel.transpose())
    # print("window:", window)

    mu1 = cv2.filter2D(img1, -1, window)  # valid
    # print("mu1:", mu1, mu1.shape)
    mu1 = mu1[:, 5:-5, 5:-5]
    # print("mu1_:", mu1, mu1.shape)
    mu2 = cv2.filter2D(img2, -1, window)[:, 5:-5, 5:-5]
    # print("mu2:", mu2)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[:, 5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[:, 5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[:, 5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    # print(">> SSIM_map:", ssim_map)
    return ssim_map.mean()


def torchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
    rmse = (imdff ** 2).mean().sqrt()
    ps = 20 * torch.log10(1 / rmse)
    return ps


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def numpyPSNR(tar_img, prd_img):
    imdff = np.float32(prd_img) - np.float32(tar_img)
    rmse = np.sqrt(np.mean(imdff ** 2))
    ps = 20 * np.log10(255 / rmse)
    return ps


def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img
