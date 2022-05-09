
import os
import sys
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(base_dir)
import numpy as np
from Methodology.Traditional.PCAKmeans.algorithm import pca_k_means
from Methodology.Traditional.PCAKmeans.util import diff_image
from ImageRegistration.align_transform import Align
import imageio

def main():
    al = Align('DJI_20220506101539_0085_Z.JPG', 'DJI_20220506105215_0086_Z.JPG', threshold=1)
    before_img, after_img = al.align_img_patch()
    # before_img = imageio.imread('img1.jpg')[:, :, 0:3]
    # after_img = imageio.imread('img2.jpg')[:, :, 0:3]
    eig_dim = 10
    block_sz = 4

    diff_img = diff_image(before_img, after_img, is_abs=True, is_multi_channel=True)
    change_img = pca_k_means(diff_img, block_size=block_sz,
                             eig_space_dim=eig_dim)
    imageio.imwrite('PCAKmeans.png', change_img)


if __name__ == '__main__':
    main()
