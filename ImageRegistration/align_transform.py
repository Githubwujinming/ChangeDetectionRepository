# Created by Qixun Qu
# quqixun@gmail.com
# 2017/04/11
#


import random
import cv2
import numpy as np
from ImageRegistration.affine_ransac import Ransac
from ImageRegistration.affine_transform import Affine


# The ration of the best match over second best match
#      distance of best match
# ------------------------------- <= MATCH_RATIO
#  distance of second best match
RATIO = 0.8


class Align():

    def __init__(self, source_path, target_path,
                 K=3, threshold=1):
        ''' __INIT__

            Initialize the instance.

            Input arguments:

            - source_path : the path of sorce image that to be warped 是之后拍的照片
            - target_path : the path of target image 是之前拍的照片
            - K : the number of corresponding points, default is 3
            - threshold : a threshold determins which points are outliers
            in the RANSAC process, if the residual is larger than threshold,
            it can be regarded as outliers, default value is 1

        '''

        self.source_path = source_path
        self.target_path = target_path
        self.K = K
        self.threshold = threshold

    def read_image(self, path, mode=1):
        ''' READ_IMAGE

            Load image from file path.

            Input arguments:

            - path : the image to be read
            - mode : 1 for reading color image, 0 for grayscale image
            default is 1

            Output:

            - the image to be processed

        '''

        # img =  cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
        # # img = cv2.resize(img, (700,600))
        # return img
        return cv2.imread(path, mode)

    def extract_SIFT(self, img):
        ''' EXTRACT_SIFT

            Extract SIFT descriptors from the given image.

            Input argument:

            - img : the image to be processed

            Output:

            -kp : positions of key points where descriptors are extracted
            - desc : all SIFT descriptors of the image, its dimension
            will be n by 128 where n is the number of key points


        '''

        # Convert the image to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Extract key points and SIFT descriptors
        sift = cv2.SIFT_create()
        kp, desc = sift.detectAndCompute(img_gray, None)

        # Extract positions of key points
        kp = np.array([p.pt for p in kp]).T

        return kp, desc

    def match_SIFT(self, desc_s, desc_t):
        ''' MATCH_SIFT

            Match SIFT descriptors of source image and target image.
            Obtain the index of conrresponding points to do estimation
            of affine transformation.

            Input arguments:

            - desc_s : descriptors of source image
            - desc_t : descriptors of target image

            Output:

            - fit_pos : index of corresponding points

        '''

        # Match descriptor and obtain two best matches
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc_s, desc_t, k=2)

        # Initialize output variable
        fit_pos = np.array([], dtype=np.int32).reshape((0, 2))

        matches_num = len(matches)
        for i in range(matches_num):
            # Obtain the good match if the ration id smaller than 0.8
            if matches[i][0].distance <= RATIO * matches[i][1].distance:
                temp = np.array([matches[i][0].queryIdx,
                                 matches[i][0].trainIdx])
                # Put points index of good match
                fit_pos = np.vstack((fit_pos, temp))

        return fit_pos

    def affine_matrix(self, kp_s, kp_t, fit_pos):
        ''' AFFINE_MATRIX

            Compute affine transformation matrix by corresponding points.

            Input arguments:

            - kp_s : key points from source image
            - kp_t : key points from target image
            - fit_pos : index of corresponding points

            Output:

            - M : the affine transformation matrix whose dimension
            is 2 by 3

        '''

        # Extract corresponding points from all key points
        kp_s = kp_s[:, fit_pos[:, 0]]
        kp_t = kp_t[:, fit_pos[:, 1]]

        # Apply RANSAC to find most inliers
        _, _, inliers = Ransac(self.K, self.threshold).ransac_fit(kp_s, kp_t)

        # Extract all inliers from all key points
        kp_s = kp_s[:, inliers[0]]
        kp_t = kp_t[:, inliers[0]]

        # Use all inliers to estimate transform matrix
        A, t = Affine().estimate_affine(kp_s, kp_t)
        M = np.hstack((A, t))

        return M

    def warp_image(self, source, target, M):
        ''' WARP_IMAGE

            Warp the source image into target with the affine
            transformation matrix.

            Input arguments:

            - source : the source image to be warped
            - target : the target image
            - M : the affine transformation matrix

        '''

        # Obtain the size of target image
        rows, cols, _ = target.shape

        # Warp the source image
        warp = cv2.warpAffine(source, M, (cols, rows))

        # Merge warped image with target image to display
        merge = np.uint8(target * 0.5 + warp * 0.5)

        # Show the result
        cv2.imwrite("results.jpg",merge)

        return

    def align_image(self):
        ''' ALIGN_IMAGE

            Warp the source image into target image.
            Two images' path are provided when the
            instance Align() is created.

        '''

        # Load source image and target image
        img_source = self.read_image(self.source_path)
        img_target = self.read_image(self.target_path)

        print("Extract key points and SIFT descriptors")
        # Extract key points and SIFT descriptors from
        # source image and target image respectively
        kp_s, desc_s = self.extract_SIFT(img_source)
        kp_t, desc_t = self.extract_SIFT(img_target)

        # Obtain the index of correcponding points
        print("Obtain the index of correcponding points")
        fit_pos = self.match_SIFT(desc_s, desc_t)
        print("Compute the affine transformation matrix")
        # Compute the affine transformation matrix
        M = self.affine_matrix(kp_s, kp_t, fit_pos)
        print("Warp the source image and display result")
        # Warp the source image and display result
        self.warp_image(img_source, img_target, M)

        return

    def align_img_patch(self):
        # Load source image and target image
        img_source = self.read_image(self.source_path)
        img_target = self.read_image(self.target_path)
        # 获取同名点
        p1, p2 = self.GetSamePoints(img_source, img_target, 600, 600)
        # 绘制同名点
        # drawImg = self.DrawSamePoint(img_source, img_target, p1, p2)
        # 配准
        T, _ = cv2.findHomography(p2, p1, cv2.RANSAC, 0.1)
        nimg2 = cv2.warpPerspective(img_target, T, (img_source.shape[1], img_source.shape[0]))
        # cv2.imwrite('samepoints.jpg', drawImg)
        # cv2.imwrite('result3.jpg', nimg2*0.5+img_source*0.5)
        return img_source, nimg2

    def align_images(self, img_source, img_target):
        # 获取同名点
        p1, p2 = self.GetSamePoints(img_source, img_target, 600, 600)
        # 绘制同名点
        drawImg = self.DrawSamePoint(img_source, img_target, p1, p2)
        # 配准
        T, _ = cv2.findHomography(p2, p1, cv2.RANSAC, 0.1)
        nimg2 = cv2.warpPerspective(img_target, T, (img_source.shape[1], img_source.shape[0]))
        return img_source, nimg2

    def GetSamePoints(self, img1, img2, patchheight=2000, patchwidth=2000):
        """
        使用SIFT算法获取同名点
        @img1 第一张影像
        @img2 第二张影像
        @return p1、p2分别为两张影像上点
        ps: 当两张影像过大时会进行分块
        """
        # 初始化sift
        # sift = cv2.xfeatures2d.SIFT_create(600)
        sift = cv2.SIFT_create(600)
        # 判断是否需要分块
        rows, cols = img1.shape[0:2]
        rownum = (1 if rows <= patchheight else rows // patchheight)
        colnum = (1 if cols <= patchwidth else cols // patchwidth)
        # 根据分块结果进行同名点匹配
        p1 = np.empty([0, 1, 2], dtype=np.float32)
        p2 = np.empty([0, 1, 2], dtype=np.float32)
        # 测试
        # badp1 = np.empty([0, 1, 2], dtype=np.float32)
        # badp2 = np.empty([0, 1, 2], dtype=np.float32)
        for i in range(rownum):
            for j in range(colnum):
                # 获取分块影像
                mimg1 = img1[i*patchheight:(i+1)*patchheight,
                            j*patchwidth:(j+1)*patchwidth]
                mimg2 = img2[i*patchheight:(i+1)*patchheight,
                            j*patchwidth:(j+1)*patchwidth]
                timg = np.r_[mimg1, mimg2]
                # 测试分块重叠区域是否足够
                # cv2.namedWindow('test', 0)
                # cv2.imshow('test', timg)
                # cv2.waitKey()
                # 提取特征点
                kp1, des1 = sift.detectAndCompute(mimg1, None)
                kp2, des2 = sift.detectAndCompute(mimg2, None)
                # 匹配
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des1, des2, k=2)
                good = []
                # 舍弃大于0.7的匹配,初步筛除
                for m, n in matches:
                    if m.distance < RATIO * n.distance:
                        good.append(m)
                
                # 剔除误匹配
                tp1 = np.float32([kp1[m.queryIdx].pt
                                for m in good]).reshape(-1, 1, 2)
                tp2 = np.float32([kp2[m.trainIdx].pt
                                for m in good]).reshape(-1, 1, 2)  
                M, mask = cv2.findHomography(tp1, tp2, cv2.RANSAC, 0.1)
                matchmask = mask.ravel().tolist()
                pnum = matchmask.count(1)
                mp1 = np.zeros([pnum, 1, 2], dtype=np.float32)
                mp2 = np.zeros([pnum, 1, 2], dtype=np.float32)
                iter = 0
                # 剔除误匹配的同时恢复分块点坐标至原影像
                for k in range(len(matchmask)):
                    if matchmask[k] == 1:
                        mp1[iter, 0, 0] = tp1[k, 0, 0] + j*patchwidth
                        mp1[iter, 0, 1] = tp1[k, 0, 1] + i*patchheight
                        mp2[iter, 0, 0] = tp2[k, 0, 0] + j*patchwidth
                        mp2[iter, 0, 1] = tp2[k, 0, 1] + i*patchheight
                        iter = iter + 1
                # 将每一块的同名点放到一起
                p1 = np.vstack((p1, mp1))
                p2 = np.vstack((p2, mp2))
                # 测试
        #         mbadp1 = tp1 + j*patchwidth
        #         mbadp2 = tp2 + i*patchheight
        #         badp1 = np.vstack((badp1, mbadp1))
        #         badp2 = np.vstack((badp2, mbadp2))
        # drawImg = DrawSamePoint(img1, img2, badp1, badp2)
        # cv2.imwrite('data/samepoints_bad.jpg', drawImg)
        return p1, p2
