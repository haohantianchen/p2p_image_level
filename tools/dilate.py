import cv2
import numpy as np

# 读取图像
image = cv2.imread('/home/jianshu/code/prompt_travel/stylize/warp_test/50_densepose_384_mask.png', cv2.IMREAD_GRAYSCALE)

# 定义膨胀核（kernel）
kernel = np.ones((5,5), np.uint8)  # 5x5的全白方块，用于膨胀操作

# 使用cv2.dilate进行膨胀操作
dilated_image = cv2.dilate(image, kernel, iterations=2)

# 显示原始图像和膨胀后的图像
cv2.imwrite("/home/jianshu/code/prompt_travel/stylize/warp_test/50_densepose_384_mask_dilate.png", dilated_image)
