import os

import cv2

input = "/raid/cvg_data/lurenjie/P2P/prompt_travel/stylize/warp_test/output_classify_body_a085/50.png"
output = "/raid/cvg_data/lurenjie/P2P/prompt_travel/stylize/warp_test/output_classify_body_a085/50_512.png"
size = (512, 512)
img = cv2.imread(input)
img = cv2.resize(img, size)
cv2.imwrite(output, img)
