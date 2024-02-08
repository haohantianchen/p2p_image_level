import cv2

in_img = "../stylize/warp_test/50_densepose_384.png"
out_img = "../stylize/warp_test/50_densepose_384_mask.png"
img = cv2.imread(in_img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.imwrite(out_img, img)
