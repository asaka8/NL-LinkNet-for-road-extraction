import cv2

image = cv2.imread('label.png')  # 读入图片
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 二值化函数
cv2.threshold(image, 2, 255, 0, image)  # 二值化函数

cv2.namedWindow("Image")  # 图片显示框的名字 这行没啥用
cv2.imshow("Image", image)  # 图片显示

cv2.waitKey(0)
cv2.imwrite('1.png', image)  # 保存当前灰度值处理过后的文件
