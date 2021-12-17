import os, cv2
label_path = 'D:/x/'
out = 'D:/2/'
labels = os.listdir(label_path)
for label in labels:
    name = label.split('.')[0]
    lb_path = os.path.join(label_path, label)
    lb = cv2.imread(lb_path)
    lb = cv2.resize(lb, (256, 256), interpolation = cv2.INTER_NEAREST)
    #lb[lb>0] = 255
    cv2.imwrite(os.path.join(out, name+'.png'),lb , [int(cv2.IMWRITE_PNG_COMPRESSION), 0]) #注意一下存出来的图像质量喔，不要损失信息
