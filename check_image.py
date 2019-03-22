import numpy as np
import functionalCV as FCV
import cv2

img = cv2.imread('/Users/chenyue21/Pictures/Lenna.jpg')

print(FCV._is_numpy_image(img))
newimg = FCV.resize(img,[8,8])
print(newimg.shape)
padimg = FCV.pad(newimg,(1,2,3,4),(1,2,3),padding_mode='edge')
print(padimg.shape)
print(padimg)
cropimg = FCV.crop(padimg,2,1,8,8)
print(cropimg.shape)
print(cropimg)
center_cropimg = FCV.center_crop(cropimg,4)
print(center_cropimg.shape)
print(center_cropimg)

recropimg = FCV.resized_crop(padimg,2,1,8,8,16)
print(recropimg.shape)
print(recropimg)

hfimg = FCV.hflip(recropimg)
print(hfimg)
vfimg = FCV.vflip(recropimg)
print('or',vfimg)

five = FCV.five_crop(vfimg,2)
# print(five)

ten = FCV.ten_crop(vfimg,2)
# print(ten)

brimg = FCV.adjust_brightness(vfimg,-100)
# print("br",brimg)

conimg = FCV.adjust_contrast(vfimg,0.5)
# print('con',conimg)

satimg = FCV.adjust_saturation(vfimg,0.1)
# print('sat', satimg)

gamimg = FCV.adjust_gamma(vfimg,0.5)
# print('gaimg', satimg)

roimg = FCV.rotate(vfimg,-45)
print(roimg[:,:,1])
print(roimg.shape)


print(FCV.to_tensor(img))
a = FCV.normalize_caffe(FCV.to_tensor(img), [104/255, 117/255, 123/255], [1,1,1])
print(a.size())
print(FCV.normalize_caffe(FCV.to_tensor(img), [104/255, 117/255, 123/255], [1,1,1]))
