import cv2


def resize_image(image, img_shape=(204, 204)):
    im = cv2.resize(image, img_shape)
    im = im.transpose((2, 0, 1))  # convert the image to RGBA
    # print(im.shape)
    return im
