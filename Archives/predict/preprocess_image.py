import cv2
import matplotlib.pyplot as plt


def get_image_from_url(img_url):
    img = cv2.imread(img_url)

    return img


def plot_image(img, img_file_path=None):
    print(img_file_path)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()


def resize_image(image, img_shape=(204, 204)):
    im = cv2.resize(image, img_shape)
    im = im.transpose((2, 0, 1))  # convert the image to RGBA
    # print(im.shape)
    return im
