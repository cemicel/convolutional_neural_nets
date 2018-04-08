import os
import cv2
from PIL import ImageFont, Image, ImageDraw
from strgen import StringGenerator
import strgen
import sys
import numpy as np


current_work_directory = os.getcwd()
lib_root = current_work_directory + '/fonts/'
numb_pic_per_class = None

img_height = 38
img_width = 176
text_size = 22


# custom dir is under developing
def create_dirs(fonts_tff, mode=None, custom_dir=None):
    fonts = []
    path_to_data = []
    for f in fonts_tff:
        fonts.append(f.split('.')[0])

    if not os.path.exists(current_work_directory + '/{}'.format(mode)):
        os.makedirs(current_work_directory + '/{}'.format(mode))

    for f in fonts:
        # create data in work directory
        if custom_dir is None:
            current_dir = os.getcwd()

            if not os.path.exists(current_dir + '/{}/{}'.format(mode, f)):
                os.makedirs(current_dir + '/{}/{}'.format(mode, f))
                path_to_data.append(current_dir + '/{}/{}'.format(mode, f))
            else:
                path_to_data.append(current_dir + '/{}/{}'.format(mode, f))
        # for custom directory
        else:
            if not os.path.exists(custom_dir + '/{}'.format(f)):
                os.makedirs(custom_dir + '/{}'.format(f))
                path_to_data.append(custom_dir + '/data/{}'.format(f))
            else:
                path_to_data.append(custom_dir + '/data/{}'.format(f))

    return path_to_data


def get_random_inputs(input_number=10, length=13, mode=None):
    """ generates random inpust
        if mode is training, generates minimum 5 images per class
        otherwise can generate one image per class

    """
    list_for_return = []
    str_patt = "[a-z]{" + str(length) + "}"
    dig_patt = "[0-9]{" + str(length) + "}"

    if mode == 'training':
        default_string = 'abcdefghijklmnopqrstuvwxyz' + 'abcdefghijklmnopqrstuvwxyz'.upper() + '0123456789'
        list_for_return = []
        div = int(len(default_string) / length)
        for i in range(div):
            list_for_return.append(default_string[i * length:(i + 1) * length])

        passed = length * div

        list_for_return.append(default_string[passed:])

        input_number = int(input_number)
        input_number -= len(list_for_return)

    for i in range(1, int(input_number) + 1):

        if i % 3 == 0:
            list_for_return.append(strgen.StringGenerator(str_patt).render().upper())
        elif i % 9 == 0:
            list_for_return.append(strgen.StringGenerator(dig_patt).render().upper())
        else:
            list_for_return.append(strgen.StringGenerator(str_patt).render().lower())

    return list_for_return


def save_element(font, text, index, path):
    img = np.zeros((img_height, img_width), np.uint8)
    p_image = Image.fromarray(img, mode='L')
    draw = ImageDraw.Draw(p_image)
    draw.text(xy=(5, 2), text=text, font=font, fill=255)
    # Display the image
    # cv2.imshow("img", np.array(p_image))
    # Save image
    image = np.array(p_image)
    cv2.imwrite(path + "/{}.png".format(index), image)

    cv2.waitKey(0)
    return image


def get_custom_pic(font, text):

    font = ImageFont.truetype(font, text_size)
    img = np.zeros((img_height, img_width), np.uint8)
    p_image = Image.fromarray(img, mode='L')
    draw = ImageDraw.Draw(p_image)
    draw.text(xy=(5, 2), text=text, font=font, fill=255)

    #cv2.imshow("img", np.array(p_image))
    #cv2.waitKeyEx(0)


    image = np.array(p_image)
    return image


def generate_data(fonts, mode=None):
    # get font paths from standard library


    # create directories for future data
    data_paths = create_dirs(fonts_tff=fonts, mode=mode)

    # and create in each directory
    for i in range(len(data_paths)):
        print(data_paths[i])

        font = ImageFont.truetype(fonts[i], text_size)
        # generate list of random data.
        # minimum 10
        random_data = get_random_inputs(numb_pic_per_class, mode=mode)
        for index_, element_ in enumerate(random_data):
            save_element(font, element_, index_, data_paths[i])


if __name__ == '__main__':


    try:

        fonts = os.listdir(lib_root)
        fonts = [i for i in fonts if i.split('.')[-1] == 'ttf']
        mode = sys.argv[1]
        numb_pic_per_class = sys.argv[2]
        print(fonts)


        generate_data(fonts, mode)

    except FileNotFoundError as f_err:
        print('fonts directory is missing')
    except IndexError  as i_err:

        mode = None
        if mode == 'training':
            numb_pic_per_class = 20
            generate_data(fonts, mode)
            print('Default number of training pictures per class: ', numb_pic_per_class)
        elif mode == 'test':
            numb_pic_per_class = 1
            generate_data(fonts, mode)
            print('Default number of test pictures per class: ', numb_pic_per_class)
        else:
            numb_pic_per_class = 20
            generate_data(fonts, 'training')
            print('Default number of training pictures per class: ', numb_pic_per_class)

            numb_pic_per_class = 1
            generate_data(fonts, 'test')
            print('Default number of test pictures per class: ', numb_pic_per_class)


    except OSError as os_err:
        print(os_err)
