import os
import cv2
from PIL import ImageFont, Image, ImageDraw
from strgen import StringGenerator
import strgen
import sys
import numpy as np
from optparse import OptionParser

current_work_directory = os.getcwd()
fonts = current_work_directory + '/fonts'
numb_pic_per_class = None

original = fonts + '/standard'
variety = fonts + '/styled'

img_height = 38
img_width = 176
text_size = 22


# custom dir is under developing
def create_dirs(fonts_tff, mode=None, underline=False):
    fonts = []
    path_to_data = []
    for f in fonts_tff:
        fonts.append(f.split('.')[0])

    if not os.path.exists(current_work_directory + '/{}'.format(mode)):
        os.makedirs(current_work_directory + '/{}'.format(mode))

    for f in fonts:
        # not underline
        if not underline:

            if not os.path.exists(current_work_directory + '/{}/{}'.format(mode, f)):
                os.makedirs(current_work_directory + '/{}/{}'.format(mode, f))
                path_to_data.append(current_work_directory + '/{}/{}'.format(mode, f))
            else:
                path_to_data.append(current_work_directory + '/{}/{}'.format(mode, f))
        else:

            # underline
            if not os.path.exists(current_work_directory + '/{}/{}'.format(mode, f + ' underline')):
                os.makedirs(current_work_directory + '/{}/{}'.format(mode, f + ' underline'))
                path_to_data.append(current_work_directory + '/{}/{}'.format(mode, f + ' underline'))
            else:
                path_to_data.append(current_work_directory + '/{}/{}'.format(mode, f + ' underline'))

    return path_to_data


def get_random_inputs(input_number=10, length=13, mode=None):
    """ generate random inpust
        if mode is training, generates minimum 5 images per class
        otherwise can generate one image per class

    """
    list_for_return = []
    str_patt = "[a-z]{" + str(length) + "}"
    dig_patt = "[0-9]{" + str(length) + "}"

    if mode == 'train':
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


# text position
text_position = [5, 2]


def save_element(font, text, index, path, underline=False):
    img = np.zeros((img_height, img_width), np.uint8)
    p_image = Image.fromarray(img, mode='L')
    draw = ImageDraw.Draw(p_image)

    if underline:
        # theight + 1 -> plus one more pixel between text and line
        twidth, theight = draw.textsize(text, font=font)
        lx, ly = text_position[0], text_position[1] + theight + 1

        draw.text(xy=(text_position[0], text_position[1]), text=text, font=font, fill=255)
        draw.line((lx, ly, lx + twidth, ly), fill=255)

        image = np.array(p_image)
        cv2.imwrite(path + "/{}.png".format(index), image)
    else:

        draw.text(xy=(text_position[0], text_position[1]), text=text, font=font, fill=255)

        image = np.array(p_image)

        cv2.imwrite(path + "/{}.png".format(index), image)
    # Display the image
    # cv2.imshow("img", np.array(p_image))
    cv2.waitKey(0)
    return image


def get_custom_pic(font, text):
    print(font)
    font = ImageFont.truetype(font, text_size)
    img = np.zeros((img_height, img_width), np.uint8)
    p_image = Image.fromarray(img, mode='L')
    draw = ImageDraw.Draw(p_image)
    draw.text(xy=(5, 2), text=text, font=font, fill=255)

    cv2.imshow("img", np.array(p_image))
    cv2.waitKeyEx(0)

    image = np.array(p_image)
    return image


def generate_data(fonts, mode=None, underline=False):
    # get font paths from standard library

    # create directories for future data

    if not underline:
        data_paths = create_dirs(fonts_tff=fonts, mode=mode)

        for i in range(len(data_paths)):
            print(data_paths[i])

            font = ImageFont.truetype(fonts[i], text_size)
            # generate list of random data.
            # minimum 10
            random_data = get_random_inputs(numb_pic_per_class, mode=mode)

            for index_, element_ in enumerate(random_data):
                save_element(font, element_, index_, data_paths[i])
    else:

        for is_under in range(2):

            data_paths = create_dirs(fonts_tff=fonts, mode=mode, underline=is_under)

            for i in range(len(data_paths)):
                print(data_paths[i])

                font = ImageFont.truetype(fonts[i], text_size)
                # generate list of random data.
                # minimum 10
                random_data = get_random_inputs(numb_pic_per_class, mode=mode)

                for index_, element_ in enumerate(random_data):
                    save_element(font, element_, index_, data_paths[i], underline=is_under)


if __name__ == '__main__':

    parser = OptionParser()

    parser.add_option('-m', '--mode',
                      dest='mode',
                      default='train', help='train,test,valid')

    parser.add_option('-n', '--numb_pic',
                      dest='numb_pic',
                      default='100')

    options, _ = parser.parse_args()

    try:

        # original will be duplicated with underline
        original_fonts = [i for i in os.listdir(original) if i.split('.')[-1] == 'ttf']
        variety_fonts = [i for i in os.listdir(variety) if i.split('.')[-1] == 'ttf']

        mode = options.mode
        numb_pic_per_class = options.numb_pic

        generate_data(original_fonts, mode, underline=True)
        generate_data(variety_fonts, mode, underline=False)



    except FileNotFoundError as f_err:
        print('directory is missing')

    except OSError as os_err:
        print(os_err)
