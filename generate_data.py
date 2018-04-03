import os
import cv2
from PIL import ImageFont, Image, ImageDraw
from strgen import StringGenerator
import strgen
import sys
import numpy as np

lib_root = '/library/Fonts/'
current_work_directory = os.getcwd()
numb_pic_per_class = None

img_height = 28
img_width = 168
text_size = 17


# custom dir is under developing
def create_dirs(font_paths, mode=None, custom_dir=None):
    
    fonts = []
    path_to_data = []
    for font_name in font_paths:
        fonts.append(font_name.split("/")[-1].split('.')[0])

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


def get_font_paths(font_number, font_list=None):
    _, _, files = next(os.walk(lib_root))
    input_font_paths = []

    # is under developing
    if font_list is None:
        print('Type font names:')  # example: 'font_name'

        for font_index in range(font_number):
            inp = input()
            if inp + '.ttf' in files:
                input_font_paths.append(lib_root + inp + '.ttf')
            else:
                print('no {} in lib'.format(inp))
                return
    else:
        for font_index in font_list:
            input_font_paths.append(lib_root + font_index + '.ttf')

    return input_font_paths


def generate_data(font_name, mode=None):
    # get font paths from standard library
    font_paths_ = get_font_paths(len(font_name), font_list=font_name)

    #create directories for future data
    data_paths = create_dirs(font_paths=font_paths_, mode=mode)

    # iterate trough paths where data is stored
    # and create in each directory
    for i in range(len(data_paths)):
        print(data_paths[i])
        font = ImageFont.truetype(font_paths_[i], text_size)
        # generate list of random data.
        # minimum 10
        random_data = get_random_inputs(numb_pic_per_class, mode=mode)
        for index_, element_ in enumerate(random_data):
            save_element(font, element_, index_, data_paths[i])


if __name__ == '__main__':

    try:
        font_names_path = os.getcwd() + '/font_names.txt'


        with open(font_names_path) as f:
            line_ = f.read()
        mode = sys.argv[1]
        numb_pic_per_class = sys.argv[2]
        # generate data based on given fonts
        generate_data([elem.strip() for elem in line_.split(',')], mode)

    except FileNotFoundError as f_err:
        print('no such file: {}'.format(str(f_err.filename).split('/')[-1]))
    except IndexError  as i_err:
        print("ERROR input args: ", i_err)
    except OSError as os_err:
        print(os_err)
