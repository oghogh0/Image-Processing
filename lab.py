"""
6.1010 Spring '23 Lab 2: Image Processing 2
"""

#!/usr/bin/env python3

# NO ADDITIONAL IMPORTS!
# (except in the last part of the lab; see the lab writeup for details)
import math
from PIL import Image


# VARIOUS FILTERS
def split_color(image):
    """
    Splits image into red, green, blue
    """
    image_pix = image['pixels'].copy()
    red_pix=[]
    green_pix=[]
    blue_pix=[]
    red_im= {      #red image dict
        'height': image['height'],
        'width': image['width']
    }
    green_im = {   #green image dict
        'height': image['height'],
        'width': image['width']
    }
    blue_im = {    #blue image dict
        'height': image['height'],
        'width': image['width']
    }

    for i, val in enumerate(image_pix):
        red_pix.append(val[0]) #indx 0 in tuple is red val
        green_pix.append(val[1])
        blue_pix.append(val[2])


    red_im['pixels']=red_pix
    green_im['pixels']=green_pix
    blue_im['pixels']=blue_pix

    return red_im, green_im, blue_im

def recombine_greyscale(image1, image2, image3):
    """
    New image with pixels of images combined
    """
    im1_pix= image1['pixels'].copy()
    im2_pix=image2['pixels'].copy()
    im3_pix=image3['pixels'].copy()
    recomb_pix=[]
    recomb_im={
        'height': image1['height'],
        'width': image1['width']}

    for i,val in enumerate(im1_pix):
        newtup=(val, im2_pix[i], im3_pix[i])
        recomb_pix.append(newtup)

    # for i in range(len(im1_pix)):
    #     newtup=(im1_pix[i], im2_pix[i], im3_pix[i]) #tuple of (r,g,b) back together
    #     recomb_pix.append(newtup)

    recomb_im['pixels']=recomb_pix
    return recomb_im


def color_filter_from_greyscale_filter(filt):
    """
    Args: filter that takes a greyscale image as input
    and produces a greyscale image as output
    Returns: function that takes a color image as input and
    produces the filtered color image.
    """
    def filtered_color_im(color_im):
        split_im = split_color(color_im)  #splits image into r,g,b

        filt_im1 = filt(split_im[0])  #applies filter to red imag
        filt_im2 = filt(split_im[1])  #applies filter to green imag
        filt_im3 = filt(split_im[2])  #" " blue img

        final_im = recombine_greyscale(filt_im1, filt_im2, filt_im3) #recombs filtimgs
        return final_im
    return filtered_color_im


def make_blur_filter(kernel_size):
    """
    Args: kernel size
    Returns: blur filter (arg: single image)
    """
    def blur_filt(image):
        return blurred(image, kernel_size)
    return blur_filt



def make_sharpen_filter(kernel_size):
    """
    Args: kernel size
    Returns: sharpen filter (arg: single image)
    """
    def sharpen_filt(image):
        return sharpened(image, kernel_size)
    return sharpen_filt


def filter_cascade(filters):
    """
    Given a list of filters (implemented as functions on images), returns a new
    single filter such that applying that filter to an image produces the same
    output as applying each of the individual ones in turn.
    """
    def func_image(image):
        if len(filters) == 1:  #checks if 1 filt
            change_im = filters[0](image)
            return change_im
        elif len(filters) > 1:
            change_im = filters[0](image)
            for i in range(1,len(filters)):
                change_im = filters[i](change_im) #applies multiple filters in loop
            return change_im
    return func_image
# SEAM CARVING

# Main Seam Carving Implementation


def seam_carving(image, ncols):
    """
    Starting from the given image, use the seam carving technique to remove
    ncols (an integer) columns from the image. Returns a new image.
    """
    grey_img = greyscale_image_from_color_image(image)
    energy_img = compute_energy(grey_img)
    cum_img = cumulative_energy_map(energy_img)
    min_energy_img = minimum_energy_seam(cum_img)
    seamless_img = image_without_seam(image,min_energy_img)

    for i in range(1,ncols):
        grey_img = greyscale_image_from_color_image(seamless_img)
        energy_img = compute_energy(grey_img)
        cum_img = cumulative_energy_map(energy_img)
        min_energy_img = minimum_energy_seam(cum_img)
        seamless_img = image_without_seam(seamless_img,min_energy_img)

    return seamless_img


# Optional Helper Functions for Seam Carving


def greyscale_image_from_color_image(image):
    """
    Given a color image, computes and returns a corresponding greyscale image.

    Returns a greyscale image (represented as a dictionary).
    """
    color_pixel = image['pixels'].copy()
    grey_image = {
        'height': image['height'],
        'width': image['width']
    }
    grey_pixels=[]
    for i in color_pixel:
        greyscale= round((.299*i[0]) + (.587*i[1]) + (.114*i[2])) #formula
        grey_pixels.append(greyscale)
    grey_image['pixels'] = grey_pixels

    return grey_image

def compute_energy(grey):
    """
    Given a greyscale image, computes a measure of "energy", in our case using
    the edges function from last week.

    Returns a greyscale image (represented as a dictionary).
    """
    energy_map = edges(grey)
    return energy_map


def cumulative_energy_map(energy):
    """
    Given a measure of energy (e.g., the output of the compute_energy
    function), computes a "cumulative energy map" as described in the lab 2
    writeup.

    Returns a dictionary with 'height', 'width', and 'pixels' keys (but where
    the values in the 'pixels' array may not necessarily be in the range [0,
    255].
    """

    energy_pix = energy['pixels'].copy()
    cumulative_img = {
        'height': energy['height'],
        'width': energy['width'],
        'pixels': energy_pix
    }

    for x in range(1, energy['height']): #row
        for y in range(energy['width']): #col
            cum_lst = []
            pix=get_pixel(cumulative_img, x, y, 'zero')

            row_up = x-1 #gets adjacent 3 pix row & col
            col_right = y+1
            col_left = y-1

            up_pix = get_pixel(cumulative_img, row_up, y)
            right_pix = get_pixel(cumulative_img, row_up, col_right)
            left_pix = get_pixel(cumulative_img, row_up, col_left)

            if y==0: #edge cases
                cum_lst.append(up_pix)
                cum_lst.append(right_pix)

            elif y==(energy['width']-1):
                cum_lst.append(up_pix)
                cum_lst.append(left_pix)

            else:
                cum_lst.append(up_pix)
                cum_lst.append(right_pix)
                cum_lst.append(left_pix)

            min_pix = min(cum_lst) #minimum adj
            new_pix = min_pix + pix #add
            set_pixel(cumulative_img, x, y, new_pix)
    return cumulative_img




def minimum_energy_seam(cem):
    """
    Given a cumulative energy map, returns a list of the indices into the
    'pixels' list that correspond to pixels contained in the minimum-energy
    seam (computed as described in the lab 2 writeup).
    """
    min_cem_pix = cem['pixels'].copy()
    seam_img = {
        'height': cem['height'],
        'width': cem['width'],
        'pixels': min_cem_pix
    }

    last_row_num = seam_img['height']-1 #indx
    last_row_pix = seam_img['pixels'][cem['width']*last_row_num:].copy()
    min_val = last_row_pix[0]
    min_col=0

    indx_list=[]

    for i in range(1,seam_img['width']):
        if last_row_pix[i] < min_val:
            min_val = last_row_pix[i]
            min_col = i
    indx_list.append((seam_img['width'] * last_row_num) + min_col)

    for x in range(last_row_num, 0, -1):
        min_dict={}
        min_lst=[]

        col_left = min_col-1
        row_up = x-1
        col_right = min_col+1

        col_left_pix= get_pixel(seam_img, row_up, col_left)
        row_up_pix = get_pixel(seam_img, row_up, min_col)
        col_right_pix = get_pixel(seam_img, row_up, col_right)


        if min_col == 0 : #dict to access col
            min_lst.append(row_up_pix)
            min_lst.append(col_right_pix)
            min_dict={col_right_pix: col_right,
                      row_up_pix: min_col}
        elif min_col == seam_img['width']-1:
            min_lst.append(col_left_pix)
            min_lst.append(row_up_pix)
            min_dict={row_up_pix: min_col,
                      col_left_pix: col_left}
        else:
            min_lst.append(col_left_pix)
            min_lst.append(row_up_pix)
            min_lst.append(col_right_pix)
            min_dict={col_right_pix: col_right,
                  row_up_pix: min_col,
                  col_left_pix: col_left}


        min_pix = min(min_lst)
        min_indx = (seam_img['width'] * row_up) + min_dict[min_pix]
        indx_list.append(min_indx)
        min_col = min_dict[min_pix]

    return indx_list




def image_without_seam(image, seam):
    """
    Given a (color) image and a list of indices to be removed from the image,
    return a new image (without modifying the original) that contains all the
    pixels from the original image except those corresponding to the locations
    in the given list.
    """
    new_image_pix = image['pixels'].copy()
    new_image = {
        'height': image['height'],
        'width': image['width']-1,   #width -1
    }
    seamless_lst=[]

    for i,val in enumerate(new_image_pix):
        if i not in seam:
            seamless_lst.append(val)


    new_image['pixels']= seamless_lst #new lst of pix

    return new_image



def custom_feature(kernel_size, start, radius, step, color):
    """
    Draws multiple circles with varying radius
    in range(start, radius, step) above right-angle triangle
    Kernel_size is odd and square
    Returns new image
    """
    new_image = {
        'height': kernel_size,
        'width': kernel_size,
        'pixels': [0]*(kernel_size**2)}
    
    x = kernel_size//2 #centre of kernel
    y = kernel_size//2

    #circle
    for rad in range(start, radius, step):
        for x_p in range(new_image['height']):
            for y_p in range(new_image['width']):
                dist = ((x_p - x)**2+(y_p - y)**2)**(1/2)
                if abs(dist-rad) < 0.5:
                    set_pixel(new_image, x_p, y_p, color)

    #triangle
    for i in range(0, (kernel_size+1)//2):
        x_val = x - i
        y_val = y + i
        # print(x_val, y_val)
        set_pixel(new_image, x_val, y_val, color)
        set_pixel(new_image, kernel_size//2, y_val, color)
        x_val = x + i
        y_val = y - i
        set_pixel(new_image, x_val, y_val, color)
        set_pixel(new_image, kernel_size//2, y_val, color)
    
    return new_image

    
    
    


#IMAGE PROCESSING 1 FNC

def get_pixel(image, row, col, boundary_behavior='zero'):
    """
    Gets value at index
    Args:
        image: dict
        row: int
        col: int
        boundary_behavior: str
    Returns:
        image["pixels"][indx]: int
    """
    if row in range(image['height']) and col in range(image['width']):  # inboud
        scale = int((image['width'] * row) + col)  # formula
        return image['pixels'][scale]  # gets pixel at indx
    else:
        if boundary_behavior == 'zero':
            return 0
        elif boundary_behavior == 'extend':
            if row < 0:
                row = 0
            elif row > image['height'] - 1:
                row = image['height'] - 1
            if col < 0:
                col = 0
            elif col > image['width'] - 1:
                col = image['width'] - 1
            scale = image['width'] * row + col  # formula
            return image['pixels'][scale]  # gets pixel at indx
        else:  # wrap
            if row < 0 or row > image['height'] - 1:
                row = row % image['height']

            if col < 0 or col > image['width'] - 1:
                col = col % image['width']

            scale = image['width'] * row + col  # formula
            return image['pixels'][scale]  # gets pixel at indx


def set_pixel(image, row, col, color):
    """
    Changes colour of pixel
    Args:
        image: dict
        row: int
        col: int
        color: int
    Returns:
        None
    """
    scale = image['width'] * row + col
    image['pixels'][scale] = color  # new colour


def apply_per_pixel(image, func):
    """
    Applies function per pixel
    Args:
        image: dict
    Returns:
        result: new dict
    """
    pixel_copy = image['pixels'].copy()
    result = {
        'height': image['height'],
        'width': image['width'],
        'pixels': pixel_copy,
    }  # dict copy

    for col in range(image['width']):
        for row in range(image['height']):
            color = get_pixel(image, row, col)
            new_color = func(color)
            set_pixel(result, row, col, new_color)
    return result


def inverted(image):
    """
    Inverts image
    Args:
        image: dict with keys "height",
            "width","pixels"
    Returns:
        result: new dict with "pixels" inverted
    """
    return apply_per_pixel(image, lambda color: 255 - color)



def correlate(image, kernel, boundary_behavior):
    """
    Compute the result of correlating the given image with the given kernel.
    `boundary_behavior` be will one of the strings "zero", "extend", or "wrap",
    and this function will treat out-of-bounds pixels as having the value zero,
    the value of the nearest edge, or the value wrapped around the other edge
    of the image, respectively.

    if boundary_behavior is not one of "zero", "extend", or "wrap", return
    None.

    Otherwise, the output of this function should have the same form as a 6.101
    image (a dictionary with "height", "width", and "pixels" keys), but its
    pixel values do not necessarily need to be in the range [0,255], nor do
    they need to be integers (they should not be clipped or rounded at all).

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.

    DESCRIBE YOUR KERNEL REPRESENTATION HERE
    kernel: {height: ,
           width: ,
           pixels: ,
    }
    """
    pixel_copy = []
    kernel_size = kernel['width']
    centre = (kernel_size - 1) / 2  # kernel centre
    for x in range(image['height']):
        for y in range(image['width']):
            tot = []
            t_row, t_col = centre, centre  # coord centre
            for k in range(len(kernel['pixels'])):
                kernel_x, kernel_y = (
                    k // kernel_size,
                    k % kernel_size,
                )  # coordinate of pixel
                row_rel, col_rel = (
                    kernel_x - t_row,
                    kernel_y - t_col,
                )  # centre to other pixel
                row, col = (x + row_rel, y + col_rel)  # coord of image pixel
                # print(row,col)
                image_pix = get_pixel(image, int(row), int(col), boundary_behavior)
                scale_pix = image_pix * kernel['pixels'][k]  # mult
                tot.append(scale_pix)
            pixel_copy.append(sum(tot))  # sum

    image_copy = {
        'height': image['height'],
        'width': image['width'],
        'pixels': pixel_copy,
    }
    # print(image_copy)
    return image_copy


def round_and_clip_image(image):
    """
    Given a dictionary, ensure that the values in the "pixels" list are all
    integers in the range [0, 255].

    All values should be converted to integers using Python's `round` function.

    Any locations with values higher than 255 in the input should have value
    255 in the output; and any locations with values lower than 0 in the input
    should have value 0 in the output.
    """
    pixels_list = image['pixels'].copy()
    round_pixels_list = []
    for i in pixels_list:
        if i < 0:
            i = 0
        elif i > 255:
            i = 255
        if not isinstance(i,int): #use isinstance not type
            i = round(i)
        round_pixels_list.append(i)

    fix_image = {
        'height': image['height'],
        'width': image['width'],
        'pixels': round_pixels_list,
    }
    return fix_image


# FILTERS
def box(n):
    """
    Returns nxn box kernel of identical values sum to 1
    """
    square_pix = [1 / (n * n)] * (n * n)
    kernel = {'height': n, 'width': n, 'pixels': square_pix}

    return kernel


def blurred(image, kernel_size):
    """
    Return a new image representing the result of applying a box blur (with the
    given kernel size) to the given input image.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    """
    # first, create a representation for the appropriate n-by-n kernel (you may
    # wish to define another helper function for this)

    # then compute the correlation of the input image with that kernel

    # and, finally, make sure that the output is a valid image (using the
    # helper function from above) before returning it.
    kernel = box(kernel_size)
    blurred_image = correlate(image, kernel, 'extend')
    # print(blurred_image)
    return round_and_clip_image(blurred_image)


def sharpened(image, n):
    """
    Compute blurred image and for each pixel compute s=2*I - B
    """
    blurred_image = blurred(image, n)  # blurred
    sharpen_lst = []
    for i in range(len(blurred_image['pixels'])):
        sharpen_pix = (2 * image['pixels'][i]) - blurred_image['pixels'][i]
        sharpen_lst.append(round(sharpen_pix))
    blurred_image['pixels'] = sharpen_lst  # new values
    return round_and_clip_image(blurred_image)  # round and clip in range


def edges(image):
    """
    Compute image_1=correlation(image,k_row)
            image_2=correlation(image,k_col)
    Return new image with each corresponding index with Sobel operator
    """
    k_row = {'height': 3, 'width': 3, 'pixels': [-1, -2, -1, 0, 0, 0, 1, 2, 1]}
    k_col = {'height': 3, 'width': 3, 'pixels': [-1, 0, 1, -2, 0, 2, -1, 0, 1]}
    row_image = correlate(image, k_row, 'extend')
    col_image = correlate(image, k_col, 'extend')
    sobel_list = []
    for i in range(len(image['pixels'])):
        sobel_pix = ((row_image['pixels'][i]) ** 2 +
                     (col_image['pixels'][i]) ** 2) ** ( 1 / 2)
        sobel_list.append(sobel_pix)

    edges_img = {
        'height': image['height'],
        'width': image['width'],
        'pixels': sobel_list,
    }
    return round_and_clip_image(edges_img)



# HELPER FUNCTIONS FOR LOADING AND SAVING COLOR IMAGES


def load_color_image(filename):
    """
    Loads a color image from the given file and returns a dictionary
    representing that image.

    Invoked as, for example:
       i = load_color_image('test_images/cat.png')
    """
    with open(filename, 'rb') as img_handle:
        img = Image.open(img_handle)
        img = img.convert('RGB')  # in case we were given a greyscale image
        img_data = img.getdata()
        pixels = list(img_data)
        width, height = img.size
        return {'height': height, 'width': width, 'pixels': pixels}


def save_color_image(image, filename, mode='PNG'):
    """
    Saves the given color image to disk or to a file-like object.  If filename
    is given as a string, the file type will be inferred from the given name.
    If filename is given as a file-like object, the file type will be
    determined by the 'mode' parameter.
    """
    out = Image.new(mode='RGB', size=(image['width'], image['height']))
    out.putdata(image['pixels'])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns an instance of this class
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_greyscale_image('test_images/cat.png')
    """
    with open(filename, 'rb') as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith('RGB'):
            pixels = [
                round(0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]) for p in img_data
            ]
        elif img.mode == 'LA':
            pixels = [p[0] for p in img_data]
        elif img.mode == 'L':
            pixels = list(img_data)
        else:
            raise ValueError(f'Unsupported image mode: {img.mode}')
        width, height = img.size
        return {'height': height, 'width': width, 'pixels': pixels}


def save_greyscale_image(image, filename, mode='PNG'):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the 'mode' parameter.
    """
    out = Image.new(mode='L', size=(image['width'], image['height']))
    out.putdata(image['pixels'])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


if __name__ == '__main__':
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place for
    # generating images, etc.

    # hello = load_color_image('test_images/cat.png')
    # color_filt_img_fnc = color_filter_from_greyscale_filter(inverted)
    # color_filt_img = color_filt_img_fnc(hello)
    # save_color_image(color_filt_img, "color_filt_cat.png")

    # hello = load_color_image('test_images/python.png')
    # color_filt_img_fnc = color_filter_from_greyscale_filter(make_blur_filter(9))
    # color_filt_img = color_filt_img_fnc(hello)
    # save_color_image(color_filt_img, "color_filt_blur_python.png")

    # hello = load_color_image('test_images/sparrowchick.png')
    # color_filt_img_fnc = color_filter_from_greyscale_filter(make_sharpen_filter(7))
    # color_filt_img = color_filt_img_fnc(hello)
    # save_color_image(color_filt_img, "color_filt_sharpen_sparrowchick.png")

    # hello = load_color_image('test_images/frog.png')
    # filter1 = color_filter_from_greyscale_filter(edges)
    # filter2 = color_filter_from_greyscale_filter(make_blur_filter(5))
    # filt = filter_cascade([filter1, filter1, filter2, filter1])
    # cascade_img = filt(hello)
    # save_color_image(cascade_img, "cascade_filt_frog.png")

    # hello = load_color_image('test_images/twocats.png')
    # seam_carved_img = seam_carving(hello, 100)
    # save_color_image(seam_carved_img, "seam_carved_cats.png")

    # hello = load_color_image('test_images/sparrowchick.png')
    # custom_img = custom_feature(hello, 2, 2)
    # save_color_image(custom_img, "custom_sparrowchick.png")

    # hello = load_color_image('test_images/mushroom.png')
    # custom_im = custom_feature(100, 0, 50, 2, 255)
    # # cust = custom_feature(grey_im,2,2)
    # # grey_img = color_filter_from_greyscale_filter(cust)
    # save_greyscale_image(custom_im, "custom_image.png")

    hello = load_color_image('test_images/flood_input_copy.png')
    print(hello['width'], hello['height'])
    # custom_im = custom_feature(100, 0, 50, 2, 255)
    # # cust = custom_feature(grey_im,2,2)
    # # grey_img = color_filter_from_greyscale_filter(cust)
    # save_greyscale_image(custom_im, "custom_image.png")