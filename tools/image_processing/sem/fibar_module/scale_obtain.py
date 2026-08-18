# CODE for obtaining the scale from the SEM image #
# NB! x_coords from up to down, y_coords from left to right #
#!/usr/bin/python3
# libraries
import numpy as np
import cv2
import pytesseract
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from fibar_module.resnet import ResNet, BasicBlock

from tensorflow import keras

# pot_val, pot_unit, scale length in px
value_unit_scale = []
# list of units
pot_units = ["nm", "um"]
# pot scale values - should be updated if some new scales used!
pot_values = ["1", "2", "3", "4", "10", "20", "30", "100", "200", "400"]



import __main__
setattr(__main__, "ResNet", ResNet)
setattr(__main__, "BasicBlock", BasicBlock)

resnet_model = None
vgg_model = None
digit_backend = "resnet"


def load_digit_models(resnet_path, vgg_path, backend="resnet"):
    """Load user-supplied digit models used to read the SEM scale label."""
    global resnet_model, vgg_model, digit_backend
    digit_backend = backend
    if backend == "resnet":
        # The upstream checkpoint serializes the complete model object.
        resnet_model = torch.load(
            resnet_path, map_location=torch.device("cpu"), weights_only=False
        )
        resnet_model.eval()
        # Validate the companion model input without loading TensorFlow weights.
        with open(vgg_path, "rb") as handle:
            if handle.read(8) != b"\x89HDF\r\n\x1a\n":
                raise ValueError("The supplied VGG model is not an HDF5 file")
    elif backend == "vgg":
        vgg_model = keras.models.load_model(vgg_path, compile=False)
        # Confirm the supplied PyTorch checkpoint can be opened as requested.
        torch.load(resnet_path, map_location=torch.device("cpu"), weights_only=False)
    else:
        raise ValueError("Unknown digit recognition backend: %s" % backend)


# INSERT YOUR tesseract.exe PATH here in case having an error
# make sure tesseract OCR is installed too
# uncomment for lab pc
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def scale_obtain(file):

    """
    Function for obtaining the scale from the original SEM input image
    """

    img = cv2.imread(file, 0)

    value_unit_scale, four_contours = [None, None, None], [] 

    ###########################################################################
    # number and unit obtaining

    img[img < 10] = 0

    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)[1]

    result = img.copy()
    try:
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        #cv2.imshow("thresh", np.uint8(thresh))

        contours = contours[0] if len(contours) == 2 else contours[1]
        #print(contours)

        
        for contour in contours:
            if len(contour) == 4:
                four_contours.append(contour)
        
        #print(four_contours)
        try:
            if len(four_contours) != 0:
                big_contour = max(four_contours, key=cv2.contourArea)
            else: 
                big_contour = max(contours, key=cv2.contourArea)
        except ValueError: # no scale bar visible 
            return value_unit_scale
    except:
        return value_unit_scale
    
    #print(big_contour)

    # big_contour struc: [a][0][b], where a is the corner idx (runs from upper left + ↓ and upper right + ↓), b is either y [0] or x [1] coord
    # initial flattened big contour list y1 x1 y2 x2 etc.

    y_coords = big_contour.flatten()[::2]
    x_coords = big_contour.flatten()[1::2]

    # for drawing purposes
    if len(big_contour) > 4:
        big_contour = np.array([ [[min(y_coords), max(x_coords)]], [[min(y_coords), min(x_coords)]], [[max(y_coords), min(x_coords)]], [[max(y_coords), max(x_coords)]] ])

    cv2.drawContours(result, [big_contour], 0, (255,255,255), 5)

    #cv2.imshow("contours", np.uint8(result))

    # # 1px height cross-section
    cross = img[int( (min(x_coords)+ max(x_coords)) //2 ): int( (min(x_coords)+ max(x_coords)) //2 )+1, min(y_coords):max(y_coords)+5]

    # in case no scalebar has been found 
    try:

        ar_z = np.insert(np.nonzero(cross), 0, 0)
        # diff between non-zero values - fetching the zero amount of zeros and fetching the
        # start idx of the longest sub-arr of consecutive zeros
        min_val = ar_z[np.argmax(np.ediff1d(ar_z))]
        # end idx of the zero arr
        max_val = ar_z[np.argmax(np.ediff1d(ar_z)) +1]

        cutting_idx = int(np.mean((min_val, max_val)))


        # extracting unit
        unit = img[min(x_coords):max(x_coords), min(y_coords) + cutting_idx:max(y_coords)]
        unit = np.pad(unit, pad_width = [(1, 1),(1, 1)], mode = "constant")
        #cv2.imshow("unit", np.uint8(unit))

        # extracting number
        number = img[min(x_coords):max(x_coords), min(y_coords): min(y_coords) + cutting_idx]
        number = np.pad(number, pad_width = [(1, 1),(1, 1)], mode = "constant")
        #cv2.imshow("number", np.uint8(number))


        ## number to digits ## 
        first_row_of_whites = np.min(np.nonzero(number)[0])
        full_extent = number.shape[1]

        nr_cross = number[first_row_of_whites: first_row_of_whites+1, :full_extent].flatten()

        nonz_locs = np.nonzero(nr_cross)[0]

        nr_ar_z = np.insert(np.nonzero(nr_cross), 0, 0)
        diff_between_zeros = np.ediff1d(nr_ar_z) 

        idxs = np.where(diff_between_zeros > 1)[0]

        cuts = [int(np.mean((nonz_locs[val], nonz_locs[val-1]))) for val in idxs[idxs!=0]]
        
        cuts.insert(0,0)
        cuts.append(number.shape[1])
        custom_config= r'--psm 10' 

        compound_nr = ""
        for i, cut in enumerate(cuts):
            if i != len(cuts)-1:

                digit = number[:number.shape[0], cut:cuts[i+1]]
                digit = np.pad(digit, pad_width = [(1, 1),(1, 1)], mode = "constant")

                if digit_backend == "vgg":
                    im = cv2.resize(digit, (28, 28))
                    im = np.float32(im.reshape((1,) + im.shape + (1,)) / 255)
                    pred = np.argmax(vgg_model.predict(im, verbose=0))
                    # Upstream compatibility workaround: scale labels contain no 7s.
                    if pred == 7:
                        pred = 1
                else:
                    im = cv2.resize(digit, (27,27))
                    im = im.reshape((1,)+(1,)+im.shape)
                    im = im/255
                    im = torch.tensor(im)
                    im = im.to("cpu",  dtype=torch.float)
                    with torch.no_grad():
                        logits, probas = resnet_model(im)
                    pred = probas.cpu().data.numpy().argmax()
            

                compound_nr += str(pred)

        value_unit_scale[0] = int(compound_nr)
        # cv2.imshow("number", np.uint8(number))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # segmentation technique - one line with many possible char-s
        # https://github.com/madmaze/pytesseract
        # https://muthu.co/all-tesseract-ocr-options/
        # takes an image as one single character
        custom_config= r'--psm 10' 

        #try:


        for val in range(10,30, 1):
            detected_unit = str(pytesseract.image_to_string(cv2.resize(unit, (val, val)), config =custom_config, timeout = 5)).split("y\n\x0c")[0].strip() # Timeout after 2 seconds
            #print(detected_unit)
            if detected_unit in pot_units:
                value_unit_scale[1] = detected_unit

                break


        # except RuntimeError as timeout_error:
        # # Tesseract processing is terminated
        #     pass
        ################################################################

        # contouring the scale
        th2 = cv2.threshold(img[max(x_coords):, min(y_coords)-5: max(y_coords) + img.shape[1]//2], 254, 255, cv2.THRESH_BINARY)[1]

        ## line length in pixels
        # try:
        scale_length = np.max(np.nonzero(th2)[1]) - np.min(np.nonzero(th2)[1])
        value_unit_scale[2] = scale_length
            
        return value_unit_scale
        # # in case no unit or number is found
        # except:
        #     return None
    
    # if the biggest contour has been found in the wrong area => pixel based measurements
    except ValueError:
        return value_unit_scale
            
    
