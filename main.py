## Céleste Duguay
## 300122287
## CSI 4533 - Laboratoire 2


import numpy as np
import cv2


##mask = np.zeros(img.shape[:2],np.uint8) from demo  https://docs.opencv.org/3.4/d8/d83/tutorial_py_grabcut.html

def grabcut_program(img_path, mask):
    # Import my image.
    #this will read our image
    
    myimg = cv2.imread(img_path)

    #if cannot read image path.
    #based on demo code
    assert myimg is not None, "file could not be read, check with os.path.exists()"

    #Create a mask for foreground and background
    #The goal of this is to create a 2D NumPy matrix of zeros.
    #image.shape takes the heigth and width dims of image
    bin_mask = np.zeros(myimg.shape[:2], np.uint8)
    bin_mask[mask] = 1


    #from the demo provided!
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    #Grabcut Algorithm implementation

    #Setting the number of subjects to extract --
    num_subjects = 3
    subject_rects = []

    #using roi for better manual selection from the lab description
    #using roi = Regions of Interest is a fomr of hardcoding lets manually select the image
    #
    for i in range(num_subjects):
        rect = cv2.selectROI(myimg)
        subject_rects.append(rect)

        #Apply the Grabcut Algorithm on our ROI!
        cv2.grabCut(myimg, bin_mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

        #Let's see our results after each segmentation
        results = myimg * np.where((bin_mask == 2) | (bin_mask == 0), 0, 1)[:, :, np.newaxis]
        # result display
        results = results.astype(np.uint8)
        cv2.imshow(f'my result after ROI {i + 1}', results)
        cv2.waitKey(0)

    # Modify masks --> to binary
    mask_res = np.where((bin_mask == 2) | (bin_mask == 0), 0, 1).astype('uint8')

    #apply masks to original images.
    result = myimg * mask_res[:, :, np.newaxis]


    return result

#loading my imgaes
myimg1 = 'img1.png'
mask_1 = np.array([[0, 0, 0, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0]], dtype=np.uint8)

myimg2 = 'img2.png'
mask_2 = np.array([[0, 0, 0, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0]], dtype=np.uint8)

result_1 = grabcut_program(myimg1, mask_1)
result_2 = grabcut_program(myimg2, mask_2)

#display
cv2.imshow('My result 1', result_1)
cv2.imshow('My result 2', result_2)
cv2.waitKey(0)
cv2.destroyAllWindows()