import cv2

from realsense_wapper import realsense

def get_obj_bbox(img):
    crop_offset_col = 300
    crop_offset_row = 100

    crop = img[crop_offset_row:, crop_offset_col:-60]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (4,4))
    _, binary = cv2.threshold(blur, 200, 256, cv2.THRESH_BINARY)
    # cntrs, _ = cv2.findContours(binary, 1, 2)
    # cnt = cntrs[0]

    x,y,w,h = cv2.boundingRect(binary)

    x += crop_offset_col
    y += crop_offset_row 

    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    #cv2.imshow('original', img)
    #cv2.imshow('crop', crop)
    # cv2.imshow('Gray image', gray)
    # cv2.imshow('Blur', blur)
    #cv2.imshow('binary', binary)
    #cv2.imshow('bbox', img)

    #cv2.waitKey(10)

    return x, y, w, h

def check_gripper_bbox(img):
    crop_offset_col = 300
    crop_offset_row = 100

    crop = img[crop_offset_row:crop_offset_row+100, crop_offset_col:-60]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (4,4))
    _, binary = cv2.threshold(blur, 200, 256, cv2.THRESH_BINARY)
    # cntrs, _ = cv2.findContours(binary, 1, 2)
    # cnt = cntrs[0]

    x,y,w,h = cv2.boundingRect(binary)

    x += crop_offset_col
    y += crop_offset_row 

    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    #cv2.imshow('original', img)
    #cv2.imshow('crop', crop)
    # cv2.imshow('Gray image', gray)
    # cv2.imshow('Blur', blur)
    #cv2.imshow('binary', binary)
    #cv2.imshow('bbox', img)

    #cv2.waitKey(10)

    return x, y, w, h

if __name__ == '__main__':

    cam = realsense()

    while(True):
        _, img = cam.get_frame_cv()

        crop_offset_col = 300
        crop_offset_row = 100

        crop = img[crop_offset_row:, crop_offset_col:-60]

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.blur(gray, (4,4))
        _, binary = cv2.threshold(blur, 200, 256, cv2.THRESH_BINARY)
        # cntrs, _ = cv2.findContours(binary, 1, 2)
        # cnt = cntrs[0]

        x,y,w,h = cv2.boundingRect(binary)

        x += crop_offset_col
        y += crop_offset_row 

        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        print("bbox: {}, {}, {}, {}".format(x,y,w,h))

        #cv2.imshow('original', img)
        cv2.imshow('crop', crop)
        # cv2.imshow('Gray image', gray)
        # cv2.imshow('Blur', blur)
        cv2.imshow('binary', binary)
        cv2.imshow('bbox', img)

        cv2.waitKey(10)
        #cv2.destroyAllWindows()
