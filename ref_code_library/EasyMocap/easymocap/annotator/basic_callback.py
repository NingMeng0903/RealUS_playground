'''
  @ Date: 2021-04-21 14:18:50
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-08-09 19:56:43
  @ FilePath: /EasyMocapPublic/easymocap/annotator/basic_callback.py
'''
import cv2

class CV_KEY:
    BLANK = 32
    ENTER = 13
    LSHIFT = 225 # does not work on Mac
    NONE = 255
    TAB = 9
    q = 113
    ESC = 27
    BACKSPACE = 8
    WINDOW_WIDTH = int(1920*0.8)
    WINDOW_HEIGHT = int(1080*0.8)
    LEFT = ord('a')
    RIGHT = ord('d')
    UP = ord('w')
    DOWN = ord('s')
    MINUS = 45
    PLUS = 61

def get_key():
    k = cv2.waitKey(10) & 0xFF
    if k == CV_KEY.LSHIFT:
        key1 = cv2.waitKey(500) & 0xFF
        if key1 == CV_KEY.NONE:
            return key1
        # Convert to uppercase
        k = key1 - ord('a') + ord('A')
    return k

def point_callback(event, x, y, flags, param):
    """
        Simple OpenCV callback that implements three basic behaviors:
        1. For click-and-drag, record start and end (current) points
        2. For clicks, record the selected point
        3. Track whether a mouse button is currently held
    """
    if event not in [cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP]:
        return 0
    param['button_down'] = flags == cv2.EVENT_FLAG_LBUTTON
    # Write the selected point position directly
    if event == cv2.EVENT_LBUTTONDOWN:
        # If a bbox is selected, do not clear on mouse down
        param['click'] = None
        param['start'] = (x, y)
        param['end'] = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        param['end'] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        if x == param['start'][0] and y == param['start'][1]:
            param['click'] = param['start']
            param['start'] = None
            param['end'] = None
        else:
            param['click'] = None
    return 1
