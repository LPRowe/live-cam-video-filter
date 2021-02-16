# matrix filter: sliding repeat image of text that whose intensity is modified by a canny filter
# AI line filter: ai lines for 5 by 5 squares or 7 by 7 squares
#                 25 inputs one for each pixel 
#                 2 outputs: length [0 .. 1] * chord length of square, angle [0 .. 1] * 180 output
#         training: 2**25 inputs approx 33 million
#                   generate all inputs and calculate the optimal line length and orientation

# have keyboard input that goes left to right for mode type
#    modes = [orientation: flip horizontal, vertical, both,
#             color scheme: red, blue, green only,
#             blur: adjust level of blurriness,
#             contour: differeent contour methods, sobel, canny, adaptive threshold, laplacian,
#             mask: draw a shape that will be auto filled and used as a mask (up / down rectangle, square, star, etc),
#            ]


# Once face tracking is added, ahve a draw or stamp feature that moves the drawing (mustache or glasses) with the face and
# resizes it based on the height and width of the face detection box and recenters it accordingly



# TODO:
# Add a mask that can go over the video
# Add a skew option
# Add a drawing option
# add a reset button
# update readme (remember ctrl c and ctrl s functions work)



import time
import itertools

import cv2 as cv
import numpy as np

class Video:
    
    def __init__(self, **kwargs):
        
        for key in kwargs:
            self.__dict__[key] = kwargs[key]
            
        self.ARROW_MAP = {2555904: "RIGHT",
                          2621440: "DOWN",
                          2424832: "LEFT",
                          2490368: "UP"}
        
        self.SUBMODES = {"TRANSLATE_X": 0,    # pixel offset x direction
                         "TRANSLATE_Y": 0,    # pixel offset y direction
                         "FLIP": None,        # image flip horiz, vert, both
                         "ROTATE": 0,         # image rotation angle
                         "EDGE": None,        # change the edge detection mode
                         "SCALE_X": None,     # adjust x scale of the image
                         "SCALE_Y": None,     # adjust y scale of the image
                         "BLUR": None,        # control how much the image is blurred
                         "COLOR": None        # adjust the image colors
                         }
        
        self.MESSAGE_TIME = time.time() + self.MESSAGE_DISPLAY_TIME * self.WELCOME_MESSAGE  # if time < MESSAGE_TIME then blit messages
        
        self.MODE_WHEEL = ["FLIP", "EDGE", "BLUR", "COLOR", 
                           "TRANSLATE_X", "TRANSLATE_Y", "ROTATE"] # list of modes
        self.FLIP_WHEEL = itertools.cycle([None, 0, 1, -1]) # cycle through 3 image flip options
        self.EDGE_WHEEL = itertools.cycle([None, self._edge_canny, self._edge_laplacian1, self._edge_laplacian2]) # cycle through 3 edge detectors
        self.COLOR_WHEEL = itertools.cycle([None, self._hue_saturation_value, self._gray, self._lab, self._red, self._green, self._blue])
        
        # =============================================================================
        # USER MESSAGES BASED ON CURRENT ACTIVITY
        # =============================================================================
        self.FLIP_MESSAGES = {None: 'No Corrections',
                              0: 'Vertical Mirror',
                              1: 'Horizontal Mirror',
                              -1: 'Horizontal and Vertical Mirror'
                              }
        
        self.EDGE_MESSAGES = {None: "No Corrections",
                              self._edge_canny: "Canny",
                              self._edge_laplacian1: "Laplacian 1",
                              self._edge_laplacian2: "Laplacian 2"
                              }
        
        self.COLOR_MESSAGES = {None: "No Corrections",
                               self._red: "Red",
                               self._green: "Green",
                               self._blue: "Blue",
                               self._gray: "Gray Scale",
                               self._hue_saturation_value: "Hue Saturated",
                               self._lab: "L*a*b"
                               }
    
        self.BLUR_MESSAGES = lambda k: f"Kernel {k}" if k else "No Smoothing"
        self.ROTATE_MESSAGES = lambda alpha: f"Rotated {alpha} deg." if alpha else "No Rotation"
        self.TRANSLATE_MESSAGES = lambda dx, dy: f"(x, y): ({dx}, {dy})" if dx | dy else "No Translation"
        
    def reset(self):
        self.SUBMODES = {"TRANSLATE_X": None, # pixel offset x direction
                         "TRANSLATE_Y": None, # pixel offset y direction
                         "FLIP": None,        # image flip horiz, vert, both
                         "ROTATE": None,      # image rotation angle
                         "EDGE": 0,
                         "BLUR": None
                         }
        self.MODE = 0

    def run(self):
            
        capture = cv.VideoCapture(0)
        while True:
            is_running, frame = capture.read()
            key = cv.waitKeyEx(self.FRAME_SPEED)
            if key != -1:
                print(key)
            if key == ord('q') or key == ord('Q'):
                break
            
            # Change Mode or Submode Values
            if key in self.ARROW_MAP:
                self.MESSAGE_TIME = time.time() + self.MESSAGE_DISPLAY_TIME
                direction = self.ARROW_MAP[key]
                if direction in ["UP", "DOWN"]:
                    self.change_submode(direction)
                else:
                    self.change_mode(direction)
                self.update_message(direction)
            
            # Modify frame
            frame = self.adjust_color(frame)
            frame = self.translate(frame)
            frame = self.rotate(frame)
            frame = self.flip(frame)
            frame = self.blur(frame)
            frame = self.edge_detection(frame)
            
            # Blit message to the user (like current setting or value)
            self.add_message(frame)
            
            cv.imshow("Video", frame)
    
        capture.release()
        cv.destroyAllWindows()

    # =============================================================================
    # ADJUST MODE / SETTINGS
    # =============================================================================
    def change_mode(self, direction):
        self.MODE = (self.MODE + 1) if direction == "RIGHT" else (self.MODE - 1)
        self.MODE %= len(self.MODE_WHEEL)
        self.MESSAGE = self.get_mode()
        
    def change_submode(self, direction):
        mode = self.get_mode()
        if mode == "FLIP":
            self.SUBMODES[mode] = next(self.FLIP_WHEEL)
        elif mode == "EDGE":
            self.SUBMODES[mode] = next(self.EDGE_WHEEL)
        elif mode == "BLUR":
            if self.SUBMODES[mode] is None:
                self.SUBMODES[mode] = 1
            if direction == 'UP':
                self.SUBMODES[mode] = min(15, self.SUBMODES[mode] + 1)
            else:
                self.SUBMODES[mode] -= 1
                if self.SUBMODES[mode] <= 1:
                    self.SUBMODES[mode] = None
        elif mode == "TRANSLATE_X":
            self.SUBMODES[mode] += 1 if direction == 'UP' else -1
        elif mode == "TRANSLATE_Y":
            self.SUBMODES[mode] += -1 if direction == 'UP' else 1
        elif mode == "ROTATE":
            self.SUBMODES[mode] += 1 if direction == 'UP' else -1
            self.SUBMODES[mode] %= 360
        elif mode == "COLOR":
            self.SUBMODES[mode] = next(self.COLOR_WHEEL)
            
    def get_mode(self):
        return self.MODE_WHEEL[self.MODE]
    
    # =============================================================================
    # MESSAGES
    # =============================================================================
    def add_message(self, frame):
        if time.time() < self.MESSAGE_TIME:
            cv.putText(frame, self.MESSAGE, self.FONT_POSITION, cv.FONT_HERSHEY_COMPLEX, 
                       self.FONT_SCALE, self.FONT_COLOR, 2)
            
    def update_message(self, direction):
        
        # SUBMODE CHANGES
        if direction in ['UP', 'DOWN']:
            mode = self.get_mode()
            if mode == 'FLIP':
                self.MESSAGE = self.FLIP_MESSAGES[self.SUBMODES['FLIP']]
            elif mode == 'EDGE':
                self.MESSAGE = self.EDGE_MESSAGES[self.SUBMODES['EDGE']]
            elif mode == 'BLUR':
                self.MESSAGE = self.BLUR_MESSAGES(self.SUBMODES['BLUR'])
            elif 'TRANSLATE' in mode:
                self.MESSAGE = self.TRANSLATE_MESSAGES(self.SUBMODES['TRANSLATE_X'], self.SUBMODES['TRANSLATE_Y'])
            elif mode == 'ROTATE':
                self.MESSAGE = self.ROTATE_MESSAGES(self.SUBMODES['ROTATE'])
            elif mode == "COLOR":
                self.MESSAGE = self.COLOR_MESSAGES[self.SUBMODES['COLOR']]
        
        # MODE CHANGES
        else:
            self.MESSAGE = self.get_mode()
            
    # =============================================================================
    # COLOR OPTIONS
    # =============================================================================
    def _dummy_layer(self, frame):
        """Returns frame of all zero valued pixels"""
        return np.full(frame.shape[:2], 0, dtype='uint8')
    
    def _red(self, frame):
        """returns only the red components of the image"""
        dummy = self._dummy_layer(frame)
        return cv.merge([dummy, dummy, cv.split(frame)[2]])
    
    def _green(self, frame):
        """returns only the green components of the image"""
        dummy = self._dummy_layer(frame)
        return cv.merge([dummy, cv.split(frame)[1], dummy])
    
    def _blue(self, frame):
        """returns only the blue components of the image"""
        dummy = self._dummy_layer(frame)
        return cv.merge([cv.split(frame)[0], dummy, dummy])
    
    def _gray(self, frame):
        """returns a gray scale image"""
        return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    def _hue_saturation_value(self, frame):
        """returns a hue saturated image"""
        return cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    def _lab(self, frame):
        """returns L*a*b image"""
        return cv.cvtColor(frame, cv.COLOR_BGR2LAB)
    
    def adjust_color(self, frame):
        func = self.SUBMODES["COLOR"]
        if func is None:
            return frame
        return func(frame)

    # =============================================================================
    # GEOMETRIC FILTERS    
    # =============================================================================
    def flip(self, frame):
        """Mirrors the image."""
        val = self.SUBMODES["FLIP"]
        if val is None:
            return frame
        return cv.flip(frame, val)
    
    def translate(self, frame):
        dx, dy = self.SUBMODES["TRANSLATE_X"], self.SUBMODES["TRANSLATE_Y"]
        if (dx | dy) == 0:
            return frame
        translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv.warpAffine(frame, translation_matrix, (frame.shape[1], frame.shape[0]))
    
    def rotate(self, frame):
        angle = self.SUBMODES["ROTATE"]
        if angle == 0:
            return frame
        width, height = frame.shape[:2]
        center = height // 2, width // 2
        rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0) # point_of_rotation, angle, scale
        return cv.warpAffine(frame, rotation_matrix, (height, width))
        
    # =============================================================================
    # EDGE AND BLUR FILTERS 
    # =============================================================================
    def _edge_canny(self, frame):
        return cv.Canny(frame, 125, 175)
    
    def _edge_laplacian1(self, frame):
        return cv.Laplacian(frame, cv.CV_64F)

    def _edge_laplacian2(self, frame):
        frame = cv.Laplacian(frame, cv.CV_64F)
        return np.uint8(np.absolute(frame))
    
    def edge_detection(self, frame):
        val = self.SUBMODES["EDGE"]
        if val is None:
            return frame
        return val(frame)
        
    def blur(self, frame):
        val = self.SUBMODES["BLUR"]
        if val is None:
            return frame
        return cv.blur(frame, (val, val))

if __name__ == "__main__":
    settings = {
                'FRAME_SPEED': 50, # one frame per 20 ms
                'MODE': 0,
                'SUBMODE': 0
                }
    
    text_settings = {
                'MESSAGE_DISPLAY_TIME': 2, # messages will display for 2 seconds
                'MESSAGE': "hello world",
                'WELCOME_MESSAGE': True,
                'FONT_COLOR': (255, 255, 255),
                'FONT_SCALE': 1.0,
                'FONT_POSITION': (10, 30)    
                }
    
    video = Video(**settings, **text_settings)
    video.run()