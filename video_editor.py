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
        
        self.SUBMODES = {"TRANSLATION": None, # current mode index, [dx, dy] values
                         "FLIP": None,        # image flip horiz, vert, both
                         "ROTATION": None,    # image rotation angle
                         "EDGE": self._edge_canny,        # change the edge detection mode
                         "SCALE_X": None,     # adjust x scale of the image
                         "SCALE_Y": None      # adjust y scale of the image
                         }
        
        self.MODE_WHEEL = ["FLIP", "EDGE", ""] # list of modes
        self.FLIP_WHEEL = itertools.cycle([None, 0, 1, -1])
        self.EDGE_WHEEL = itertools.cycle([None, self._edge_canny, self._edge_laplacian1, self._edge_laplacian2])
        
        self.MESSAGE_TIME = time.time() + self.MESSAGE_DISPLAY_TIME * self.WELCOME_MESSAGE  # if time < MESSAGE_TIME then blit messages
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
        
    def reset(self):
        self.SUBMODES = {"TRANSLATION": None, # current mode index, [dx, dy] values
                         "FLIP": None,        # image flip horiz, vert, both
                         "ROTATION": None,    # image rotation angle
                         "EDGE": 0,
                         "BLUR": None
                         }
        self.MODE = 0

    def run(self):
            
        capture = cv.VideoCapture(0)
        while True:
            isTrue, frame = capture.read()
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
            
            
            
            # Edit frame
            frame = self.flip(frame)
            
            frame = self.edge_detection(frame)
            
            
            
            # Blit message to the user (like current setting or value)
            self.add_message(frame)
            
            
            
            
            
            
            cv.imshow("Video", frame)
    
    
    
        capture.release()
        cv.destroyAllWindows()

    # =============================================================================
    # MODES
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
        
        # MODE CHANGES
        else:
            self.MESSAGE = self.get_mode()
    
    # =============================================================================
    # FILTERS    
    # =============================================================================
    def flip(self, frame):
        """Mirrors the image."""
        val = self.SUBMODES["FLIP"]
        if val is None:
            return frame
        return cv.flip(frame, val)
    
    def translation(self, frame):
        val = self.SUBMODES["TRANSLATION"]
        if val is None:
            return frame
    
    def rotation(self, frame):
        val = self.SUBMODES["ROTATION"]
        if val is None:
            return frame
    
    # =============================================================================
    # EDGE FILTERS    
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
        return cv.GaussianBlur(frame, (5, 5), cv.BORDER_DEFAULT)
        
        

if __name__ == "__main__":
    settings = {
                'FRAME_SPEED': 20, # one frame per 20 ms
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
    




























