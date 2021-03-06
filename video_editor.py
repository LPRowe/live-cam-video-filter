"""
  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.

  By downloading, copying, installing or using the software you agree to this license.
  If you do not agree to this license, do not download, install,
  copy or use the software.
  
                        Intel License Agreement
                For Open Source Computer Vision Library

 Copyright (C) 2000, Intel Corporation, all rights reserved.
 Third party copyrights are property of their respective owners.

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

   * Redistribution's of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.

   * Redistribution's in binary form must reproduce the above copyright notice,
     this list of conditions and the following disclaimer in the documentation
     and/or other materials provided with the distribution.

   * The name of Intel Corporation may not be used to endorse or promote products
     derived from this software without specific prior written permission.

 This software is provided by the copyright holders and contributors "as is" and
 any express or implied warranties, including, but not limited to, the implied
 warranties of merchantability and fitness for a particular purpose are disclaimed.
 In no event shall the Intel Corporation or contributors be liable for any direct,
 indirect, incidental, special, exemplary, or consequential damages
 (including, but not limited to, procurement of substitute goods or services;
 loss of use, data, or profits; or business interruption) however caused
 and on any theory of liability, whether in contract, strict liability,
 or tort (including negligence or otherwise) arising in any way out of
 the use of this software, even if advised of the possibility of such damage.
"""

import time
import itertools

import cv2 as cv
import numpy as np

class Video:
    
    def __init__(self, **kwargs):
        
        for key in kwargs:
            self.__dict__[key] = kwargs[key]
            
        self.FRAME_SHAPE = self.get_frame_shape() # shape of video feed window (height, width)
            
        self.ARROW_MAP = {2555904: "RIGHT",
                          2621440: "DOWN",
                          2424832: "LEFT",
                          2490368: "UP"}
        
        self.SUBMODES = {"TRANSLATE_X": 0,    # pixel offset x direction
                         "TRANSLATE_Y": 0,    # pixel offset y direction
                         "FLIP": 1,           # image flip horiz, vert, both
                         "ROTATE": 0,         # image rotation angle
                         "EDGE": None,        # change the edge detection mode
                         "SCALE_X": None,     # adjust x scale of the image
                         "SCALE_Y": None,     # adjust y scale of the image
                         "BLUR": None,        # control how much the image is blurred
                         "COLOR": None,       # adjust the image colors
                         "SCALE": 1,          # adjust the scale of the image
                         "FACE": 7,           # adjust minimum neighbors for face detection
                         "EYES": 0,           # adjust minimum neighbors for eyes detection
                         "SMILE": -1,         # adjust minimum neighbors for smile detection
                         "FOREGROUND": 0      # add scenery to the foreground
                         }
        
        self.MESSAGE_TIME = time.time() + self.MESSAGE_DISPLAY_TIME * self.WELCOME_MESSAGE  # if time < MESSAGE_TIME then blit messages
        self.MODE_WHEEL = ["FLIP", "EDGE", "BLUR", "COLOR", 
                           "TRANSLATE_X", "TRANSLATE_Y", "ROTATE", "FOREGROUND",
                           "SCALE", "FACE", "SMILE", "EYES", "SECRET_IDENTITY"] # list of modes
        self.FLIP_WHEEL = itertools.cycle([None, 0, 1, -1]) # cycle through 3 image flip options
        self.EDGE_INDEX = 0
        self.EDGE_WHEEL = [None, self._edge_canny, self._edge_laplacian1, self._edge_laplacian2,
                           self._sobel_x, self._sobel_y, self._sobel_xy] # cycle through edge detectors
        
        self.COLOR_INDEX = 0
        self.COLOR_WHEEL = [None, self._hue_saturation_value, self._gray, self._lab, self._red, self._green, self._blue]
        
        self.FOREGROUND_INDEX = 0
        self.FOREGROUND_WHEEL = [None,
                                 cv.resize(cv.imread('images/fruit.png', cv.IMREAD_UNCHANGED), self.FRAME_SHAPE),
                                 cv.resize(cv.imread('images/citrus.png', cv.IMREAD_UNCHANGED), self.FRAME_SHAPE),
                                 cv.resize(cv.imread('images/soccer.png', cv.IMREAD_UNCHANGED), self.FRAME_SHAPE),
                                 cv.resize(cv.imread('images/fire.png', cv.IMREAD_UNCHANGED), self.FRAME_SHAPE),
                                 cv.resize(cv.imread('images/tiger.png', cv.IMREAD_UNCHANGED), self.FRAME_SHAPE)
                                 ]
        self.FOREGROUND_PIXEL_MAP = [None]*len(self.FOREGROUND_WHEEL)
        
        def remove_alpha(img):
            foreground_pixels = []
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if img[i][j][3] < self.ALPHA_THRESHOLD:
                        img[i][j] = [0, 0, 0, 0]
                    else:
                        foreground_pixels.append((i, j))
            return img, foreground_pixels
        
        for i in range(1, len(self.FOREGROUND_WHEEL)):
            self.FOREGROUND_WHEEL[i], self.FOREGROUND_PIXEL_MAP[i] = remove_alpha(self.FOREGROUND_WHEEL[i])
        
        self.MASK_INDEX = 0
        self.MASK_WHEEL = [None, 
                           cv.resize(cv.imread('images/p5mask-gzwop-imgbin.png', cv.IMREAD_UNCHANGED), (0, 0), fx=0.1, fy=0.1),
                           cv.resize(cv.imread('images/p5-canidae-fox-mask.png', cv.IMREAD_UNCHANGED), (0, 0), fx=0.1, fy=0.1),
                           cv.resize(cv.imread('images/female_masquerade_mask.png', cv.IMREAD_UNCHANGED), (0, 0), fx=0.1, fy=0.1),
                           cv.imread('images/gas_mask.png', cv.IMREAD_UNCHANGED),
                           cv.resize(cv.imread('images/spiderman_mask.png', cv.IMREAD_UNCHANGED), (0, 0), fx=0.2, fy=0.2),
                           cv.imread('images/floral_frame.png', cv.IMREAD_UNCHANGED)
                           ]
        
        self.MASK_STYLE = {1: "EYE_MASK",
                           2: "FACE_MASK",
                           3: "EYE_MASK",
                           4: "FACE_MASK",
                           5: "FACE_MASK",
                           6: "FACE_MASK"
                           }
        
        self.MASK_SCALE = {1: 1.25, # magnify mask size by scale
                           2: 1.33,
                           3: 1.25,
                           4: 1.33,
                           5: 2.75,
                           6: 2.05
                           } 
        
        self.MASK_OFFSET_FACTOR = {1: (0, -0.25),   # (x offset (factor of width), y offset (factor of height))
                                   2: (0, -0.33),
                                   3: (0, -0.65),
                                   4: (0.1, -0.1),
                                   5: (0, 0.33),
                                   6: (0, -0.05)
                                   }
        
        self.face_rect = self.eye_rect1 = self.eye_rect2 = self.smile_rectangle = tuple() # initiated as empty before face recognition starts
        
        self.BOX_THICKNESS_WHEEL = itertools.cycle([1, 3, 5, -1, 0])
        
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
                              self._edge_laplacian2: "Laplacian 2",
                              self._sobel_x: "Sobel X Gradient",
                              self._sobel_y: "Sobel Y Gradient",
                              self._sobel_xy: "Sobel"
                              }
        
        self.COLOR_MESSAGES = {None: "No Corrections",
                               self._red: "Red",
                               self._green: "Green",
                               self._blue: "Blue",
                               self._gray: "Gray Scale",
                               self._hue_saturation_value: "Hue Saturated",
                               self._lab: "L*a*b*"
                               }
        
        self.MASK_MESSAGES = {0: "No Mask",
                              1: "JOKER",
                              2: "FOX",
                              3: "MASQUERADE",
                              4: "GAS MASK",
                              5: "SPIDERMAN",
                              6: "FLORAL"
                              }

        self.FOREGROUND_MESSAGES = {0: "No Foreground",
                                    1: "Fruit",
                                    2: "Citrus",
                                    3: "Soccer",
                                    4: "Fire",
                                    5: "Tiger",
                                    }
    
        self.BLUR_MESSAGES = lambda k: f"Kernel {k}" if k else "No Smoothing"
        self.ROTATE_MESSAGES = lambda alpha: f"Rotated {alpha} deg." if alpha else "No Rotation"
        self.TRANSLATE_MESSAGES = lambda dx, dy: f"(x, y): ({dx}, {dy})" if dx | dy else "No Translation"
        self.SCALE_MESSAGES = lambda scale: f"Scale: {scale: .1f}" if scale != 1 else "Scale: Original Scale"
        self.FACE_MESSAGES = lambda neigh: f"Min. Neighbors: {neigh}" if neigh != -1 else "Face Detection: OFF"
        self.EYES_MESSAGES = lambda neigh: f"Eye Detection: ON" if neigh != -1 else "Eye Detection: OFF"
        self.SMILE_MESSAGES = lambda neigh: f"Smile Detection: ON" if neigh != -1 else "Smile Detection: OFF"
    
    def get_frame_shape(self):
        """Returns the shape of the input video feed.
        Used for preprocessing foregrounds."""
        capture = cv.VideoCapture(0)
        is_running, frame = capture.read()
        capture.release()
        return (frame.shape[1], frame.shape[0])
        
    def reset(self):
        """Reset all settings to default. Activated by key press r or R."""
        self.SUBMODES = {"TRANSLATE_X": 0,    # pixel offset x direction
                         "TRANSLATE_Y": 0,    # pixel offset y direction
                         "FLIP": 1,           # image flip horiz, vert, both
                         "ROTATE": 0,         # image rotation angle
                         "EDGE": None,        # change the edge detection mode
                         "SCALE_X": None,     # adjust x scale of the image
                         "SCALE_Y": None,     # adjust y scale of the image
                         "BLUR": None,        # control how much the image is blurred
                         "COLOR": None,       # adjust the image colors
                         "SCALE": 1,          # adjust the scale of the image
                         "FACE": 7,           # adjust minimum neighbors for face detection
                         "EYES": 0,           # adjust minimum neighbors for eyes detection
                         "SMILE": -1,         # adjust minimum neighbors for smile detection
                         "FOREGROUND": 0      # add scenery to the foreground
                         }
        self.COLOR_INDEX = 0
        self.EDGE_INDEX = 0
        self.FOREGROUND_INDEX = 0
        self.face_rect = self.eye_rect1 = self.eye_rect2 = self.smile_rectangle = tuple()

    def run(self):
        """Main video loop. Handles key input, modifies frame, and displays frames."""
        capture = cv.VideoCapture(0)
        while True:
            is_running, frame = capture.read()
            key = cv.waitKeyEx(self.FRAME_SPEED)
            if key != -1:
                print(key)
            if key in [ord('q'), ord('Q')]:
                break
            
            # Reset image adjustments to None
            elif key in [ord('r'), ord('R')]:
                self.reset()
            
            # Change thickness of face detection box / circle
            elif key in [ord('h'), ord('H')]:
                self.BOX_THICKNESS = next(self.BOX_THICKNESS_WHEEL)
            
            # Superimpose foreground pixels with frame if True, else writes over frame pixels
            elif key in [ord('g'), ord('G')]:
                self.FOREGROUND_SUPERPOSITION = not self.FOREGROUND_SUPERPOSITION
                self.MESSAGE_TIME = time.time() + self.MESSAGE_DISPLAY_TIME
                self.MESSAGE = "FOREGROUND: SUPERIMPOSED (light)" if self.FOREGROUND_SUPERPOSITION else "FOREGROUND: SOLID (comp. heavy)"
                
            elif key in [ord('f'), ord('F')]:
                self.SUBMODES['FACE'] = -1 if self.SUBMODES['FACE'] != -1 else 7
                self.MESSAGE_TIME = time.time() + self.MESSAGE_DISPLAY_TIME
                self.MESSAGE = "FACE DETECTION: ON" if self.SUBMODES['FACE'] != -1 else "FACE DETECTION: OFF"
            
            # Change Mode or Submode Values
            elif key in self.ARROW_MAP:
                self.MESSAGE_TIME = time.time() + self.MESSAGE_DISPLAY_TIME
                direction = self.ARROW_MAP[key]
                if direction in ["UP", "DOWN"]:
                    self.change_submode(direction)
                else:
                    self.change_mode(direction)
                self.update_message(direction)
            
            # Modify frame, if a feature is off just passes frame pointer back and forth O(1)
            frame = self.face_detect(frame)
            frame = self.wear_mask(frame)
            frame = self.add_foreground(frame)
            frame = self.scale(frame)
            frame = self.adjust_color(frame)
            frame = self.translate(frame)
            frame = self.flip(frame)
            frame = self.blur(frame)
            frame = self.edge_detection(frame)
            frame = self.rotate(frame)
            
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
            self.EDGE_INDEX += 1 if direction == "UP" else -1
            self.EDGE_INDEX %= len(self.EDGE_WHEEL)
            self.SUBMODES[mode] = self.EDGE_WHEEL[self.EDGE_INDEX]
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
            self.COLOR_INDEX += 1 if direction == 'UP' else -1
            self.COLOR_INDEX %= len(self.COLOR_WHEEL)
            self.SUBMODES[mode] = self.COLOR_WHEEL[self.COLOR_INDEX]
        elif mode == "SCALE":
            self.SUBMODES[mode] += 0.1 if direction == 'UP' else -0.1
            self.SUBMODES[mode] = max(self.SUBMODES[mode], 0.1)
        elif mode == "FACE":
            self.SUBMODES[mode] += 1 if direction == 'UP' else -1
            self.SUBMODES[mode] = max(-1, self.SUBMODES[mode])
        elif mode == "EYES":
            self.SUBMODES[mode] += 1 if direction == 'UP' else -1
            self.SUBMODES[mode] = min(0, max(-1, self.SUBMODES[mode]))
        elif mode == "SMILE":
            self.SUBMODES[mode] += 1 if direction == 'UP' else -1
            self.SUBMODES[mode] = min(0, max(-1, self.SUBMODES[mode]))
        elif mode == "SECRET_IDENTITY":
            self.MASK_INDEX += 1 if direction == "UP" else -1
            self.MASK_INDEX %= len(self.MASK_WHEEL)
            self.SUBMODES[mode] = self.MASK_WHEEL[self.MASK_INDEX]
        elif mode == "FOREGROUND":
            self.FOREGROUND_INDEX += 1 if direction == "UP" else -1
            self.FOREGROUND_INDEX %= len(self.FOREGROUND_WHEEL)
            self.SUBMODES[mode] = self.FOREGROUND_WHEEL[self.FOREGROUND_INDEX]
            
            
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
            elif mode == "SCALE":
                self.MESSAGE = self.SCALE_MESSAGES(self.SUBMODES['SCALE'])
            elif mode == "FACE":
                self.MESSAGE = self.FACE_MESSAGES(self.SUBMODES['FACE'])
            elif mode == "SMILE":
                self.MESSAGE = self.SMILE_MESSAGES(self.SUBMODES['SMILE'])
            elif mode == "EYES":
                self.MESSAGE = self.EYES_MESSAGES(self.SUBMODES['EYES'])
            elif mode == "SECRET_IDENTITY":
                self.MESSAGE = self.MASK_MESSAGES[self.MASK_INDEX]
            elif mode == "FOREGROUND":
                self.MESSAGE = self.FOREGROUND_MESSAGES[self.FOREGROUND_INDEX]
        
        # MODE CHANGES
        else:
            self.MESSAGE = self.get_mode()
    
    # =============================================================================
    # ADD FOREGROUND TO FRAME
    # =============================================================================
    def add_foreground(self, frame):
        if self.FOREGROUND_INDEX == 0:
            return frame
        if self.FOREGROUND_SUPERPOSITION: # superimposes foreground and frame
            frame = frame + self.FOREGROUND_WHEEL[self.FOREGROUND_INDEX][:,:,:3]
        else: # writes foreground over frame (more computation time needed)
            for (i, j) in self.FOREGROUND_PIXEL_MAP[self.FOREGROUND_INDEX]:
                frame[i][j] = self.FOREGROUND_WHEEL[self.FOREGROUND_INDEX][i][j][:3]
        return frame
    
    # =============================================================================
    # FACE DETECTION AND MASK WEARING
    # =============================================================================
    @staticmethod
    def _manhattan(x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)
    
    def face_detect(self, frame):
        min_neighbors = self.SUBMODES['FACE']
        if min_neighbors == -1:
            return frame
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        face_rectangles, reject_levels, confidence = self.CLF_FACE.detectMultiScale3(gray, 
                                                                           scaleFactor = 1.1, 
                                                                           minNeighbors = min_neighbors, 
                                                                           minSize = (150, 150), 
                                                                           outputRejectLevels = True)
        face_rectangles = [face_rectangles[i] for i in range(len(face_rectangles)) if confidence[i] >= self.CONFIDENCE_FACE]
        if face_rectangles == []:
            return frame # no faces were found
        
        # Draw shape around face
        self.face_rect = face_rectangles[0]
        x, y, width, height = self.face_rect # for simplicity, just use the first face rectangle for this app
        center = (x + width // 2, y + height // 2)
        radius = max(width, height) // 2
        if self.BOX_THICKNESS:
            cv.circle(frame, center, radius, self.BOX_COLOR, thickness = self.BOX_THICKNESS)
        
        # Detect eyes and draw shape around eyes
        if self.SUBMODES['EYES'] != -1:
            dx, dy = x, y
            eye_rectangles, reject_levels, confidence = self.CLF_EYES.detectMultiScale3(gray[y:y+height, x:x+width], 
                                                             scaleFactor = 1.1, 
                                                             minNeighbors = min_neighbors,
                                                             minSize = (15, 15),
                                                             outputRejectLevels = True)
            eye_rectangles = [eye_rectangles[i] for i in range(len(eye_rectangles)) if confidence[i] > self.CONFIDENCE_EYES]
            if eye_rectangles != []:
                self.eye_rect1 = eye_rectangles[0]
                x, y, width, height = self.eye_rect1
                x += dx
                y += dy
                center = (x + width // 2, y + height // 2)
                radius = max(width, height) // 2
                if self.BOX_THICKNESS:
                    cv.circle(frame, center, radius, self.BOX_COLOR, thickness = self.BOX_THICKNESS)
                
                # Pick second eye that is furthest from the first eye
                if len(eye_rectangles) > 1:
                    self.eye_rect2 = max(eye_rectangles, key = lambda rect: self._manhattan(rect[0], rect[1], x, y)) 
                    x, y, width, height = self.eye_rect2
                    x += dx
                    y += dy
                    center = (x + width // 2, y + height // 2)
                    radius = max(width, height) // 2
                    if self.BOX_THICKNESS:
                        cv.circle(frame, center, radius, self.BOX_COLOR, thickness = self.BOX_THICKNESS)
        
        # Detect smile and draw shape around smile    
        if self.SUBMODES['SMILE'] != -1:
            smile_rectangles, reject_levels, confidence = self.CLF_SMILE.detectMultiScale3(gray, 
                                                                      scaleFactor = 1.1, 
                                                                      minNeighbors = min_neighbors,
                                                                      minSize = (15, 15),
                                                                      outputRejectLevels = True)
            smile_rectangles = [smile_rectangles[i] for i in range(len(smile_rectangles)) if confidence[i] > self.CONFIDENCE_SMILE]
            if self.smile_rectangle != []:
                x, y, width, height = smile_rectangles[0]
                center = (x + width // 2, y + height // 2)
                radius = max(width, height) // 2
                if self.BOX_THICKNESS:
                   cv.circle(frame, center, radius, self.BOX_COLOR, thickness = self.BOX_THICKNESS)
                
        return frame
    
    def _alpha_overlay(self, frame, mask, x, y):
        """Overlays only the mask on top of the frame where the mask is not transparent.
        x and y are the horizontal and vertical offset between the top left corner fo the frame, 
        and the top left corner of the mask"""
        thresh = self.ALPHA_THRESHOLD
        for i in range(max(0, -y), min(frame.shape[0] - y, mask.shape[0])):
            for j in range(max(0, -x), min(frame.shape[1] - x, mask.shape[1])):
                if mask[i][j][3] > thresh:
                    frame[y+i][x+j] = mask[i][j][:3]
        return frame
        
    
    def wear_mask(self, frame):
        """Places a mask on the user's face.
        Face masks adjust shape with the face rectangle.
        Eye masks adjust height according to the first eye rectangle."""
        if any([self.MASK_INDEX == 0, self.SUBMODES['FACE'] == -1, self.SUBMODES['EYES'] == -1,
                self.face_rect == (), self.eye_rect1 == ()]):
            return frame
        style = self.MASK_STYLE[self.MASK_INDEX] # is it a face mask or eye mask?
        scale = self.MASK_SCALE[self.MASK_INDEX] # magnify mask size by scale
        mask = self.MASK_WHEEL[self.MASK_INDEX]
        ratio = self.face_rect[2] / mask.shape[1]
        h0, w0 = mask.shape[0]*ratio, mask.shape[1]*ratio
        h1, w1 = scale * h0, scale * w0
        dh, dw = h1 - h0, w1 - w0
        fx = fy = scale * ratio # ratio if mask width to face width
        mask = cv.resize(mask, (0, 0), fx = fx, fy = fy)
        offset_x, offset_y = self.MASK_OFFSET_FACTOR[self.MASK_INDEX]
        if style == "EYE_MASK":
            x = self.face_rect[0] + w0 * offset_x
            y = self.face_rect[1] + self.eye_rect1[1] + h0 * offset_y # top of eye height
        else:
            x, y = self.face_rect[0] + w0 * offset_x, self.face_rect[1] + h0 * offset_y
        y = int(y - dh // 2)
        x = int(x - dw // 2)
        frame = self._alpha_overlay(frame, mask, x, y)
        return frame
            
        
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
        """returns L*a*b* image"""
        return cv.cvtColor(frame, cv.COLOR_BGR2LAB)
    
    def adjust_color(self, frame):
        """Applies one of the above color adjustments to the image."""
        func = self.SUBMODES["COLOR"]
        if func is None:
            return frame
        return func(frame)

    # =============================================================================
    # GEOMETRIC FILTERS    
    # =============================================================================
    def flip(self, frame):
        """Mirrors the image about x-axis, about y-axis, or both."""
        val = self.SUBMODES["FLIP"]
        if val is None:
            return frame
        return cv.flip(frame, val)
    
    def translate(self, frame):
        """Shifts the image left, right, up, or down"""
        dx, dy = self.SUBMODES["TRANSLATE_X"], self.SUBMODES["TRANSLATE_Y"]
        if (dx | dy) == 0:
            return frame
        translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv.warpAffine(frame, translation_matrix, (frame.shape[1], frame.shape[0]))
    
    def rotate(self, frame):
        """Rotates the image about it's center point."""
        angle = self.SUBMODES["ROTATE"]
        if angle == 0:
            return frame
        width, height = frame.shape[:2]
        center = height // 2, width // 2
        rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0) # point_of_rotation, angle, scale
        return cv.warpAffine(frame, rotation_matrix, (height, width))
    
    def scale(self, frame):
        """Resizes the frame by scale s."""
        s = self.SUBMODES["SCALE"]
        if s == 1:
            return frame
        return cv.resize(frame, (int(frame.shape[1] * s), int(frame.shape[0] * s)), interpolation=cv.INTER_AREA)        
        
    # =============================================================================
    # EDGE AND BLUR FILTERS 
    # =============================================================================
    def _sobel_x(self, frame, gray = None):
        """Applies Sobel edge detection filter using x-gradients (highlights vertical lines) to frame."""
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) if gray is None else gray
        return cv.Sobel(gray, cv.CV_64F, 1, 0) # 1 x dir and 0 y dir
        
    def _sobel_y(self, frame, gray = None):
        """Applies Sobel edge detection filter using y-gradients (highlights horizontal lines) to frame."""
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) if gray is None else gray
        return cv.Sobel(gray, cv.CV_64F, 0, 1) # 0 x dir and 1 y dir
        
    def _sobel_xy(self, frame):
        """Applies Sobel edge detection filter to frame."""
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        return cv.bitwise_or(self._sobel_x(frame, gray = gray), self._sobel_y(frame, gray = gray))
    
    def _edge_canny(self, frame):
        """Applies Canny edge detection filter to frame."""
        return cv.Canny(frame, self.CANNY_THRESHOLD1, self.CANNY_THRESHOLD2)
    
    def _edge_laplacian1(self, frame):
        """Applies Laplacian edge detection filter to the frame."""
        return cv.Laplacian(frame, cv.CV_64F)

    def _edge_laplacian2(self, frame):
        """Applies Laplacian edge detection filter to the frame."""
        frame = cv.Laplacian(frame, cv.CV_64F)
        return np.uint8(np.absolute(frame))
    
    def edge_detection(self, frame):
        """Applies one of the above edge detection methods to frame."""
        val = self.SUBMODES["EDGE"]
        if val is None:
            return frame
        return val(frame)
        
    def blur(self, frame):
        """Uses pixel averaging to blur the frame, larger kernel size results in greater blurring.
        Kernel size (val, val) determines the size of a region over which to average for each pixel."""
        val = self.SUBMODES["BLUR"]
        if val is None:
            return frame
        return cv.blur(frame, (val, val))

if __name__ == "__main__":
    video_settings = {
                'FRAME_SPEED': 50, # one frame per __ ms
                'MODE': 0,         # Starting edit mode
                'ALPHA_THRESHOLD': 10,
                'FOREGROUND_SUPERPOSITION': True # False: foreground writes over image, True: superimposes pixels
                }
    
    text_settings = {
                'MESSAGE_DISPLAY_TIME': 2, # messages will display for 2 seconds
                'MESSAGE': "hello world",  # message displayed on start
                'WELCOME_MESSAGE': True,
                'FONT_COLOR': (255, 255, 255),
                'FONT_SCALE': 1.0,
                'FONT_POSITION': (10, 30)    
                }
    
    classifier_settings = {
                'CLF_FACE': cv.CascadeClassifier('classifiers/haar_face.xml'),
                'CLF_EYES': cv.CascadeClassifier('classifiers/haar_eye.xml'),
                'CLF_SMILE': cv.CascadeClassifier('classifiers/haar_smile.xml'),
                'BOX_COLOR': (0, 200, 0), # (B, G, R)
                'BOX_THICKNESS': 0,       # can be cycled with "h" key
                'CONFIDENCE_FACE': 3,     # 0 is minimum, 9 is very high filters out faces below this confidence level
                'CONFIDENCE_EYES': 1.5,
                'CONFIDENCE_SMILE': 1.5
                }
    
    edge_settings = {
                'CANNY_THRESHOLD1': 125,
                'CANNY_THRESHOLD2': 175
                }
    
    video = Video(**video_settings, **text_settings, **classifier_settings, **edge_settings)
    video.run()