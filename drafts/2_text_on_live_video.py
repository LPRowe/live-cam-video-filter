import cv2 as cv

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



class Video:
    def __init__(self, **kwargs):
        for key in kwargs:
            self.__dict__[key] = kwargs[key]
        self.ARROW_MAP = {2555904: "RIGHT",
                          2621440: "DOWN",
                          2424832: "LEFT",
                          2490368: "UP"}
        self.SUBMODES = {"TRANSLATION": [0, [None, [0, 0]]], # current mode index, [dx, dy] values
                         "FLIP": [0, [None, 0, 1, -1]], # image flip horiz, vert, both
                         "ROTATION": [0, [None, 0]] # image rotation angle
                         }

    def run(self):
            
        capture = cv.VideoCapture(0)
        while True:
            isTrue, frame = capture.read()
            key = cv.waitKeyEx(self.FRAME_SPEED)
            if key == ord('q'):
                break
                    
            # flip frame horizontal
            frame = cv.flip(frame, 1)
            frame = cv.Canny(frame, 125, 125)
            
            cv.imshow("Video", frame)
    
        capture.release()
        cv.destroyAllWindows()


    def orientation(self, frame):
        """
        Applies mirror 
        """
        if mode == 0:
            return 
    
    def translation(self, frame):
        
        
        
        
        
        
        
        
        

if __name__ == "__main__":
    settings = {
                'FRAME_SPEED': 20, # one frame per 20 ms
                'MODE': 0,
                'SUBMODE': 0
                }
    video = Video(**settings)
    video.run()
    




























