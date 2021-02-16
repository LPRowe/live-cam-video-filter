import cv2 as cv


capture = cv.VideoCapture(0)
while True:
    isTrue, frame = capture.read()
    
    # flip frame horizontal
    frame = cv.flip(frame, 1)
    frame = cv.Canny(frame, 125, 125)
    
    cv.imshow("Video", frame)
    if cv.waitKey(20) & 255 == ord('d'):
        break
capture.release()
cv.destroyAllWindows()