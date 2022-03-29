import DrawHand
import cv2.cv2 as cv2

cap = cv2.VideoCapture(0)
_, frame = cap.read()
board = DrawHand.DrawBoard(frame)
size = (frame.shape[1],frame.shape[0])
writer = cv2.VideoWriter('/home/shanzoom/hand.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),30.0,size)
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    dst = board.runBoard(frame)

    cv2.imshow('dst', dst)
    writer.write(dst)
    if cv2.waitKey(1) == 27:
        break