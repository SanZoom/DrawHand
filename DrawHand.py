# 虚拟实时视频黑板 —— 基于 mediapipe 的手势识别
# 作者：杨绪康
# 日期：2021/10/25

import cv2.cv2 as cv2
import numpy as np
import mediapipe as mp
import math
import KalmanFilter

DEBUG_POSE = True


class HandDetector:
    def __init__(self, mode=False, maxHands=1, detectionCon=0.7, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.img = None
        self.results = None  # process 的结果

    def findHands(self, img):
        self.img = img
        imgRGB = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if not self.results.multi_hand_landmarks:
            return False
        return True

    '''
       @return  返回关节点的字典，序号为key，坐标为value，未识别到返回None
    '''

    def findPosition(self):
        pointlist = []
        IDlist = []
        ID_point_dir = {}
        if self.results.multi_hand_landmarks:
            for hand_landmark in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(self.img, hand_landmark, self.mpHands.HAND_CONNECTIONS)
                for pointID, lm in enumerate(hand_landmark.landmark):
                    h, w, c = self.img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    pointlist.append((cx, cy))  # 将各个手指点存入列表
                    IDlist.append(pointID)

                    # if pointID == 0:
                    #     cv2.circle(self.img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                    # else:
                    #     cv2.putText(self.img, str(pointID), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 0))
                    #     cv2.circle(self.img, (cx, cy), 5, (0, 255, 5), cv2.FILLED)
            ID_point_dir = dict(zip(IDlist, pointlist))
        return ID_point_dir

    '''
        @brief  对外API，获得图片中的手指关节点
        @param  img 输入的图片
        @return  返回关节点的字典，序号为key，坐标为value，未识别到返回None
    '''

    def getPointDict(self, img):
        self.findHands(img)
        return self.findPosition()


class DrawBoard:
    def __init__(self, src, detector=HandDetector(), color=(0, 0, 255), thickness=2):
        self.handDetector = detector
        self.shape = src.shape
        self.board = np.zeros(self.shape, dtype=np.uint8)  # 画板
        self.boards = {}
        self.current_board_index = 1
        self.boards[self.current_board_index] = self.board
        self.color = color
        self.thick = int(thickness)
        self.fingerPoints = {}
        self.pointformer = None
        self.pose = None
        self.drawPoint = None
        self.KF = None

    def draw(self, point):
        if self.pointformer is None:
            self.pointformer = point
            self.KF = KalmanFilter.KF(point)
        if point is not None:
            x,y = self.KF.run_KF(point)
            point = (int(x),int(y))
            cv2.line(self.board, self.pointformer, point, self.color, self.thick)
        self.pointformer = point

    def erase(self):
        if (7 and 17) in self.fingerPoints.keys():
            cv2.rectangle(self.board, self.fingerPoints[7], self.fingerPoints[17], (0, 0, 0), -1)

    # 新建画板
    def newBoard(self):
        new_board = np.zeros(self.shape, dtype=np.uint8)
        self.current_board_index += 1
        self.boards[self.current_board_index] = new_board
        self.board = self.boards[self.current_board_index]

    def changeBoard(self, index):
        return

    def getboardOnframe(self, frame):
        gray = cv2.cvtColor(self.board, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)
        mask_inv = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask)
        board_fg = cv2.bitwise_and(self.board, self.board, mask=mask_inv)
        dst = cv2.add(frame_bg, board_fg)

        return dst

    def getDistance(self, index1, index2):
        x1, y1 = self.fingerPoints[index1]
        x2, y2 = self.fingerPoints[index2]
        length = math.hypot(x2 - x1, y2 - y1)
        return length

    # 返回余弦值
    def getAngel(self, index1, index_center, index2):
        a = self.getDistance(index1, index_center) + 0.001
        b = self.getDistance(index2, index_center) + 0.001
        c = self.getDistance(index1, index2) + 0.001
        angle = (a * a + b * b - c * c) / 2 / b / a  # 余弦公式
        return angle

    # 1: draw, 0: erase, -1: None
    def judgePose(self, img):  # 7-8 11-12 15-16 19-20
        self.fingerPoints = self.handDetector.getPointDict(img)

        # 食指绘画, 8：指尖
        # curve_num = 0
        straight_thresh = math.cos(145.0/180.0*math.pi)
        # curve_thresh = -0.5
        # if (6 and 7 and 5 and 8) in self.fingerPoints.keys():
        #     if self.getAngel(7, 6, 5) < straight_thresh:
        #         if (9 and 10 and 11) in self.fingerPoints.keys():
        #             if self.getAngel(9, 10, 11) > curve_thresh:
        #                 curve_num += 1
        #         if (13 and 14 and 15) in self.fingerPoints.keys():
        #             if self.getAngel(15, 14, 13) > curve_thresh:
        #                 curve_num += 1
        #         if (17 and 18 and 19) in self.fingerPoints.keys():
        #             if self.getAngel(17, 18, 19) > curve_thresh:
        #                 curve_num += 1
        #         if curve_num >= 2:
        #             self.drawPoint = self.fingerPoints[8]
        #             return 1


        # "乳韩手势"以关节点距离作为基准距离
        baseDistance = 0
        if 4 and 3 in self.fingerPoints.keys():
            baseDistance = self.getDistance(4, 3) / 1.2
        # elif 7 and 8 in self.fingerPoints.keys():         # 食指
        #     baseDistance = self.getDistance(7, 8)
        elif 11 and 12 in self.fingerPoints.keys():  # 中指
            baseDistance = self.getDistance(11, 12)
        elif 15 and 16 in self.fingerPoints.keys():  # 无名指
            baseDistance = self.getDistance(15, 16)
        elif 19 and 20 in self.fingerPoints.keys():  # 小拇指
            baseDistance = self.getDistance(19, 20)
        if 4 and 8 in self.fingerPoints.keys():
            ifDraw = self.getDistance(4, 8) < baseDistance
            if ifDraw:
                x1, y1 = self.fingerPoints[4]
                x2, y2 = self.fingerPoints[8]
                x = int((x1 + x2) / 2)
                y = int((y1 + y2) / 2)
                self.drawPoint = (x, y)
                return 1

        # 弯曲判断
        # 5-6-7-8 9-10-11-12 13-14-15-16 17-18-19-20
        angle_1 = 1
        angle_2 = 1
        angle_3 = 1
        angle_4 = 1
        if (6 and 7 and 8) in self.fingerPoints.keys():
            angle_1 = self.getAngel(6, 7, 8)
        if (9 and 10 and 11) in self.fingerPoints.keys():
            angle_2 = self.getAngel(9, 10, 11)
        if (13 and 14 and 15) in self.fingerPoints.keys():
            angle_3 = self.getAngel(15, 14, 13)
        if (17 and 18 and 19) in self.fingerPoints.keys():
            angle_4 = self.getAngel(17, 18, 19)

        straight_num = 0
        if angle_1 < straight_thresh:
            straight_num += 1
        if angle_2 < straight_thresh:
            straight_num += 1
        if angle_3 < straight_thresh:
            straight_num += 1
        if angle_4 < straight_thresh:
            straight_num += 1
        ifErase = straight_num >= 3
        if ifErase:
            return 0
        return -1

    def runBoard(self, img):
        # dst = img
        pose = self.judgePose(img)
        if DEBUG_POSE:
            cv2.putText(img, str(pose), (50, 50), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255))
        if pose == 1:
            self.draw(self.drawPoint)
        else:
            self.pointformer = None

        if pose == 0:
            self.erase()
        dst = self.getboardOnframe(img)
        return dst
