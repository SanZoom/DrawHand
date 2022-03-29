# DrawHand
#### 1. mediapipe 的使用方法(hands)
````
self.mpHands = mp.solutions.hands
self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                max_num_hands=self.maxHands,
                                min_detection_confidence=self.detectionCon,
                                min_tracking_confidence=self.trackCon)
self.mpDraw = mp.solutions.drawing_utils
````
self.hands.process(img) 即可得到预测结果results，输入图需要是RGB通道