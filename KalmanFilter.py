import numpy as np


class KF:
    def __init__(self, point, state_num=4, measure_num=2, control_num=0):
        x, y = point
        self.x = np.matrix([[np.float32(x)], [np.float32(y)], [0.], [0.]])
        self.x_ = self.x
        self.z = np.matrix([[0.], [0.]])
        self.F = np.matrix([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.matrix([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.Q = np.matrix([[0.5,0.,0.,0.],
                            [0.,0.5,0.,0.],
                            [0.,0.,0.5,0.],
                            [0.,0.,0.,0.5]])
        self.R = np.matrix([[0.2, 0.],
                           [0., 0.2]])
        self.P = np.matrix([[1.,0.,0.,0.],
                            [0.,1.,0.,0.],
                            [0.,0.,1.,0.],
                            [0.,0.,0.,1.]])
        self.P_ = self.P
        self.I = np.matrix(np.eye(4))
        self.K = None

    def predict(self):
        self.x_ = self.F * self.x
        self.P_ = self.F * self.P * self.F.T + self.Q

    def correct(self, point):
        x, y = point
        self.z = np.matrix([[np.float(x)], [np.float(y)]])
        temp = self.H * self.P_ * self.H.T + self.R
        self.K = self.P_ * self.H.T * temp.I
        self.x = self.x_ + self.K * (self.z - self.H * self.x_)
        self.P = (self.I - self.K * self.H) * self.P_

        x_c = int(self.x[0][0])
        y_c = int(self.x[1][0])
        point_correct = (x_c, y_c)

        return point_correct

    def run_KF(self, point):
        self.predict()
        return self.correct(point)