from preprocess import *

class PID():
    def __init__(self, target, I=0.2):
        self.Ki = I
        self.error = 0
        self.target = target
    def update(self, value):
        self.error = self.target - value
        I_val = self.Ki * self.error
        return value + I_val

    def set_target(self, target):
        self.target = target



# Define a class to receive the characteristics of each line detection
class Line():
    queue_len = 10
    def __init__(self, name):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = deque(maxlen=self.queue_len)
        self.recent_yfitted = deque(maxlen=self.queue_len)
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = None
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        self.last_diffs = self.diffs
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        self.th_scale = 1
        self.th_pid = PID(self.th_scale)
        self.unvalid_cnt = 0
        self.name = name

        self.curvature_deque = deque(maxlen=self.queue_len)
        self.fit_deque = deque(maxlen=self.queue_len)
        self.n_x_deque = deque(maxlen=self.queue_len)       # weight
        self.last_fit = None
        self.fit_cnt = 10

    def is_detected(self):
        return self.detected

    def cal_bestx(self):
        try:
            x_values = np.hstack(self.recent_xfitted)
            self.bestx = np.mean(x_values, axis=0, dtype=np.float32)
            print("bestx: {:>.2f}".format(self.bestx))
            return self.bestx
        except ValueError:
            return None

    def cal_best_fit(self):
        if len(self.fit_deque) == self.queue_len:
            fits = np.array(self.fit_deque)
            sum_weight = 0
            sum_fit = [0, 0, 0]
            for i, fit in enumerate(fits):
                sum_weight += i+1
                sum_fit += fit * (i+1)
            best_fit = sum_fit / sum_weight
            self.best_fit = best_fit
        elif len(self.fit_deque) > 0:
            mean_fit = np.mean(self.fit_deque, axis=0)
            self.best_fit = self.current_fit * 0.5 + mean_fit * 0.5
        else:
            self.best_fit = self.current_fit
        return self.best_fit

    def cal_current_fit(self, x=None, y=None):
        try:
            if x is None or y is None:
                self.current_fit = np.polyfit(self.ally, self.allx, 2)
            else:
                self.current_fit = np.polyfit(y, x, 2)
        except TypeError or ValueError as e:
            print(self.name, self.cal_current_fit.__name__, e)
            return None
        return self.current_fit

    def store_current_fit(self):
        if self.current_fit is not None:
            # store fitted x value
            self.recent_xfitted.append(self.allx)
            self.recent_yfitted.append(self.ally)
            # x = np.concatenate(self.recent_xfitted, axis=0)
            # y = np.concatenate(self.recent_yfitted, axis=0)
            # self.best_fit = np.polyfit(y, x, 2)
            # if self.best_fit is not None:
            #     self.last_fit = self.best_fit * 0.5 + self.current_fit * 0.5
            # else:
            #     self.last_fit = self.current_fit
            self.fit_deque.append(self.current_fit)

    def cal_radius_of_curvature(self):
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        y_eval = np.max(self.ally)
        try:
            # x = np.concatenate(self.recent_xfitted, axis=0)
            # y = np.concatenate(self.recent_yfitted, axis=0)
            x = self.allx
            y = self.ally
            # Fit new polynomials to x,y in world space
            fit_cr = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
            # Calculate the new radii of curvature
            curve = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) \
                                       / abs(2 * fit_cr[0])
        except:
            print(self.name, self.cal_radius_of_curvature.__name__)
            return self.radius_of_curvature

        self.curvature_deque.append(curve)
        self.radius_of_curvature = np.mean(self.curvature_deque, dtype=np.float32)
        return self.radius_of_curvature

    def cal_diff(self):
        try:
            print(self.name, "last fit", self.last_fit)
            print(self.name, 'curr fit', self.current_fit)
            self.last_diffs = self.diffs
            self.diffs = np.array(self.current_fit) - self.last_fit
            return self.diffs
        except:
            print(self.name, self.cal_diff.__name__, "diff failed")
            self.last_fit = self.current_fit
            return None

    def set_allx(self, x):
        self.allx = x

    def set_ally(self, y):
        self.ally = y

    def cal_line_base_pos(self):
        try:
            xm_per_pix = 3.7 / 700
            self.line_base_pos = (self.bestx - 640)  * xm_per_pix
            return self.line_base_pos
        except TypeError:
            return None

    def valid_xy(self, x, y, fit_thresh=(1e-2, 3e-1, 1e2)):
        self.th_scale = self.th_pid.update(self.th_scale)
        # print(self.name, "Th Scale: ", self.th_scale)
        try:
            if len(self.curvature_deque) > 0:
                curve = abs(np.mean(self.curvature_deque, dtype=np.float32))
            else:
                curve = 10000

            fit_thresh = np.array(fit_thresh)
            if curve < 50: factor = 5
            elif curve < 100: factor = 4
            elif curve < 150: factor = 3
            elif curve < 200: factor = 2
            elif curve < 400: factor = 2.5
            elif curve < 500: factor = 2.4
            elif curve < 600: factor = 2.3
            elif curve < 700: factor = 2.2
            elif curve < 800: factor = 2.0
            elif curve < 1000: factor = 1.5
            elif curve < 2000: factor = 1.3
            elif curve < 3000: factor = 1.2
            elif curve < 5000: factor = 1.1
            elif curve < 10000: factor = 1.0
            else: factor = 0.9
            fit_thresh = fit_thresh * factor * self.th_scale
            print("Fit_Thresh: ", fit_thresh, curve, factor)
        except:
            print('curve miss')

        self.set_allx(x)
        self.set_ally(y)
        if self.cal_current_fit() is None:
            self.unvalid_cnt+= 1
            return False
        fit_diff = self.cal_diff()
        if fit_diff is None:
            self.detected = True
            self.store_current_fit()
            return True

        thresh_logic = abs(fit_diff[0]) > fit_thresh[0] or abs(fit_diff[1]) > fit_thresh[1] or abs(fit_diff[2]) > fit_thresh[2]
        # diffs_logic = (abs(fit_diff[0] - self.last_diffs[0]) > 0.0005 or \
        #               abs(fit_diff[1] - self.last_diffs[1]) > 0.25) or \
        #               abs(fit_diff[2] - self.last_diffs[2]) > 2 * abs(self.last_diffs[2])

        diffs_logic = (abs(fit_diff[0] - self.last_diffs[0]) > self.th_scale * max((min((abs(fit_diff[0]), abs(self.last_diffs[0]))), 0.00015)) or \
                      abs(fit_diff[1] - self.last_diffs[1]) > self.th_scale * max((min((abs(fit_diff[1]), abs(self.last_diffs[1]))), 0.15)) ) or \
                     abs(fit_diff[2] - self.last_diffs[2]) > self.th_scale * max((min((abs(fit_diff[2]), abs(self.last_diffs[2]))), 100))


        abs_logic = abs(self.last_diffs[0]) < abs(fit_diff[0]) and \
                    abs(self.last_diffs[1]) < abs(fit_diff[1]) and \
                    abs(self.last_diffs[2]) < abs(fit_diff[2])

        print(self.name, thresh_logic, diffs_logic, abs_logic)
        print(self.name, "last:{:>10.5f} {:>10.5f} {:>10.5f}:".format(self.last_diffs[0], self.last_diffs[1], self.last_diffs[2]))
        print(self.name, "diff:{:>10.5f} {:>10.5f} {:>10.5f}:".format(fit_diff[0], fit_diff[1], fit_diff[2]))
        gap = abs(fit_diff - self.last_diffs)
        print(self.name, "gap:{:>10.5f} {:>10.5f} {:>10.5f}:".format(gap[0], gap[1], gap[2]))
        if thresh_logic or diffs_logic:
            self.unvalid_cnt += 1
            print(self.name, 'Abandon', "diff:{:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f}:".\
                  format(fit_diff[0], fit_diff[1], fit_diff[2], len(self.allx)))
            self.detected = False
            breakpoint(True)
            self.th_pid.target += 2
            # self.last_fit = self.current_fit * 0.8 + self.best_fit * 0.2
            # self.current_fit = self.last_fit
            if self.fit_cnt > 0:
                self.fit_cnt -= 1
            return False
        else:
            self.detected = True
            # self.store_current_fit()
            if self.unvalid_cnt > 0:
                self.unvalid_cnt -= 1
            self.fit_cnt += 1
            if self.th_pid.target > 1.9:
                self.th_pid.target = max((np.floor(self.th_pid.target - 1), 0.5))
            return True

    def clean_deque(self):
        self.fit_deque.clear()
        self.recent_xfitted.clear()
        self.recent_yfitted.clear()
        # self.last_fit = None
        self.curvature_deque.clear()

    def re_detected(self):
        if self.unvalid_cnt > self.queue_len//3:
            self.unvalid_cnt = 0
            self.fit_cnt = 0
            self.th_pid.target = 1
            print(self.name, self.re_detected.__name__)
            # last_fit = self.fit_deque[-1]
            # if len(self.fit_deque) > 0:
            #     self.fit_deque.popleft()
            # self.fit_deque.append(last_fit)
            # self.fit_deque.append(self.best_fit)
            # self.recent_xfitted.clear()
            # self.recent_yfitted.clear()
            # self.last_fit = None
            # self.best_fit = None
            # self.curvature_deque.clear()
            return True
        return False


    def fit_xy(self):

        self.cal_bestx()
        self.cal_best_fit()
        self.cal_line_base_pos()
        self.cal_radius_of_curvature()
        return self.bestx, self.line_base_pos, self.radius_of_curvature, self.best_fit


if __name__ == '__main__':
    # Generate some fake data to represent lane-line pixels
    ploty = np.linspace(0, 719, num=720)  # to cover same y-range as image
    quadratic_coeff = 3e-4  # arbitrary quadratic coefficient
    # For each y position generate random x position within +/-50 pix
    # of the line base position in each case (x=200 for left, and x=900 for right)
    leftx = np.array([200 + (y ** 2) * quadratic_coeff + np.random.randint(-50, high=51)
                      for y in ploty])
    rightx = np.array([900 + (y ** 2) * quadratic_coeff + np.random.randint(-50, high=51)
                       for y in ploty])

    x = Line('left')
    y = Line('right')

    print(x.valid_xy(leftx, ploty))
    print(x.is_detected())
    print(x.cal_current_fit())
    print(x.is_detected())
    print(x.cal_line_base_pos())
    print(x.cal_radius_of_curvature())
    x.set_allx(leftx)
    x.set_ally(ploty)
    y.set_allx(rightx)
    y.set_ally(ploty)
    # print(x.cal_best_fit())
    print(x.valid_xy(leftx, ploty))
    print(x.cal_bestx())
    print(y.cal_bestx())
    print(x.is_detected())
    print(x.cal_radius_of_curvature())

    x.cal_current_fit()
    y.cal_current_fit()

    x.store_current_fit()
    x.store_current_fit()
    x.store_current_fit()
    y.store_current_fit()

    print(x.cal_diff())
    print(y.cal_diff())

    print(x.cal_radius_of_curvature())
    print(y.cal_radius_of_curvature())

    print(x.cal_bestx())
    print(y.cal_bestx())

    print(x.is_detected())

    print(x.cal_best_fit())
    # print(y.cal_best_fit())

    print(x.valid_xy(leftx, ploty))