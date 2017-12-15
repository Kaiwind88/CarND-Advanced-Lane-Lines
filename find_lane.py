import numpy as np
import cv2
from line import *
from preprocess import *

class Lane():
    def __init__(self, parameters):
        self.leftx = None
        self.lefty = None
        self.rightx = None
        self.righty = None
        self.binary_warped = None
        self.left_line = Line('left')
        self.right_line = Line('right')
        self.redetect = True
        self.redetect_cnt = -1
        self.M = parameters['M']
        self.MInv = parameters['MInv']
        self.margin = 55
        self.left_fit = None
        self.right_fit = None
        self.offset = None
        self.left_bestx = None
        self.right_bestx = None
        self.left_radius_of_curvature = None
        self.right_radius_of_curvature = None
        self.valid_both_lanes_cnt = 0
        self.minpix = 150
        self.left_confidence = True
        self.right_confidence = True
        self.distance_queue = deque(maxlen=10)
        self.use_color = True
        self.using_min_M = False
        self.fit_well = 0

    def find_lane(self):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(self.binary_warped[self.binary_warped.shape[0] // 2:, :], axis=0) #TODO: using avg center
        # Create an output image to draw on and  visualize the result
        # out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)#TODO
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(self.binary_warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = self.binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = self.margin
        # Set minimum number of pixels found to recenter window
        minpix = self.minpix
        maxpix = margin * self.binary_warped.shape[0]
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        left_hit = 0
        right_hit = 0
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = self.binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = self.binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            good_left_inds_len = len(good_left_inds)
            good_right_inds_len = len(good_right_inds)
            print('good inds len ', good_left_inds_len, good_right_inds_len)
            if good_left_inds_len > minpix:
                left_hit += 1
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if good_right_inds_len > minpix:
                right_hit += 1
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        if left_hit < 2 or right_hit < 2:
            self.redetect = True
            return False
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        leftx = nonzerox[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        middle = (np.mean(rightx) + np.mean(leftx)) / 2
        if 100 < abs(650 - middle):
            print("find_lane distance failed:{:>.2f}".format(middle))
            return False

        # Extract left and right line pixel positions
        self.leftx = leftx #nonzerox[left_lane_inds]
        self.lefty = nonzeroy[left_lane_inds]
        self.rightx = rightx #nonzerox[right_lane_inds]
        self.righty = nonzeroy[right_lane_inds]

        return True

    def find_lane_skip_window(self):
        nonzero = self.binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = self.margin

        left_fit = self.left_fit
        right_fit = self.right_fit

        if left_fit is None or right_fit is None:
            return False

        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                        left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                        left_fit[1] * nonzeroy + left_fit[2] + margin)))

        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                        right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                        right_fit[1] * nonzeroy + right_fit[2] + margin)))

        minpix = self.minpix * 5
        leftx = nonzerox[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        if len(leftx) < minpix or len(rightx) < minpix:
            print('Too little points', len(leftx), len(rightx))
            # self.leftx = 0
            # self.lefty = 0
            # self.rightx = 0
            # self.righty = 0
            return False
        else:
            # Again, extract left and right line pixel positions
            self.leftx = nonzerox[left_lane_inds]
            self.lefty = nonzeroy[left_lane_inds]
            self.rightx = nonzerox[right_lane_inds]
            self.righty = nonzeroy[right_lane_inds]

            return True

    def show_lane(self):
        binary_warped = self.binary_warped
        left_fit = self.left_fit
        right_fit = self.right_fit
        margin = self.margin

        if self.left_fit is None or self.right_fit is None:
            return np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[self.lefty, self.leftx] = [255, 0, 0]
        out_img[self.righty, self.rightx] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        # when you are drawing a polygon you need to give points in a consecutive
        # manner so you go from left bottom to left top and then you need to go from
        # right TOP to right BOTTOM. Hence you need to flip the right side polynomial
        # to make them go from top to bottom.
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.5, 0)

        return result

    def project_back(self, orig):
        if self.MInv is None or self.left_fit is None or self.right_fit is None:
            return orig
        binary_warped = self.binary_warped
        left_fit = self.left_fit
        right_fit = self.right_fit
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.MInv, (orig.shape[1], orig.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(orig, 1, newwarp, 0.6, 0)
        return result

    def show_default_img(self, img):
        print('cannot find both')
        z = np.zeros_like(img)
        # out_img = np.dstack((z, z, z)) * 255
        return np.zeros_like(img)


    def fit_lane(self, warped_img):
        self.binary_warped = warped_img
        if self.redetect:
            self.left_confidence = False
            self.right_confidence = False
            self.left_line.last_fit = self.left_line.best_fit
            self.right_line.last_fit = self.right_line.best_fit
            if parameters['brightness'] > 90:
                use_color(True)
            else:
                use_color(False)
            print('fit_lane redetect')
            # self.distance_queue.clear()
            if not self.find_lane():
                return False, self.show_default_img(warped_img)
        else:
            if not self.find_lane_skip_window():
                self.redetect_cnt += 1
                return False, self.show_default_img(warped_img)

        self.redetect = False

        redetect = self.fitxy()
        # if redetect:
        #     self.redetect = False
        try:
            self.offset = self.left_line_base_pos + self.right_line_base_pos
            self.left_fit, self.right_fit = self.fit_smoothing(self.left_line, self.right_line)
            # print('left best fit: {:>9.4} {:>9.4} {:>9.4}'.format(self.left_fit[0], self.left_fit[1], self.left_fit[2]))
            # print('right best fit: {:>9.4} {:>9.4} {:>9.4}'.format(self.right_fit[0], self.right_fit[1], self.right_fit[2]))
            # print('offset: {:>9.4} {:>9.4} {:>9.4}'.format(self.left_line_base_pos, self.right_line_base_pos, self.offset))
            # print('curvature: {:>9.4} {:>9.4}'.format(self.left_radius_of_curvature, self.right_radius_of_curvature))
        except:
            print('offset', None)

        return True, self.show_lane()

    def fitxy(self):
        left_val = self.left_line.valid_xy(self.leftx, self.lefty)
        right_val = self.right_line.valid_xy(self.rightx, self.righty)
        if left_val:
            self.left_bestx, self.left_line_base_pos, self.left_radius_of_curvature, _ = self.left_line.fit_xy()
        else:
            if self.left_line.re_detected():
                self.redetect_cnt += 1
        if right_val:
            self.right_bestx, self.right_line_base_pos, self.right_radius_of_curvature, _ = self.right_line.fit_xy()
        else:
            if self.right_line.re_detected():
                self.redetect_cnt += 1

        if left_val and right_val:
            return True
        else:
            return False

    def get_lane_prop(self):
        return self.offset, self.left_bestx, self.right_bestx, self.left_radius_of_curvature, self.right_radius_of_curvature

    def valid_redetected_current_fit(self, left_line, right_line):
        if self.left_confidence and self.right_confidence:
            return 0#both confident, use best_fit
        max_y = self.binary_warped.shape[0]
        valid_y = np.array([max_y, max_y // 2])

        left_fit = left_line.current_fit
        right_fit = right_line.current_fit
        left_fitx = left_fit[0] * valid_y ** 2 + left_fit[1] * valid_y + left_fit[2]
        right_fitx = right_fit[0] * valid_y ** 2 + right_fit[1] * valid_y + right_fit[2]

        lane_current_distance = right_fitx - left_fitx
        avg_distance = np.mean(lane_current_distance)
        print("Redetect Distance:", lane_current_distance, avg_distance)
        abandon_current_fit = False
        for i, distance in enumerate(lane_current_distance):
            if distance < 300 or abs(avg_distance - distance) > 100:
                abandon_current_fit = True
                break

        if abandon_current_fit:
            return 2
        else:
            if left_line.detected and right_line.detected:
                left_line.last_fit = left_line.current_fit
                left_line.store_current_fit()

                right_line.last_fit = right_line.current_fit
                right_line.store_current_fit()

                self.right_confidence = True
                self.left_confidence = True

                left_line.clean_deque()
                right_line.clean_deque()
                left_line.best_fit = left_line.current_fit
                right_line.best_fit = right_line.current_fit
                # self.left_fit = left_line.current_fit
                # self.right_fit = right_line.current_fit
                self.distance_queue.clear()
            else:
                left_line.last_fit = self.left_line.best_fit
                right_line.last_fit = self.right_line.best_fit

            return 1


    def valid_both_lanes(self, left_line, right_line):
        ret = self.valid_redetected_current_fit(left_line, right_line)
        if ret > 0 :
            breakpoint(ret > 0)
            if ret == 1:
                print("Fit: Redetected Current")
                # return left_line.current_fit, right_line.current_fit
                if left_line.detected and right_line.detected:
                    return left_line.current_fit, right_line.current_fit
                elif left_line.detected:
                    return left_line.current_fit, self.right_fit
                elif right_line.detected:
                    return self.left_fit, right_line.current_fit
                else:
                    return self.left_fit, self.right_fit
            elif ret == 2:
                print("Fit: Redetected Defatult")
                return self.left_fit, self.right_fit
            elif ret == 3:
                print("Fit: Redetected Best")
                return left_line.best_fit, right_line.best_fit
        else:
            print("Both Confidence")
            pass

        max_y = self.binary_warped.shape[0]
        valid_y = np.array([max_y, max_y//2, 0])
        if len(self.distance_queue) > 0:
            lane_space = np.mean(self.distance_queue, axis=0)
        else:
            lane_space = np.array([600, 600, 600, 650])
        bad_current_margin = np.array([150,250,350,80])
        good_current_margin = np.array([100, 140, 180, 80])

        if left_line.detected and right_line.detected:
            factor_c = abs(left_line.current_fit[2] - right_line.current_fit[2])
            delta = abs(factor_c-650)
            print('Delta: ', delta)
            good_current_margin = good_current_margin + delta
            bad_current_margin = bad_current_margin + delta

        left_fit = left_line.best_fit
        right_fit = right_line.best_fit

        left_fitx = left_fit[0] * valid_y ** 2 + left_fit[1] * valid_y + left_fit[2]
        right_fitx = right_fit[0] * valid_y ** 2 + right_fit[1] * valid_y + right_fit[2]
        best_middle = (left_fitx[0] + right_fitx[0]) / 2
        lane_best_distance = right_fitx - left_fitx
        avg_best_distance = np.mean(lane_best_distance)
        lane_best_distance = np.append(lane_best_distance, best_middle)

        if left_line.detected:
            left_fit = left_line.current_fit
        else:
            left_fit = self.left_fit * 0.5 + left_line.current_fit * 0.5

        if right_line.detected:
            right_fit = right_line.current_fit
        else:
            right_fit = self.right_fit * 0.5 + right_line.current_fit * 0.5


        left_fitx = left_fit[0] * valid_y ** 2 + left_fit[1] * valid_y + left_fit[2]
        right_fitx = right_fit[0] * valid_y ** 2 + right_fit[1] * valid_y + right_fit[2]
        current_middle = (left_fitx[0] + right_fitx[0]) / 2
        lane_current_distance = right_fitx - left_fitx
        avg_current_distance = np.mean(lane_current_distance)
        lane_current_distance = np.append(lane_current_distance, current_middle)

        print('Distance Curr: ', lane_current_distance, avg_current_distance)
        print('Curr Gap: ', lane_current_distance-lane_space)
        print('Distance Best: ', lane_best_distance, avg_best_distance)
        print('Best Gap: ', lane_best_distance-lane_space)
        lane_distance = lane_current_distance * 0.8 + lane_best_distance * 0.2
        # print('lane_distance', lane_distance)

        # self.distance_queue.append(lane_distance * 0.7 + np.array([600, 600, 600, 650]) * 0.3)

        abandon_best_fit = False
        abandon_current_fit = False
        # for i, distance in enumerate(lane_best_distance):
        #     if distance < 0 or abs(distance - lane_space[i]) > bad_current_margin[i]:
        #         abandon_best_fit = True
        #         abandon_current_fit = True
        #         break
        #     elif abs(distance - lane_space[i]) > good_current_margin[i]:
        #         abandon_current_fit = True
        #         break
        #     else:
        #         continue


        for i, distance in enumerate(lane_current_distance[:-1]):
            if distance < 0 or abs(distance - avg_current_distance) > 400 or avg_current_distance > 800:
                abandon_best_fit = True
                abandon_current_fit = True
                break
            elif abs(distance - avg_best_distance) > 300:
                abandon_current_fit = True
                break
            else:
                continue

        if current_middle + 100 > right_fitx[0] or current_middle - 100 < left_fitx[0]:
            abandon_current_fit = True

        # mid = lane_best_distance[1]
        # margin = 120
        # if (lane_best_distance[0]-mid > margin) or \
        #         (lane_best_distance[2]-mid > margin):
        #     abandon_best_fit = True

        mid = lane_current_distance[1]
        margin = 100
        if (lane_current_distance[0]-mid > margin) and \
                    (lane_current_distance[2]-mid > margin):
            abandon_current_fit = True
        else:
            for distance in lane_current_distance:
                if distance < 0:
                    abandon_current_fit = True

        left_curv = min(self.left_line.radius_of_curvature, 1000)
        right_curv = min(self.right_line.radius_of_curvature, 1000)

        if left_curv > 5 * right_curv or right_curv > 5 * left_curv:
            abandon_best_fit = True

        # if (self.left_line.radius_of_curvature > 1000 and self.right_line.radius_of_curvature < 500) or \
        #         (self.left_line.radius_of_curvature < 500 and self.right_line.radius_of_curvature > 1000):
        #     if not self.left_line.detected or not self.right_line.detected:
        #         abandon_best_fit = True


        # if abandon_best_fit and abandon_current_fit:
        #     print("FIT: 1 fit")
        #     left_fit = self.left_fit
        #     right_fit = self.right_fit
        #     left_line.last_fit = left_fit
        #     right_line.last_fit = right_fit
        #     self.redetect_cnt += 1
        #     self.distance_queue.append(lane_best_distance * 0.7 + np.array([600, 600, 600, 650]) * 0.3)
        if abandon_best_fit:
            print("FIT: 2 fit")
            left_fit = self.left_fit #left_line.best_fit
            right_fit = self.right_fit #right_line.best_fit
            # left_line.last_fit = left_line.best_fit
            # right_line.last_fit = right_line.best_fit
            self.redetect_cnt += 1
            self.fit_well = 0
            self.distance_queue.append(lane_distance)
        elif abandon_current_fit:
            print("FIT: 3 fit")
            factor = 0.1
            if True or left_line.detected:
                left_fit = self.left_fit
                left_line.last_fit = self.left_fit
            else:
                left_fit = self.left_fit
                left_line.last_fit = left_line.best_fit

            if True or right_line.detected:
                right_fit = self.right_fit
                right_line.last_fit = self.left_fit
            else:
                right_fit = self.right_fit
                right_line.last_fit = right_line.best_fit
            self.redetect_cnt += 1
            self.fit_well = 0
            self.distance_queue.append(lane_distance)
        else:
            factor = 0.2
            print("FIT: 4 fit")
            if left_line.detected and right_line.detected:
                if left_line.fit_cnt > 0:
                    left_fit = left_line.current_fit
                    left_line.store_current_fit()
                else:
                    left_fit = self.left_fit
                left_line.last_fit = left_line.current_fit

                if right_line.fit_cnt > 0:
                    right_fit = right_line.current_fit
                    right_line.store_current_fit()
                else:
                    right_fit = self.right_fit
                right_line.last_fit = right_line.current_fit
            else:
                left_line.last_fit = self.left_fit
                left_fit = self.left_fit
                right_line.last_fit = self.right_fit
                right_fit = self.right_fit

            if left_line.detected and right_line.detected:
                if self.redetect_cnt > 0:
                    self.redetect_cnt -= 1
                self.fit_well += 1
            self.distance_queue.append(lane_current_distance)

        return left_fit, right_fit

    def fit_smoothing(self, left_line, right_line):
        try:
            left_fit, right_fit = self.valid_both_lanes(left_line, right_line)
        except:
            left_fit = self.left_fit
            right_fit = self.right_fit
        return left_fit, right_fit


    def lane_detection(self, img):
        if self.redetect_cnt > 8:
            self.redetect_cnt -= 1
            self.redetect = True

        if self.redetect and (self.redetect_cnt != -1):
            self.left_line.fit_cnt = 0
            self.right_line.fit_cnt = 0
            self.window_size = 20
            self.using_min_M = True
        else:
            if self.redetect_cnt == 0:
                self.using_min_M = False

        if self.using_min_M:
            self.M = parameters['M_min']
            self.MInv = parameters['MInv_min']
            if self.window_size > 0:
                self.window_size -= 1
            else:
                self.using_min_M = False
        else:
            self.M = parameters['M']
            self.MInv = parameters['MInv']
        unwarp_img, undist_img, wy_img, gray, color, combined, edge, undist_img_enhance = pipline(img, self.M)
        ret, fit_lane_img = self.fit_lane(unwarp_img)

        offset, left_bestx, right_bestx, left_curverad, right_curverad = self.get_lane_prop()

        parameters['left_curverad'] = left_curverad if left_curverad is not None else 0
        parameters['right_curverad'] = right_curverad if right_curverad is not None else 0

        scale = 0.5
        resize_shape = (int(img.shape[1] * scale), int(img.shape[0] * scale))
        project = self.project_back(undist_img)
        unwarp_project = unwarp(project, self.M, parameters['mtx'], parameters['dist'])
        show_text(project, parameters)

        resize_project_img = resize_image(project, resize_shape, 'Projected')
        resize_color_img = resize_image(wy_img, resize_shape, 'Yellow & White')
        resize_combined_img = resize_image(combined, resize_shape, 'Combined')
        resize_unwarped_img = resize_image(unwarp_img, resize_shape, 'Combined Unwarped')
        resize_enhance_img = resize_image(undist_img_enhance, resize_shape, 'Enhanced')
        resize_color_unwarped = resize_image(unwarp_project, resize_shape, 'Unwarped')
        resize_undist_img = resize_image(undist_img, resize_shape, 'Undist')
        resize_fit_lane_img = resize_image(fit_lane_img, resize_shape, 'Lane')
        resize_edge_img = resize_image(edge, resize_shape, 'Sobel Edge')

        final_img = np.vstack((np.hstack((resize_color_img, resize_fit_lane_img, resize_undist_img)), \
                               np.hstack((resize_combined_img, resize_unwarped_img, resize_enhance_img)),
                               np.hstack((resize_project_img, resize_color_unwarped, resize_edge_img))))

        return final_img, project





    def fit_smoothing_abondon(self):
        pass
        # left_weight = np.array(left_line.n_x_deque)
        # right_weight = np.array(right_line.n_x_deque)
        # left_fit = np.array(left_line.fit_deque)
        # right_fit = np.array(right_line.fit_deque)
        # n_left = len(left_weight)
        # n_right = len(right_weight)
        # gap = abs(n_left - n_right)
        # if gap != 0:
        #     if n_left < n_right:
        #         left_weight = np.append(left_weight, np.zeros(gap))
        #         left_fit = np.append(left_fit, np.zeros((gap, 3)), axis=0)
        #     else:
        #         right_weight = np.append(right_weight, np.zeros(gap))
        #         right_fit = np.append(right_fit, np.zeros(gap, 3), axis=0)
        #
        # left_w = left_weight / (left_weight + right_weight)
        # left_w = left_w.reshape((-1, 1))
        # right_w = right_weight / (left_weight + right_weight)
        # right_w = right_w.reshape((-1, 1))
        # left_fit = left_fit[:, :-1]
        # right_fit = right_fit[:, :-1]
        # left_fit = left_fit * left_w
        # right_fit = right_fit * right_w
        # both_fit = left_fit + right_fit
        # both_fit = np.mean(both_fit, axis=0, dtype=np.float32)
        # left_best_fit = np.append(both_fit, left_line.best_fit[2])
        # right_best_fit = np.append(both_fit, right_line.best_fit[2])

        # left_weight = np.mean(left_line.n_x_deque)
        # right_weight = np.mean(right_line.n_x_deque)
        # left_fit = left_line.best_fit
        # right_fit = right_line.best_fit
        # print('left best fit: {:>9.4} {:>9.4} {:>9.4}'.format(self.left_fit[0], self.left_fit[1], self.left_fit[2]))
        # print('right best fit: {:>9.4} {:>9.4} {:>9.4}'.format(self.right_fit[0], self.right_fit[1], self.right_fit[2]))
        # left_w = left_weight / (left_weight + right_weight)
        # right_w = right_weight / (left_weight + right_weight)
        # print('left_weight: {:>9.4}'.format(left_w))
        # print('right_weight: {:>9.4}'.format(right_w))
        # left_fit1 = left_fit[:-1] * left_w
        # right_fit1 = right_fit[:-1] * right_w
        # both_fit = left_fit1 + right_fit1
        # print('left best fit1: {:>9.4} {:>9.4}'.format(left_fit1[0], left_fit1[1]))
        # print('right best fit1: {:>9.4} {:>9.4}'.format(right_fit1[0], right_fit1[1]))
        # l_c = self.left_line.best_fit[2]
        # r_c = self.right_line.best_fit[2]
        # # l_c = self.offset + 220
        # # r_c = self.offset + 900
        # # l_c = (self.left_line.bestx + self.left_line.best_fit[2]) / 2 + self.offset
        # # r_c = (self.right_line.bestx + self.right_line.best_fit[2]) / 2 + self.offset
        # left_best_fit = np.append(both_fit, l_c)
        # right_best_fit = np.append(both_fit, r_c)
        # print('left fit: {:>9.4} {:>9.4} {:>9.4}'.format(left_best_fit[0], left_best_fit[1], left_best_fit[2]))
        # print('right fit: {:>9.4} {:>9.4} {:>9.4}'.format(right_best_fit[0], right_best_fit[1], right_best_fit[2]))

        # return left_best_fit, right_best_fit
        # return left_line.best_fit, right_line.best_fit


def find_lane(img, m, v):
    if m.redetect_cnt > 3:
        m.redetect_cnt -= 1
        m.redetect = True

    if m.redetect and (m.redetect_cnt != -1):
        m.left_line.fit_cnt = 0
        m.right_line.fit_cnt = 0
        v.window_size = 5
        v.redetect = True
        m.using_min_M = True
        m.fit_well = 0
    if m.using_min_M:
        v.M = parameters['M_min']
        v.MInv = parameters['MInv_min']
        obj = v
    else:
        obj = m

    warp_img, undist_img, wy_img, gray, color, combined, edge, undist_img_enhance, rgb_img, hls_img = pipline(img, obj.M)
    ret, fit_lane_img = obj.fit_lane(warp_img)

    if m.using_min_M:
        m_unwarp, *_ = pipline(img, m.M)
        ret, _ = m.fit_lane(m_unwarp)
        if v.window_size > 0:
            v.window_size -= 1
        elif v.window_size < 0:
            v.window_size = 5
        if m.fit_well > 3:
            m.using_min_M = False

    offset, left_bestx, right_bestx, left_curverad, right_curverad = obj.get_lane_prop()

    parameters['left_curverad'] = left_curverad if left_curverad is not None else 0
    parameters['right_curverad'] = right_curverad if right_curverad is not None else 0

    scale = 0.3
    resize_shape = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    project = obj.project_back(undist_img)
    unwarp_project = unwarp(project, obj.M, parameters['mtx'], parameters['dist'])
    show_text(project, parameters)
    output_project = resize_image(project, (640, 360), '')

    resize_origin_img = resize_image(img, resize_shape, '1. Original')
    resize_project_img = resize_image(project, resize_shape, '11. Projection')
    resize_color_img = resize_image(wy_img, resize_shape, '4. Yellow & White 1st Filter')
    resize_combined_img = resize_image(combined, resize_shape, '8. Combination')
    resize_combine_warped_img = resize_image(warp_img, resize_shape, '9. Combined Warp')
    resize_enhance_img = resize_image(undist_img_enhance, resize_shape, '3. Enhanced')
    resize_project_warped = resize_image(unwarp_project, resize_shape, '12. Project Warped')
    resize_undisort_img = resize_image(undist_img, resize_shape, '2. Undisort')
    resize_fit_lane_img = resize_image(fit_lane_img, resize_shape, '10. Lane Fit')
    resize_edge_img = resize_image(edge, resize_shape, '7. Sobel Filter On Enhanced')
    resize_rgb_img = resize_image(rgb_img, resize_shape, '5. RGB Filter On Y&W')
    resize_hls_img = resize_image(hls_img, resize_shape, '6. HLS Filter On Y&W')


    final_img = np.vstack((np.hstack((resize_origin_img, resize_rgb_img, resize_combine_warped_img)), \
                           np.hstack((resize_undisort_img, resize_hls_img, resize_fit_lane_img)), \
                           np.hstack((resize_enhance_img, resize_edge_img, resize_project_img)), \
                           np.hstack((resize_color_img, resize_combined_img, resize_project_warped))))


    return final_img, output_project




def find_lane_cnn(warped):
    # window settings
    window_width = 50
    window_height = 80  # Break image into 9 vertical layers since image height is 720
    margin = 100  # How much to slide left and right for searching

    def window_mask(width, height, img_ref, center, level):
        output = np.zeros_like(img_ref)
        output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
        max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
        return output

    def find_window_centroids(image, window_width, window_height, margin):

        window_centroids = []  # Store the (left,right) window centroid positions per level
        window = np.ones(window_width)  # Create our window template that we will use for convolutions

        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template

        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(image[int(3 * image.shape[0] / 4):, :int(image.shape[1] / 2)], axis=0)
        l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
        r_sum = np.sum(image[int(3 * image.shape[0] / 4):, int(image.shape[1] / 2):], axis=0)
        r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(image.shape[1] / 2)

        # Add what we found for the first layer
        window_centroids.append((l_center, r_center))

        # Go through each layer looking for max pixel locations
        for level in range(1, (int)(image.shape[0] / window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(
                image[int(image.shape[0] - (level + 1) * window_height):int(image.shape[0] - level * window_height), :],
                axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = window_width / 2
            l_min_index = int(max(l_center + offset - margin, 0))
            l_max_index = int(min(l_center + offset + margin, image.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center + offset - margin, 0))
            r_max_index = int(min(r_center + offset + margin, image.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
            # Add what we found for that layer
            window_centroids.append((l_center, r_center))

        return window_centroids

    window_centroids = find_window_centroids(warped, window_width, window_height, margin)

    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
            r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1))] = 255
            r_points[(r_points == 255) | ((r_mask == 1))] = 255

        # Draw the results
        template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
        zero_channel = np.zeros_like(template)  # create a zero color channel
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
        warpage = np.dstack((warped, warped, warped)) * 255  # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)  # overlay the orignal road image with window results

    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped, warped, warped)), np.uint8)

    return output