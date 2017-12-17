from line import *
from preprocess import *


def pipeline(img, m, v):
    if m.redetect_cnt > 3:
        m.redetect_cnt -= 1
        m.redetect = True

    if m.redetect and (m.redetect_cnt != -1):
        m.left_line.fit_cnt = 0
        m.right_line.fit_cnt = 0
        v.redetect = True
        m.using_min_M = True
        m.fit_well = 0
    if m.using_min_M and not parameters['color_sw']:
        v.M = parameters['M_min']
        v.MInv = parameters['MInv_min']
        obj = v
    else:
        obj = m

    warp_img, undist_img, wy_img, gray, color, combined, edge, undist_img_enhance, rgb_img, hls_img \
        = preprocessing(img, obj.M)
    ret, fit_lane_img = obj.fit_lane(warp_img)

    if m.using_min_M:
        m_warp, *_ = preprocessing(img, m.M)
        ret, _ = m.fit_lane(m_warp)
        if m.fit_well > 3:
            m.using_min_M = False

    scale = 0.3
    resize_shape = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    project = obj.project_back(undist_img)
    warp_project = warp_perspective(project, obj.M, parameters['mtx'], parameters['dist'])
    show_text(project, parameters)
    project_img = np.copy(project) #resize_image(project, (640, 360), '')
    cv2.imshow('a', project_img)
    resize_origin_img = resize_image(img, resize_shape, '1. Original')
    resize_project_img = resize_image(project, resize_shape, '11. Projection')
    resize_color_img = resize_image(wy_img, resize_shape, '4. Yellow & White 1st Filter')
    resize_combined_img = resize_image(combined, resize_shape, '8. Combination')
    resize_combine_warped_img = resize_image(warp_img, resize_shape, '9. Combined Warp')
    resize_enhance_img = resize_image(undist_img_enhance, resize_shape, '3. Enhanced')
    resize_project_warped = resize_image(warp_project, resize_shape, '12. Project Warped')
    resize_undisort_img = resize_image(undist_img, resize_shape, '2. Undisort')
    resize_fit_lane_img = resize_image(fit_lane_img, resize_shape, '10. Lane Fit')
    resize_edge_img = resize_image(edge, resize_shape, '7. Sobel Filter On Enhanced')
    resize_rgb_img = resize_image(rgb_img, resize_shape, '5. RGB Filter On Y&W')
    resize_hls_img = resize_image(hls_img, resize_shape, '6. HLS Filter On Y&W')

    debug_img = np.vstack((np.hstack((resize_origin_img, resize_rgb_img, resize_combine_warped_img)), \
                           np.hstack((resize_undisort_img, resize_hls_img, resize_fit_lane_img)), \
                           np.hstack((resize_enhance_img, resize_edge_img, resize_project_img)), \
                           np.hstack((resize_color_img, resize_combined_img, resize_project_warped))))

    return debug_img, project_img


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
        self.margin = parameters['margin']
        self.left_fit = None
        self.right_fit = None
        self.offset = None
        self.left_bestx = None
        self.right_bestx = None
        self.left_radius_of_curvature = None
        self.right_radius_of_curvature = None
        self.valid_both_lanes_cnt = 0
        self.minpix = 100
        self.left_confidence = True
        self.right_confidence = True
        self.distance_queue = deque(maxlen=10)
        self.use_color = True
        self.using_min_M = False
        self.fit_well = 0

    def find_lane(self):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(self.binary_warped[self.binary_warped.shape[0] // 2:, :], axis=0)  # TODO: using avg center
        # Create an output image to draw on and  visualize the result
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)  # TODO
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
        self.leftx = leftx  # nonzerox[left_lane_inds]
        self.lefty = nonzeroy[left_lane_inds]
        self.rightx = rightx  # nonzerox[right_lane_inds]
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
        fit_lane_img = cv2.addWeighted(out_img, 1, window_img, 0.5, 0)

        return fit_lane_img

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
        new_unwarp = cv2.warpPerspective(color_warp, self.MInv, (orig.shape[1], orig.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(orig, 1, new_unwarp, 0.6, 0)
        return result

    def show_zero_img(self, img):
        print('cannot find both')
        z = np.zeros_like(img)
        return np.zeros_like(img)

    def fit_lane(self, warped_img):
        self.binary_warped = warped_img
        if self.redetect:
            self.left_confidence = False
            self.right_confidence = False
            self.left_line.last_fit = self.left_line.best_fit
            self.right_line.last_fit = self.right_line.best_fit
            if parameters['color_sw']:
                use_color(True)
            else:
                if parameters['brightness'] > 90:
                    use_color(True)
                else:
                    use_color(False)
            print('fit_lane redetect')
            if not self.find_lane():
                return False, self.show_zero_img(warped_img)
        else:
            if not self.find_lane_skip_window():
                self.redetect_cnt += 1
                return False, self.show_zero_img(warped_img)

        self.redetect = False
        self.fitxy()

        try:
            self.offset = self.left_line_base_pos + self.right_line_base_pos
            self.left_fit, self.right_fit = self.fit_smoothing(self.left_line, self.right_line)
            print('offset: {:>9.4} {:>9.4} {:>9.4}'.format(self.left_line_base_pos, self.right_line_base_pos,
                                                           self.offset))
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

    def verify_redetected_current_fit(self, left_line, right_line):
        if self.left_confidence and self.right_confidence:
            return 0
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
                self.distance_queue.clear()
            else:
                left_line.last_fit = left_line.best_fit
                right_line.last_fit = right_line.best_fit

            return 1

    def verify_both_lanes(self, left_line, right_line):
        ret = self.verify_redetected_current_fit(left_line, right_line)
        if ret > 0:
            breakpoint(ret > 0)
            if ret == 1:
                print("Fit: Redetected Current")
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
        valid_y = np.array([max_y, max_y // 2, 0])
        if len(self.distance_queue) > 0:
            lane_space = np.mean(self.distance_queue, axis=0)
        else:
            lane_space = np.array([600, 600, 600, 650])

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
        print('Curr Gap: ', lane_current_distance - lane_space)
        print('Distance Best: ', lane_best_distance, avg_best_distance)
        print('Best Gap: ', lane_best_distance - lane_space)
        lane_distance = lane_current_distance * 0.8 + lane_best_distance * 0.2

        # True: abandon self.left_fit and self.right_fit
        abandon_last_used_fit = False
        # True: abandon left_line.current_fit and right_line.current_fit
        abandon_line_current_fit = False


        for i, distance in enumerate(lane_current_distance[:-1]):
            # filter current cross line, large spacing and center point offset
            if distance < 0 or abs(distance - avg_current_distance) > 400 or avg_current_distance > 800:
                abandon_last_used_fit = True
                abandon_line_current_fit = True
                break
            # filter bad current fit with smaller constrains
            elif abs(distance - avg_best_distance) > 300:
                abandon_line_current_fit = True
                break
            else:
                continue

        # filer small center spacing at bottom
        if current_middle + 100 > right_fitx[0] or current_middle - 100 < left_fitx[0]:
            abandon_line_current_fit = True

        # filter like hyperbolic curve
        mid = lane_current_distance[1]
        margin = 50
        if (lane_current_distance[0] - mid > margin) and \
                (lane_current_distance[2] - mid > margin):
            abandon_line_current_fit = True

        left_curv = min(self.left_line.radius_of_curvature, 1000)
        right_curv = min(self.right_line.radius_of_curvature, 1000)

        # filter large curvature difference between two lane
        if left_curv > 5 * right_curv or right_curv > 5 * left_curv:
            abandon_last_used_fit = True

        if abandon_last_used_fit:
            print("FIT: 1 fit")
            left_fit = left_line.best_fit
            right_fit = right_line.best_fit
            self.redetect_cnt += 1
            self.fit_well = 0
            self.distance_queue.append(lane_distance)
        elif abandon_line_current_fit:
            print("FIT: 2 fit")
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
            print("FIT: 3 fit")
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

                parameters['offset'] = self.offset
                parameters['left_curverad'] = self.left_radius_of_curvature \
                    if self.left_radius_of_curvature is not None else 0
                parameters['right_curverad'] = self.right_radius_of_curvature \
                    if self.right_radius_of_curvature is not None else 0
            self.distance_queue.append(lane_current_distance)

        return left_fit, right_fit

    def fit_smoothing(self, left_line, right_line):
        try:
            left_fit, right_fit = self.verify_both_lanes(left_line, right_line)
        except:
            left_fit = self.left_fit
            right_fit = self.right_fit
        return left_fit, right_fit
