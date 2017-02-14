from moviepy.editor import VideoFileClip
#from IPython.display import HTML
import numpy as np
from numpy import zeros_like
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

from tracker import tracker

dist_pickle = pickle.load(open("./camera_cal/calibration_pickle.p", 'rb'))
mtx = dist_pickle["mtx"]
dist = dist_pickle['dist']


def abs_sobel_thresh(img, orient='x',sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


# Define a function that thresholds the S-channel of HLS
def color_threshold(img, sthresh=(0, 255), vthresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > sthresh[0]) & (s_channel <= sthresh[1])] = 1

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:, :, 2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel > vthresh[0]) & (v_channel <= vthresh[1])] = 1

    binary_output = np.zeros_like(s_channel)
    binary_output[(s_binary==1) & (v_binary==1)] =1
    return binary_output

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output



# Choose a Sobel kernel size
ksize = 3  #



def process_image (img):

    # undistort the image
    img = cv2.undistort(img, mtx, dist, None, mtx)

    #  Gradient and color threshold
    gradx = abs_sobel_thresh(img, orient='x',sobel_kernel=ksize, thresh=(12, 255))
    grady = abs_sobel_thresh(img, orient='y',sobel_kernel=ksize, thresh=(25, 255))
    color = color_threshold(img, sthresh=(100, 255), vthresh=(50,255))
    combined = zeros_like(img[:, :, 0])
    combined[((gradx == 1) & (grady == 1)) | (color==1)] = 255

    # Image warp
    from matplotlib.widgets import Cursor
    import numpy as np
    import matplotlib.pyplot as plt

    # img_size = (img.shape[1], img.shape[0])
    #
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, axisbg='#FFFFCC')
    #
    # .imshow("fel",img)
    #
    # # set useblit = True on gtkagg for enhanced performance
    # cursor = Cursor(ax, useblit=True, color='red', linewidth=2)
    #
    # plt.show()

    # Image and transformed image coordinates
    src = np.float32([[585, 460], [203, 720], [1127, 720], [695, 460]])
    dst = np.float32([[320, 0], [320, 720], [960, 720], [960, 0]])

    # warped image
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    img_size = (combined.shape[1], combined.shape[0])
    warped = cv2.warpPerspective(combined, M, img_size, flags=cv2.INTER_LINEAR)
    # plt.imshow(binary_warped, cmap='gray')

    window_width = 25
    window_height = 80
    curve_centers = tracker(Mywindow_width =window_width, Mywindow_height=window_height,Mymargin=25, My_ym =10/720, My_xm=4/384, Mysmooth_factor=15)

    window_centroids = curve_centers.find_window_centroids(warped)


    # Points used to draw all the left and right windows
    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)

    leftx = []
    rightx = []

    # Go through each level and draw the windows
    for level in range(0, len(window_centroids)):
        # Window_mask is a function to draw window areas
        leftx.append(window_centroids[level][0])
        rightx.append(window_centroids[level][1])
        l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
        r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)
        # Add graphic points from window mask here to total pixels found
        l_points[(l_points == 255) | ((l_mask == 1))] = 255
        r_points[(r_points == 255) | ((r_mask == 1))] = 255

    # Draw the results
    template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
    zero_channel = np.zeros_like(template)  # create a zero color channle
    template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
    warpage = np.array(cv2.merge((warped, warped, warped)),
                       np.uint8)  # making the original road pixels 3 color channels
    output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)  # overlay the orignal road image with window results

    # # If no window centers found, just display orginal road image
    # else:
    #     output = np.array(cv2.merge((warped, warped, warped)), np.uint8)

    # # Display the final results
    # plt.imshow(output)
    # plt.title('window fitting results')
    # plt.show()

    # fit the lane boundaries to left, right, and center position found
    yvals= range(0,warped.shape[0])

    res_yvals = np.arange(warped.shape[0]-(window_height/2),0,-window_height)

    # Fit a second order polynomial to pixel positions in each fake lane line
    left_fit = np.polyfit(res_yvals, leftx, 2)
    left_fitx = left_fit[0] * yvals * yvals + left_fit[1] * yvals + left_fit[2]
    left_fitx = np.array(left_fitx,np.int32)

    right_fit = np.polyfit(res_yvals, rightx, 2)
    right_fitx = right_fit[0] * yvals * yvals + right_fit[1] * yvals + right_fit[2]
    right_fitx = np.array(right_fitx, np.int32)

    left_lane = np.array(list(zip(np.concatenate((left_fitx-window_width/2,left_fitx[::-1]+window_width/2),axis=0), np.concatenate((yvals,yvals[::-1]), axis=0))),np.int32)

    right_lane = np.array(list(
        zip(np.concatenate((right_fitx - window_width / 2, right_fitx[::-1] + window_width / 2), axis=0),
            np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)

    inner_lane = np.array(list(
        zip(np.concatenate((left_fitx + window_width / 2, right_fitx[::-1] - window_width / 2), axis=0),
            np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)

    road = np.zeros_like(img)
    road_bkg = np.zeros_like(img)
    cv2.fillPoly(road,[left_lane], color=[255,0,0])
    cv2.fillPoly(road, [right_lane], color=[0, 0, 255])
    cv2.fillPoly(road, [inner_lane], color=[0, 255, 0])
    cv2.fillPoly(road_bkg, [left_lane], color=[255, 255, 255])
    cv2.fillPoly(road_bkg, [right_lane], color=[255, 255, 255])

    road_warped = cv2.warpPerspective(road, Minv, img_size,flags=cv2.INTER_LINEAR)
    road_warped_bkg = cv2.warpPerspective(road_bkg, Minv, img_size, flags=cv2.INTER_LINEAR)

    base = cv2.addWeighted(img, 1.0, road_warped_bkg, -1.0, 0)
    result = cv2.addWeighted(base, 1.0,road_warped,.7,0)

    ym_per_pix = curve_centers.ym_per_pix
    xm_per_pix = curve_centers.xm_per_pix

    curve_fit_cr = np.polyfit(np.array(res_yvals,np.float32) * ym_per_pix, np.array(leftx,np.float32) * xm_per_pix, 2)

    curverad = ((1 + (2 * curve_fit_cr[0] * yvals[-1] * ym_per_pix + curve_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * curve_fit_cr[0])

    # Calculate the offset of the car on the road
    camera_center = (left_fitx[-1]+right_fitx[-1])/2
    center_diff = (camera_center-warped.shape[1]/2)*xm_per_pix
    side_pos = 'left'
    if center_diff<=0:
        side_pos = 'right'

    # Draw the text showing curvature, offset and speed
    cv2.putText(result, 'Radius of Curvature = ' + str(round(curverad,3)) + 'm', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    cv2.putText(result, 'Vehicle is ' + str(abs(round(center_diff, 3))) + 'm '+ side_pos+ " of center", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)

    return result

Output_video = 'output1_tracked.mp4'
Input_video = '/Users/kaswani/self_drive_local/CarND-Advanced-Lane-Lines/project_video.mp4'
clip1 = VideoFileClip(Input_video)
videoclip = clip1.fl_image(process_image)
videoclip.write_videofile(Output_video,audio =False)
