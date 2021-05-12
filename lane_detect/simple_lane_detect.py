import cv2
import math
import numpy as np
# import imageio
# from save_GIF import FrameGIF

debug_img = True

def display_window(filename, img):
  if not debug_img:
    return
  cv2.imshow(filename, img)
  cv2.waitKey(0)

def to_gray_scale(img):
  return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def reduce_noise(img, ksize):
  return cv2.GaussianBlur(img, (ksize, ksize), 0)

def interest_region(img, roi):
  if len(img.shape) > 2:
    mask_color = (255, ) * img.shape[2]
  else:
    mask_color = 255
  # height, width, channel = img.shape
  # dst = np.zeros((height, width, channel), dtype=np.uint8)
  dst = np.zeros_like(img)
  cv2.fillPoly(dst, [roi], mask_color)
  return cv2.bitwise_and(img, dst)

def edge_detector(img, low_thresh=100, high_thresh=200):
  return cv2.Canny(img, low_thresh, high_thresh)

def hough_line_detecor(img):
  lines = cv2.HoughLinesP(img, 1, np.pi/180, 30, np.array([]), 20, 20)
  img_copy = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
  mean_lines = mean_coordinate(img, lines)
  draw_lines(img_copy, mean_lines)
  return img_copy

def make_coordinate(parameter, y_max, y_min):
  x1_mean = int((y_max - parameter[1]) / parameter[0])
  x2_mean = int((y_min - parameter[1]) / parameter[0])
  return np.array([x1_mean, y_max, x2_mean, y_min])

def mean_coordinate(img, lines):
  left_fit = []
  right_fit = []
  y_max = img.shape[0]
  y_min = img.shape[0]

  for line in lines:
    x1, y1, x2, y2 = line[0] 
    y_min = min(min(y1, y2), y_min)
    parameter = np.polyfit((x1, x2), (y1, y2), 1)  ## y = ax + b, 兩點求直線公式
    slope = parameter[0]  
    intercept = parameter[1] 
    if abs(slope) < 0.5:  
      continue 
    if slope < 0:  ## 判斷斜率來表示左右
      left_fit.append((slope, intercept))
    else:
      right_fit.append((slope, intercept))

  if len(left_fit) > 0 and len(right_fit) > 0:
    left_fit_mean = np.mean(left_fit, axis=0)
    right_fit_mean = np.mean(right_fit, axis=0)
    
    left_coordinate = make_coordinate(left_fit_mean, y_max, y_min)
    right_coordinate = make_coordinate(right_fit_mean, y_max, y_min)
    return np.array([left_coordinate, right_coordinate])
  return

def draw_lines(img, lines):
  if lines is not None:
    for x1, y1, x2, y2 in lines:
      # x1, y1, x2, y2 = line[0]
      cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
  return cv2.addWeighted(initial_img, α, img, β, λ)



if __name__ in "__main__":

  cap = cv2.VideoCapture("../test_videos/project_video.mp4")

  # gif_frame = FrameGIF(60)
  while True:
    if cap.isOpened():
      ret, frame = cap.read()
      if not ret:
        break
      frame_copy = frame.copy()
      height, width, _ = frame.shape
      # roi = np.array([[int(width*4/10), int(height*6.5/10)], [int(width*1.8/10), int(height)],
      #                 [int(width*9.4/10), height], [int(width*6.1/10), int(height*6.5/10)]])
      roi = np.array([(0, height),(width / 2, height / 2),(width, height), ], np.int32)
      img_gray = to_gray_scale(frame)
      # display_window('gray', img_gray)
      img_blur = reduce_noise(img_gray, 3)
      # display_window('blur', img_blur)
      img_edge = edge_detector(img_blur)
      # display_window('edge', img_edge)
      # img_mo = cv2.morphologyEx(img_edge,cv2.MORPH_OPEN,(3,3), iterations=1)
      # img_mo =  cv2.dilate(img_mo,(15,15),iterations=5)
      # display_window("interest_img", img_mo)
      interest_img = interest_region(img_edge, roi)
      # display_window('interest_region', interest_img)
      # display_window("interest_img", interest_region(img_blur, roi))
      img_line = hough_line_detecor(interest_img)
      # display_window('hough_line_detecor', img_line)
      # display_window("img_line", img_line)
      line_detect = weighted_img(frame_copy, img_line)
      result_image = cv2.hconcat([img_line, line_detect])
      cv2.imshow("line_detect", result_image)
      imm = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

      # gif_frame.append_frame(imm)
      # gif_frame.save_gif('./test.gif')
      # cv2.waitKey(0)
      # type q to left loop
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    else:
      print("No video")
      break
  # Release everything if job is finished

  
  cap.release()
  cv2.destroyAllWindows()