import cv2
import glob
import numpy as np
from window_filter import Window, window_slide, filter_with_list, window_mask
import matplotlib.pyplot as plt
from collections import deque
# from save_GIF import FrameGIF

def debug_show(inf, mode='image'):
  if mode == 'image':
    cv2.imshow('frame', inf)
    cv2.waitKey(0)
  if mode == 'curve':
    plt.plot(inf)
    plt.show()

def show_windows(image, windows_list, margin=200):
  out_img = np.stack((image,)*3, axis=-1)
  for window in windows_list:
    x_right = window.x_position + margin
    x_left = window.x_position - margin
    y_up = window.y_end
    y_bottom = window.y_begin
    top_left = (int(x_left), y_up)
    bottom_right = (int(x_right), y_bottom)
    cv2.rectangle(out_img, top_left, bottom_right, (0,255,0), 3) 

  debug_show(out_img)
  return out_img

class DashboardCamear:
  def __init__(self, chessboard_img_files, chessboard_size, lane_shape):
    example_img = cv2.imread(chessboard_img_files[0])
    self.img_size = example_img.shape[:2]
    self.img_height = self.img_size[0]
    self.img_width = self.img_size[1]

    self.camera_matrix, self.dist_coeffs = self.calibrate(chessboard_img_files, chessboard_size)

    top_left, top_right, bottom_left, bottom_right = lane_shape
    source = np.float32([top_left, top_right, bottom_right, bottom_left])
    destination = np.float32([(bottom_left[0], 0), (bottom_right[0], 0),
                                      (bottom_right[0], self.img_height - 1), (bottom_left[0], self.img_height - 1)])
    self.overhead_transform = cv2.getPerspectiveTransform(source, destination)
    self.inverse_overhead_transform = cv2.getPerspectiveTransform(destination, source)

  def calibrate(self, chessboard_img_files, chessboard_size):
    """
    影像校正
    """
    chess_rows, chess_cols = chessboard_size
    corners = np.zeros((chess_cols*chess_rows, 3), np.float32)
    corners[:,:2] = np.mgrid[0:chess_cols, 0:chess_rows].T.reshape(-1,2)
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    for frame in chessboard_img_files:
      img = cv2.imread(frame)
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

      ret, chess_corners = cv2.findChessboardCorners(gray, (chess_rows, chess_cols), None)
      if ret == True:
        objpoints.append(corners)
        imgpoints.append(chess_corners)

    sucess, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, self.img_size, None, None)
    if not sucess:
      raise Exception('Camera calibration unsucessful')
    return camera_matrix, dist_coeffs

  def undistort(self, image):
    return cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)

  def warp_to_overhead(self, image):
    return cv2.warpPerspective(image, self.overhead_transform, dsize=(self.img_width, self.img_height))

  def warp_to_dashboard(self, image):
    return cv2.warpPerspective(image, self.inverse_overhead_transform, dsize=(self.img_width, self.img_height))

class LaneFilder:
  def __init__(self, camera, window_shape=(80, 61)):
    # self.gif_frame = FrameGIF(60)
    self.camera = camera
    self.width, self.height = camera.img_width, camera.img_height  ## 影片大小與棋盤影像大小一致，因此可以這樣使用
    self.image_windows_left = deque(maxlen=5)
    self.image_windows_right = deque(maxlen=5)
    self.image_windows_left_filter = deque(maxlen=5)
    self.image_windows_right_filter = deque(maxlen=5)
    self.windows_left = []
    self.windows_right = []
    for level in range(self.height // window_shape[0]):
      left_x_init = self.width / 4   ## 各半的中間作為起始點
      right_x_init = self.width / 4 * 3
      self.windows_left.append(Window(level, camera.img_size, window_shape, left_x_init))
      self.windows_right.append(Window(level, camera.img_size, window_shape, right_x_init))

  def score_pixel(self, img):
    settings = [{'name': 'lab_b', 'cspace': 'LAB', 'channel': 2, 'clipLimit': 2.0, 'threshold': 150},
                {'name': 'value', 'cspace': 'HSV', 'channel': 2, 'clipLimit': 6.0, 'threshold': 220},
                {'name': 'lightness', 'cspace': 'HLS', 'channel': 1, 'clipLimit': 2.0, 'threshold': 210}]

    scores = np.zeros((self.height, self.width))

    for params in settings:
      # color_t = getattr(cv2, 'COLOR_BGR2{}'.format(params['cspace']))
      color_t = eval('cv2.COLOR_BGR2{}'.format(params['cspace']))
      gray = cv2.cvtColor(img, color_t)[:, :, params['channel']]
      # Normalize regions of the image using CLAHE
      clahe = cv2.createCLAHE(params['clipLimit'], tileGridSize=(8, 8))
      norm_img = clahe.apply(gray)

      ret, binary = cv2.threshold(norm_img, params['threshold'], 1, cv2.THRESH_BINARY)
      scores += binary

    return cv2.normalize(scores, None, 0, 255, cv2.NORM_MINMAX)

  def fit_lanes(self, image_windows_left, image_windows_right, mode="mutil"):
    """

    """
    assert mode == "mutil" or mode == "single", "parameter \"mode\" must be mutil/single"
    if mode == "mutil":
      left_x = [win.x_position for windows_left in image_windows_left for win in windows_left]
      left_y = [win.y_position for windows_left in image_windows_left for win in windows_left]
      right_x = [win.x_position for windows_right in image_windows_right for win in windows_right]
      right_y = [win.y_position for windows_right in image_windows_right for win in windows_right]
    if mode == "single":
      left_x = [win.x_position for win in image_windows_left]
      left_y = [win.y_position for win in image_windows_left]
      right_x = [win.x_position for win in image_windows_right]
      right_y = [win.y_position for win in image_windows_right]

    param_left = np.polyfit(left_y, left_x, 2)
    param_right = np.polyfit(right_y, right_x, 2)
    
    return param_left, param_right


  def find_lanes(self, img):
    self.undistort_img = self.camera.undistort(img)
    self.overhead_img = self.camera.warp_to_overhead(self.undistort_img)
    
    score_img = self.score_pixel(self.overhead_img)
    # debug_show(score_img)
    window_slide(self.windows_left, self.windows_right, score_img)
    self.image_windows_left.append(self.windows_left)
    self.image_windows_right.append(self.windows_right)
    # show_windows(score_img, self.windows_left)
    # show_windows(score_img, self.windows_right)
    ## 去除有問題的 window
    left_windows, windows_left_level = filter_with_list(self.windows_left)
    right_windows, windows_right_level = filter_with_list(self.windows_right)
    self.image_windows_left_filter.append(left_windows)
    self.image_windows_right_filter.append(right_windows)
    print("left windows {}, right windows {}".format(len(left_windows), len(right_windows)))


    param_left, param_right = self.fit_lanes(self.image_windows_left_filter, self.image_windows_right_filter, mode="mutil")
    
    short_line_max_ndx = min(windows_left_level[-1], windows_right_level[-1])  ## 找最上面的 window

    self.y_fit = np.array(range(self.windows_left[short_line_max_ndx].y_begin, self.windows_left[0].y_end))
    self.x_fit_left = param_left[0] * self.y_fit ** 2 + param_left[1] * self.y_fit + param_left[2]
    self.x_fit_right = param_right[0] * self.y_fit ** 2 + param_right[1] * self.y_fit + param_right[2]
    
    self.viz_window(score_img, left_windows, right_windows)

  def viz_window(self, score_img, windows_left, windows_right):
    out_img = np.stack((score_img,)*3, axis=-1).astype('uint8')
    lw_color = window_mask(windows_left, color=(0, 255, 0))
    rw_color = window_mask(windows_right, color=(0, 255, 0))
    combined = lw_color + rw_color
    out_img = cv2.addWeighted(out_img, 1, combined, 0.5, 0)

    for i in range(len(self.x_fit_left)-1):
      cv2.line(out_img, (int(self.x_fit_left[i]), int(self.y_fit[i])), (int(self.x_fit_left[i+1]), int(self.y_fit[i+1])), (0, 255, 0), 5)
      cv2.line(out_img, (int(self.x_fit_right[i]), int(self.y_fit[i])), (int(self.x_fit_right[i+1]), int(self.y_fit[i+1])), (0, 255, 0), 5)
    # debug_show(out_img)

  def viz_lane(self):
    lane_poly_overhead = np.zeros_like(self.undistort_img).astype(np.uint8)

    pts_left = np.array([np.transpose(np.vstack([np.array(self.x_fit_left), self.y_fit]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([self.x_fit_right, self.y_fit])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(lane_poly_overhead, np.int_([pts]), (0, 255, 0))
    lane_poly_dash = self.camera.warp_to_dashboard(lane_poly_overhead)

    out = cv2.addWeighted(self.undistort_img, 1, lane_poly_dash, 0.3, 0)
    outt = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    # debug_show(out)
    return out

if __name__ in "__main__":
  lane_shape = [(584, 458), (701, 458), (295, 665), (1022, 665)]
  # lane_shape = [(500, 458), (701, 458), (295, 550), (1022, 550)]
  cailbration_files = glob.glob('../camera_cal/*.jpg')
  camera = DashboardCamear(cailbration_files, chessboard_size=(9, 6), lane_shape=lane_shape)
  cap = cv2.VideoCapture("../test_videos/project_video.mp4")
  lane_fider = LaneFilder(camera)
  
  while True:
    if not cap.isOpened():
      break
    ret, frame = cap.read()

    try:
      lane_fider.find_lanes(frame)
      out = lane_fider.viz_lane()
      cv2.imshow('frame', out)
    except:
      print("Something wrong")
      # cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()