import cv2 
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter, logpdf

def debug_show(inf, mode='image'):
  if mode == 'image':
    plt.imshow(inf)
    plt.show()
  if mode == 'curve':
    plt.plot(inf)
    plt.show()

class Window:
  def __init__(self, level, img_shape, window_shape, x_init):
    self.img_h, self.img_w = img_shape
    self.window_h, self.window_w = window_shape

    ## window 的高 
    self.y_begin = self.img_h - (level + 1) * self.window_h
    self.y_end = self.y_begin + self.window_h

    ## window x 與 y 座標
    self.x_position = x_init
    self.y_position = self.y_begin + self.window_h / 2.0


    self.detected = False  ## 是否偵測到線
    self.drop = False  ## 是否要丟去這個視窗
    self.filter = WindowFilter(pos_init=x_init)

  # def calculate_x_begin(self, column_scores, x_offset):
  #   """
  #   計算 window x 座標起始位置

  #   ✘：這會有個問題，當該點旁邊一直有數值去延伸的時候就會擴大
  #   """
  #   margin = 1
  #   center_x = np.argmax(column_scores)
  #   while True:
  #     self.x_begin = center_x - margin
  #     self.x_end = center_x + margin
  #     margin += 1
  #     if column_scores.shape[0] <= self.x_end:
  #       break
  #     if column_scores[self.x_begin] == 0 or column_scores[self.x_end] == 0:
  #       self.x_begin = self.x_begin + x_offset
  #       self.x_end = self.x_end + x_offset
  #       break

  def x_begin(self):
    x = getattr(self, 'x_position')
    return int(max(0, x - self.window_w // 2))
  
  def x_end(self):
    x = getattr(self, 'x_position')
    return int(min(x + self.window_w // 2, self.img_w))

  def update(self, pixel_scores, x_search_range):
    """
    window 位置更新
    """
    x_search_range = (max(0, int(x_search_range[0])), min(int(x_search_range[1]), self.img_w))
    x_offset = x_search_range[0]
    search_region = pixel_scores[self.y_begin:self.y_end, x_offset:x_search_range[1]]
    column_scores = gaussian_filter(np.sum(search_region, axis=0), sigma=5)

    if max(column_scores) > 0:  ## 判斷該區域是否有影像
      self.detected = True
      x_measure = np.argmax(column_scores) + x_offset   ## window 量測後的 x 位置

      ## 估測 window x 位置
      self.filter.update(x_measure)   ## kalman filter update
      self.x_position = self.filter.get_position()
      self.drop = False
    
    else:
      self.detected = False
      self.drop = True


class WindowFilter:
  def __init__(self, pos_init=0.0, means_variance=50, process_variance=1, uncertainty_init=500):
    """
    state variable = [position, velocity]
    """
    self.kf = KalmanFilter(dim_x=2, dim_z=1)
    
    self.kf.F = np.array([[1, 1], [0, 0.5]])
    self.kf.H = np.array([[1, 0]])
    self.kf.x = np.array([pos_init, 0])
    self.kf.P = np.eye(self.kf.dim_x) * uncertainty_init

    self.kf.R = np.array([[means_variance]]) 

    self.kf.Q = Q_discrete_white_noise(dim=2, dt=1, var=process_variance)

  def update(self, pos):
    self.kf.predict()
    self.kf.update(pos)

  def get_position(self):
    return self.kf.x[0]

def window_slide(windows_left, windows_right, score_img, margin=200):
  """
  window 搜尋
  """
  search_center = [start_sliding_search(windows_left, score_img, 'left'),
                   start_sliding_search(windows_right, score_img, 'right')]

  for i in range(len(windows_left)):
    x_search_ranges = [None, None]
    for j in range(len(search_center)):
      x_search_ranges[j] = [search_center[j] - margin, search_center[j] + margin]
    
    if x_search_ranges[0][1] > x_search_ranges[1][0]:
      average = (x_search_ranges[0][1] + x_search_ranges[1][0]) // 2
      x_search_ranges[0][1] = average
      x_search_ranges[1][0] = average
      # print(average)
    
    for k, window in enumerate([windows_left[i], windows_right[i]]):
      window.update(score_img, x_search_ranges[k])
      if not window.drop:
        search_center[k] = window.x_position

def start_sliding_search(windows_list, score_img, mode):
  """
  從影像最下方 window 開始搜尋
  """
  img_h, img_w = score_img.shape[:2]
  if mode == 'left':
   windows_list[0].update(score_img, (0, img_w//2))
  elif mode == 'right':
    windows_list[0].update(score_img, (img_w//2, img_w))

  search_center = windows_list[0].x_position
  return search_center

def filter_with_list(windows_list):
  """
  將 drop window 丟棄
  """
  windows_filter = []
  levels = []
  for i, window in enumerate(windows_list):
    if window.drop:
      continue
    windows_filter.append(window)
    levels.append(i)
  return windows_filter, levels

def window_mask(windows_list, color):
  """
  window 遮罩上顏色
  """
  mask = np.zeros((windows_list[0].img_h, windows_list[0].img_w, 3))
  for window in windows_list:
    mask[window.y_begin:window.y_end, window.x_begin():window.x_end()] = color
  return mask.astype('uint8')