from dataloader.loader import Loader
import numpy as np
from glob import glob
import cv2

class CustomLoader(Loader):

  def __init__(self, root, phase, img_size=128, ):
      super(CustomLoader, self).__init__(root, phase, img_size, 'custom')

      self.name = 'custom'
      self.root = root
      self.phase = phase
      
      dataset_size = 16 # 8252
      self.paras = (588.03, 587.07, 320., 240.) # wtf

      self.cube = np.asarray([300, 300, 300])
      self.dsize = np.asarray([img_size, img_size])
      self.img_size = img_size

      self.jt_num = 14
      self.aug_para = [10, 0.1, 180]

      self.data = self.make_dataset()
      self.test_cube = np.ones([dataset_size, 3]) * self.cube
      #self.test_cube[4:, :] = self.test_cube[4:, :] * 5 / 6
      self.flip = -1

      print("loading dataset, containing %d images." % len(self.data))

  def __getitem__(self, index):
    img = self.custom_reader(self.data[index][0])
    cube = self.test_cube[index]

    center_uvd = self.data[index][1].copy()

    img, M = self.crop(img, center_uvd, cube, self.dsize)
    img = self.normalize(img.max(), img, center_uvd, cube)

    image = img[np.newaxis, :].astype(np.float32)
    transformation_matrix = M.astype(np.float32)
    center_uvd = center_uvd.astype(np.float32)
    cube = cube.astype(np.float32)

    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(image[0], alpha=0.03), cv2.COLORMAP_JET)

    # window = 'Depth'
    # cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(window, 128, 128)
    # cv2.imshow(window, image[0])
    # cv2.waitKey(0)
    # cv2.destroyWindow(window)

    return image, [], [], center_uvd, transformation_matrix, cube

  def __len__(self):
    return len(self.data)

  def custom_reader(self, frame_path):
    depth = np.loadtxt(frame_path)
    return depth

  def make_dataset(self):
    data_path = '{}'.format(self.root)
    center_path = './data/custom/centroids.txt'

    data = sorted(glob(data_path + '/frame*.txt'))
    centroids = np.loadtxt(center_path)

    item = list(zip(data, centroids))
    return item

  def __str__(self):
    return str(self.__class__) + ': ' + str(self.__dict__)