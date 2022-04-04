from dataloader.loader import Loader
import numpy as np
import math

class SingleLoader(Loader):

  def __init__(self, root, phase, img_size=128):
      super(SingleLoader, self).__init__(root, phase, img_size, 'custom')
      dataset_size = 1
      self.paras = (588.03, 587.07, 320., 240.)

      self.cube = np.asarray([300, 300, 300])
      self.dsize = np.asarray([img_size, img_size])
      self.img_size = img_size

      self.jt_num = 14
      self.test_cube = np.ones([dataset_size, 3]) * self.cube

      self.in_memory_frame = None
      self.centroid = None

  def __getitem__(self):
    img = self.in_memory_frame.copy()
    cube = self.test_cube

    center_uvd = self.centroid.copy()

    img, M = self.crop(img, center_uvd, cube, self.dsize)
    img = self.normalize(img.max(), img, center_uvd, cube)

    image = img[np.newaxis, :].astype(np.float32)
    transformation_matrix = M.astype(np.float32)
    center_uvd = center_uvd.astype(np.float32)
    cube = cube.astype(np.float32)

    self.in_memory_frame = None
    self.centroid = None

    return image, [], [], center_uvd, transformation_matrix, cube

  def __len__(self):
    return 1

  def set_memory_frame(self, frame, centroid):
    self.in_memory_frame = np.flip(frame, 1)

    mass_y, mass_x = np.where((frame > 0) & (frame < 3000))

    center_x = math.ceil(np.average(mass_x))
    center_y = math.ceil(np.average(mass_y))
    center_z = centroid[2]
    
    centroid = (center_x, center_y, center_z)

    self.centroid = centroid