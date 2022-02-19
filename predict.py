import os
import os.path as osp
from tqdm import tqdm
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchnet import meter

from model.resnet_deconv import get_deconv_net
from model.hourglass import PoseNet
from model.loss import My_SmoothL1Loss
from dataloader.custom_loader import CustomLoader
from dataloader.nyu_loader import NYU
from util.feature_tool import FeatureModule
from util.eval_tool import EvalUtil
from util.vis_tool import VisualUtil
from config import opt

class Trainer(object):

    def __init__(self, config):
        torch.cuda.set_device(config.gpu_id)
        cudnn.benchmark = True

        self.config = config
        self.data_dir = osp.join(self.config.data_dir, self.config.dataset)

        # output dirs for model, log and result figure saving
        self.work_dir = osp.join(self.config.output_dir, self.config.dataset, 'checkpoint')
        self.result_dir = osp.join(self.config.output_dir, self.config.dataset, 'results' )
        if not osp.exists(self.work_dir):
            os.makedirs(self.work_dir)
        if not osp.exists(self.result_dir):
            os.makedirs(self.result_dir)

        self.stacks = int(self.config.net.split('_')[1])
        self.net = PoseNet(self.config.net, self.config.jt_num)
        self.net = self.net.cuda()

        if self.config.load_model :
            print('loading model from %s' % self.config.load_model)
            pth = torch.load(self.config.load_model)
            self.net.load_state_dict(pth['model'])
            print(pth['best_records'])
        self.net = self.net.cuda()

        if self.config.dataset == 'nyu':
          self.testData = NYU(self.data_dir, 'test', img_size=self.config.img_size, cube=self.config.cube)
        else:
          self.testData = CustomLoader(self.data_dir, 'test', img_size=self.config.img_size)

        self.criterion = My_SmoothL1Loss().cuda()

        self.FM = FeatureModule()

    @torch.no_grad()
    def test(self, epoch):
        self.testLoader = DataLoader(self.testData, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers)
        self.net.eval()

        eval_tool = EvalUtil(self.testData.img_size, self.testData.paras, self.testData.flip, self.testData.jt_num)
        joint_predictions = []
        for ii, (img, center_uvd, M, cube) in tqdm(enumerate(self.testLoader)):

            input = img.cuda()

            for stage_idx in range(self.stacks):
                offset_pred = self.net(input)[stage_idx]
                jt_uvd_pred = self.FM.offset2joint_softmax(offset_pred, input, self.config.kernel_size)

            M = M.detach().numpy()
            cube = cube.detach().numpy()
            jt_uvd_pred = jt_uvd_pred.detach().cpu().numpy()
            for i in range(jt_uvd_pred.shape[0]):
              pred = prepare_joint_prediction(jt_uvd_pred[i], center_uvd[i], M[i], cube[i], self.config.img_size)
              joint_predictions.append(pred)

        print('FINISHED!')

        if epoch == -1:
            txt_file = osp.join(self.work_dir, 'test_%.3f.txt' % 999)
            jt_uvd = np.array(joint_predictions, dtype = np.float32)
            if not txt_file == None:
                np.savetxt(txt_file, jt_uvd.reshape([jt_uvd.shape[0], self.config.jt_num * 3]), fmt='%.3f')

def prepare_joint_prediction(jt_uvd_pred, center_uvd, M, cube, img_size):
  jt_uvd_pred = np.squeeze(jt_uvd_pred).astype(np.float32)
  center_uvd = np.squeeze(center_uvd).astype(np.float32)
  M = np.squeeze(M).astype(np.float32)
  cube = np.squeeze(cube).astype(np.float32)

  M_inv = np.linalg.inv(M)

  jt_uvd_pred[:, :2] = (jt_uvd_pred[:, :2] + 1) * img_size / 2.
  jt_uvd_pred[:, 2] = jt_uvd_pred[:, 2] * cube[2] / 2. + center_uvd[2]
  jt_uvd_trans = np.hstack([jt_uvd_pred[:, :2], np.ones((jt_uvd_pred.shape[0], 1))])
  jt_uvd_pred[:, :2] = np.dot(M_inv, jt_uvd_trans.T).T[:, :2]

  return jt_uvd_pred

if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    trainer = Trainer(opt)
    trainer.test(-1)