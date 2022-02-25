import unittest
from dataloader.nyu_loader import NYU

class TestNyuLoader(unittest.TestCase): 

  def test_instance(self):
    root = './data'
    phase = 'test'
    val = False
    img_size = 128
    aug_para=[10, 0.1, 180]
    cube=[300, 300, 300]
    jt_num = 14

    dataset = NYU(
      root=root,
      phase=phase,
      val=val,
      img_size=img_size,
      aug_para=aug_para,
      cube=cube,
      jt_num=jt_num)

    self.assertIsInstance(dataset, NYU)

    print('dataset', dataset)
    print(dataset[0])

if __name__ == '__main__':
  unittest.main()