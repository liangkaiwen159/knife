from mmdet.apis import DetInferencer
import os
from tqdm import tqdm
import time

save_dir = 'output_torch'

class MyDetInferencer(DetInferencer):
    def __init__(self,show_progress=False, *args, **kwargs):
        super().__init__(*args, show_progress=show_progress, device='cuda',**kwargs)
        self.imgs = []
        self.total_time = 0

    def __call__(self, img_dir, out_dir, pred_score_thr):
        self.imgs = sorted([os.path.join(img_dir, img) for img in os.listdir(img_dir)])[:100]
        self.total_num = len(self.imgs)
        for img in tqdm(self.imgs):
            t1 = time.time()
            super().__call__(img, out_dir=out_dir, pred_score_thr=pred_score_thr)
            t2 = time.time()
            self.total_time += t2 - t1
        print(f'average torch time: {round(self.total_time / self.total_num, 3)}s')

def infer_torch():
    inferencer = MyDetInferencer(model='../icann_dino_detr/configs/dino_rot/dino-4scale_dota.py', weights='../icann_dino_detr/work_dirs/dino-4scale_dota/best_dota_mAP_0.73.pth')

    image_dir = '/home/liangkaiwen/datasets/split_ss_dota/val/images/'

    global save_dir

    if os.path.exists(save_dir):
        os.system(f'rm -rf {save_dir}')
        os.mkdir(save_dir)
    else:
        os.mkdir(save_dir)

    inferencer(image_dir, out_dir=save_dir, pred_score_thr = 0.5)

if __name__ == '__main__':
    infer_torch()