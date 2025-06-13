import sys
import argparse
import os


sys.path.append('/root/yolov11/') # Path 

from ultralytics import YOLO

def main(opt):
    yaml = opt.cfg
    model = YOLO(yaml) 

    model.info()

    model = YOLO('/root/runs/detect/train/weights/best.pt')  #这里我默认使用的是训练好的权重,可以自定义修改

    model.predict('yolov11/datasets/data/test/images/7_jpg.rf.22d61ce3fa80f03f3ec7e156ddc0907d.jpg', save=True, imgsz=640, conf=0.5)  #这里我默认在yolov11/datasets/data/test/images中选取的一张


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default= r'yolov8n.pt', help='initial weights path')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)