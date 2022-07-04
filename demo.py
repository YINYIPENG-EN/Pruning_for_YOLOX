import argparse
from loguru import logger
from tools.summary import get_model_info

@logger.catch
def main(opt):
    logger.add("logs/demo.log", rotation='1 MB')
    logger.info("Args: {}".format(opt))
    logger.info("Model Summary: {}".format(get_model_info(opt)))
    if opt.fuse:
        """
        FPS increases by 3 frames /s in fuse mode
        """
        logger.info('\tFusing model....')

    if opt.predict:
        from tools.predict import Predcit
        Predcit(opt)

if __name__ == "__main__":
    parse = argparse.ArgumentParser("YOLOX demo")
    parse.add_argument('--predict', action='store_true', default=False, help='model predict')
    parse.add_argument('--pruned', action='store_true', default=False, help='model pruned predict')
    parse.add_argument('--image', action='store_true', default=False, help='image predict')
    parse.add_argument('--video', action='store_true', default=False, help='video predict')
    parse.add_argument('--video_path', type=str, default='', help='video path')
    parse.add_argument('--camid', type=int, default=0, help='camid')
    parse.add_argument('--fps', action='store_true', default=False, help='FPS test')
    parse.add_argument('--dir_predict', action='store_true', default=False, help='dir_predict predict')
    parse.add_argument('--phi', type=str, default='s', help='s,m,l,x')
    parse.add_argument('--input_shape', type=int, default=640, help='input shape')
    parse.add_argument('--confidence', type=float, default=0.6, help='confidence thres')
    parse.add_argument('--nms_iou', type=float, default=0.5, help='iou thres')
    parse.add_argument('--num_classes', type=int, default=80, help='number of classes')
    parse.add_argument('--fuse', action='store_true', default=False, help='Fusing model')
    opt = parse.parse_args()
    main(opt)

    """
    **参数说明：**下面终端的输入都是可选的
    
    --predict:预测模式
    
    --pruned:开启剪枝预测或训练
    
    --image:图像检测
    
    --video:开始视频检测
    
    --video_path:视频路径
    
    --camid:摄像头id 默认0
    
    --fps:FPS测试
    
    --dir_predict:对一个文件夹下图像进行预测
    
    --phi:可以选择s,m,l,x等
    
    --input_shape:网络输入大小，默认640
    
    --confidence:置信度阈值
    
    --nms_iou:iou阈值
    
    --num_classes:类别数量，默认80
    
    --fuse:是否开启卷积层和BN层融合加速，默认False
    """