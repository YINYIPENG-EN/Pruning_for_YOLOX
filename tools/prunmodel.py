import torch
import torch.nn as nn
import torch_pruning as tp
from nets.yolo import YOLOX
from loguru import  logger

"""
剪枝的时候根据模型结构去剪，不要盲目的猜
剪枝完需要进行一个微调训练
"""
def save_whole_model(weights_path, num_classes):
    model = YOLOX(num_classes, 's')  # 这里需要根据自己的类数量修改
    model_dict = model.state_dict()
    pretrained_dict = torch.load(weights_path)
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() == pretrained_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()
    torch.save(model, '../model_data/whole_model.pth')
    print("保存完成\n")

@logger.catch
def Conv_pruning(whole_model_weights):
    logger.add('../logs/Conv_pruning.log', rotation='1 MB')
    model = torch.load(whole_model_weights)  # 模型的加载
    model_dict = model.state_dict()  # 获取模型的字典

    # -------------------特定卷积的剪枝--------------------
    #                  比如要剪枝以下卷积
    #              'backbone.conv1.conv.weight'
    # --------------------------------------------------
    for k, v in model_dict.items():
        if k == 'backbone.backbone.dark2.0.conv.weight':  # 对主干网络中的backone.conv1.conv权重进行剪枝
            # 1. setup strategy (L1 Norm)
            strategy = tp.strategy.L1Strategy()  # or tp.strategy.RandomStrategy()

            # 2. build layer dependency
            DG = tp.DependencyGraph()
            DG.build_dependency(model, example_inputs=torch.randn(1, 3, 640, 640))
            num_params_before_pruning = tp.utils.count_params(model)
            # 3. get a pruning plan from the dependency graph.
            pruning_idxs = strategy(v, amount=0.4)  # or manually selected pruning_idxs=[2, 6, 9, ...]
            pruning_plan = DG.get_pruning_plan((model.backbone.backbone.dark2)[0].conv, tp.prune_conv, idxs=pruning_idxs)
            logger.info(pruning_plan)

            # 4. execute this plan (prune the model)
            pruning_plan.exec()
            # 获得剪枝以后的参数量
            num_params_after_pruning = tp.utils.count_params(model)
            # 输出一下剪枝前后的参数量
            logger.info("  Params: %s => %s\n" % (num_params_before_pruning, num_params_after_pruning))
    torch.save(model, '../model_data/Conv_pruning.pth')
    logger.info("剪枝完成\n")


@logger.catch
def layer_pruning(whole_model_weights):
    logger.add('../logs/layer_pruning.log', rotation='1 MB')
    model = torch.load(whole_model_weights)  # 模型的加载
    x = torch.randn(1, 3, 640, 640)
    # -----------------对整个模型的剪枝--------------------
    strategy = tp.strategy.L1Strategy()
    DG = tp.DependencyGraph()
    DG = DG.build_dependency(model, example_inputs=x)

    num_params_before_pruning = tp.utils.count_params(model)

    # 可以对照yolox结构进行剪枝
    included_layers = list((model.backbone.backbone.dark2.modules()))  # 对主干进行剪枝
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m in included_layers:
            pruning_plan = DG.get_pruning_plan(m, tp.prune_conv, idxs=strategy(m.weight, amount=0.4))
            logger.info(pruning_plan)
            # 执行剪枝
            pruning_plan.exec()
    # 获得剪枝以后的参数量
    num_params_after_pruning = tp.utils.count_params(model)
    # 输出一下剪枝前后的参数量
    logger.info("  Params: %s => %s\n" % (num_params_before_pruning, num_params_after_pruning))
    # 剪枝完以后模型的保存(不要用torch.save(model.state_dict(),...))
    torch.save(model, '../model_data/layer_pruning.pth')
    logger.info("剪枝完成\n")

# model = torch.load('../model_data/whole_model.pth')
# model_dict = model.state_dict()
# for k,v  in model_dict.items():
#     print(k)
# print(model)
#layer = nn.ModuleList(m for m in model.backbone.backbone.dark2.modules())

layer_pruning('../model_data/whole_model.pth')
#Conv_pruning('../model_data/whole_model.pth')