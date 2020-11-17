from .base_model import BaseModel
from .ganimation import GANimationModel
from .stargan import StarGANModel



def create_model(opt):
    # specify model name here
    if opt.model == "ganimation":   # 默认使用GANimationModel()
        instance = GANimationModel()  # 声明一个GANimationModel()类
    elif opt.model == "stargan":
        instance = StarGANModel()
    else:
        instance = BaseModel()
    instance.initialize(opt)     # 调用GANimationModel()类的initialize函数
    instance.setup()
    return instance

