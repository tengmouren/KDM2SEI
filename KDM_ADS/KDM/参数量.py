from models.se_multi_cale import Encoder_and_projection
import yaml
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

config = yaml.load(open("config/config.yaml", "r"), Loader=yaml.FullLoader)
params = config["trainer"]
# 创建模型
online_network = Encoder_and_projection(**config["network"])


# 计算模型参数数量
num_params = count_parameters(online_network)
print("模型参数数量:", num_params)
# 1258968232