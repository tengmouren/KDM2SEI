import os
import statistics
import sys

from AutomaticWeightedLoss import AutomaticWeightedLoss
from models.encoder_and_projection import Encoder_and_projection

print(sys.path)
sys.path.insert(0, "models")
import torch
import yaml

# from models.encoder_and_projection import Encoder_and_projection
from models.classifier import Classifier
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
from get_dataset import (
    FineTuneDataset_prepared,
    finetune_dataset_add_feature,
    finetune_dataset_add_feature_SSML10CLASS,
)
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import os
import random
import time


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # All GPU
    os.environ["PYTHONHASHSEED"] = str(seed)  # 禁止hash随机化
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(
    online_network,
    classifier,
    classifier2,
    loss_nll,
    train_dataloader,
    optim_online_network,
    optimizer_classifier,
    scheduler_online_network,
    scheduler_classifier,
    epoch,
    device,
    writer,
    awl1,
):
    online_network.train()  # 启动训练, 允许更新模型参数
    classifier.train()
    correct = 0
    nll_loss = 0
    for data, target in train_dataloader:
        target = target.long()
        if torch.cuda.is_available():
            data = data.to(device)
            target = target.to(device)

        optim_online_network.zero_grad()
        optimizer_classifier.zero_grad()
        # IQ特征
        embedding = online_network(data)[0]
        output = F.log_softmax(classifier(embedding), dim=1)
        nll_loss_batch = loss_nll(output, target)
        # 人工特征
        embedding_feature = online_network(data)[1]
        outpute_feature = F.log_softmax(classifier2(embedding_feature), dim=1)
        nll_loss_batch_feature = loss_nll(outpute_feature, target)
        # 结合器
        result_loss_batch, weight_pa = awl1(
            nll_loss_batch,
            nll_loss_batch_feature,
        )

        result_loss_batch.backward()
        optim_online_network.step()
        optimizer_classifier.step()
        # scheduler_online_network.step()
        # scheduler_classifier.step()

        nll_loss += nll_loss_batch.item()
        nll_loss_batch_feature += nll_loss_batch_feature.item()

        output = weight_pa[0] * output + weight_pa[1] * outpute_feature
        pred = output.argmax(dim=1, keepdim=True)
        correct += (
            pred.eq(target.view_as(pred)).sum().item()
        )  # 求pred和target中对应位置元素相等的个数

    nll_loss /= len(train_dataloader)
    nll_loss_batch_feature /= len(train_dataloader)

    print("weight_pa:", awl1.params[0], awl1.params[1], weight_pa[0], weight_pa[1])
    print(
        "Train Epoch: {} \tClass_Loss: {:.6f},feature_Loss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n".format(
            epoch,
            nll_loss,
            nll_loss_batch_feature,
            correct,
            len(train_dataloader.dataset),
            100.0 * correct / len(train_dataloader.dataset),
        )
    )


def evaluate(
    online_network,
    classifier,
    classifier2,
    loss_nll,
    val_dataloader,
    epoch,
    device,
    writer,
    awl1,
):
    online_network.eval()
    classifier.eval()
    classifier2.eval()
    awl1.eval()
    test_loss = 0
    feature_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)

            # IQ特征
            embedding = online_network(data)[0]
            output = F.log_softmax(classifier(embedding), dim=1)

            # 人工特征
            embedding_feature = online_network(data)[1]
            outpute_feature = F.log_softmax(classifier2(embedding_feature), dim=1)

            test_loss += loss_nll(output, target).item()
            feature_loss += loss_nll(outpute_feature, target).item()
            output = awl1.params[0] * output + awl1.params[1] * outpute_feature
            pred = output.argmax(dim=1, keepdim=True)
            correct += (
                pred.eq(target.view_as(pred)).sum().item()
            )  # 求pred和target中对应位置元素相等的个数

    test_loss /= len(val_dataloader)
    feature_loss /= len(val_dataloader)
    fmt = "\nValidation set: test_loss: {:.4f}, feature_loss: {:.4f},  Accuracy: {}/{} ({:0f}%)\n"
    print(
        fmt.format(
            test_loss,
            feature_loss,
            correct,
            len(val_dataloader.dataset),
            100.0 * correct / len(val_dataloader.dataset),
        )
    )
    # writer.add_scalar(
    #     "Accuracy/val", 100.0 * correct / len(val_dataloader.dataset), epoch
    # )
    # writer.add_scalar("Loss/val", test_loss, epoch)
    return test_loss, feature_loss, 100.0 * correct / len(val_dataloader.dataset)


def test(online_network, classifier, classifier2, test_dataloader, device, awl1):
    online_network.eval()
    classifier.eval()
    classifier2.eval()
    test_loss = 0
    feature_loss = 0
    correct = 0
    loss = nn.NLLLoss()
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
                loss = loss.to(device)
            # IQ特征
            embedding = online_network(data)[0]
            output = F.log_softmax(classifier(embedding), dim=1)

            # 人工特征
            embedding_feature = online_network(data)[1]
            outpute_feature = F.log_softmax(classifier2(embedding_feature), dim=1)

            test_loss += loss(output, target).item()
            feature_loss += loss(outpute_feature, target).item()
            output = (
                1 - awl1.params_best
            ) * output + awl1.params_best * outpute_feature
            pred = output.argmax(dim=1, keepdim=True)
            correct += (
                pred.eq(target.view_as(pred)).sum().item()
            )  # 求pred和target中对应位置元素相等的个数

    test_loss /= len(test_dataloader)
    feature_loss /= len(test_dataloader)
    fmt = "\nTest set: test_loss: {:.4f}, feature_loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n"
    print(
        fmt.format(
            test_loss,
            feature_loss,
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )
    return 100.0 * correct / len(test_dataloader.dataset)


def train_and_test(
    online_network,
    classifier,
    classifier2,
    loss_nll,
    train_dataloader,
    val_dataloader,
    optim_online_network,
    optim_classifier,
    scheduler_online_network,
    scheduler_classifier,
    epochs,
    save_path_online_network,
    save_path_classifier,
    save_path_classifier2,
    device,
    writer,
    awl1,
):
    current_min_test_loss = 0
    for epoch in range(1, epochs + 1):
        train(
            online_network,
            classifier,
            classifier2,
            loss_nll,
            train_dataloader,
            optim_online_network,
            optim_classifier,
            scheduler_online_network,
            scheduler_classifier,
            epoch,
            device,
            writer,
            awl1=awl1,
        )
        test_loss, feature_loss, correct = evaluate(
            online_network,
            classifier,
            classifier2,
            loss_nll,
            val_dataloader,
            epoch,
            device,
            writer,
            awl1=awl1,
        )
        validation_loss = test_loss + feature_loss
        # if validation_loss < current_min_test_loss:
        #     print(
        #         "The validation loss is improved from {} to {}, new model weight is saved.".format(
        #             current_min_test_loss, validation_loss
        #         )
        #     )
        #     awl1.params_best = float(test_loss / validation_loss)
        #     current_min_test_loss = validation_loss
        #     torch.save(online_network, save_path_online_network)
        #     torch.save(classifier, save_path_classifier)
        #     torch.save(classifier2, save_path_classifier2)
        if correct > current_min_test_loss:
            print(
                "The validation loss is improved from {} to {}, new model weight is saved.".format(
                    current_min_test_loss, validation_loss
                )
            )
            awl1.params_best = float(test_loss / validation_loss)
            current_min_test_loss = correct
            torch.save(online_network, save_path_online_network)
            torch.save(classifier, save_path_classifier)
            torch.save(classifier2, save_path_classifier2)
        else:
            print("The validation loss is not improved.")
        print("------------------------------------------------")


def run(
    checkpoints_folder,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    epochs,
    save_path_online_network,
    save_path_classifier,
    save_path_classifier2,
    device,
    writer,
    config,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training with: {device}")

    #    "runs",seed300_pretrain_class_in_{config['trainer']['class_start']}-{config['trainer']['class_end']}","checkpoints",
    online_network = torch.load(os.path.join(checkpoints_folder, "model_val_best.pth"))
    # print(online_network)
    classifier = Classifier(input_dim=1024, mid_dim=512)
    classifier2 = Classifier(input_dim=1044, mid_dim=10)
    awl1 = AutomaticWeightedLoss(2)

    loss_nll = nn.NLLLoss()
    if torch.cuda.is_available():
        online_network = online_network.to(device)
        classifier = classifier.to(device)
        loss_nll = loss_nll.to(device)
        classifier2 = classifier2.to(device)

    optim_online_network = torch.optim.Adam(
        [
            {"params": online_network.parameters(), "lr": config["finetune"]["lr"]},
            {"params": awl1.parameters(), "weight_decay": 0},
            {"params": classifier2.parameters(), "lr": config["finetune"]["lr"]},
        ]
    )
    optim_classifier = torch.optim.Adam(
        classifier.parameters(), lr=config["finetune"]["lr"]
    )

    scheduler_online_network = CosineAnnealingLR(optim_online_network, T_max=20)
    scheduler_classifier = CosineAnnealingLR(optim_classifier, T_max=20)
    # 训练 验证 保存模型
    train_and_test(
        online_network,
        classifier,
        classifier2,
        loss_nll,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optim_online_network=optim_online_network,
        optim_classifier=optim_classifier,
        scheduler_online_network=scheduler_online_network,
        scheduler_classifier=scheduler_classifier,
        epochs=epochs,
        save_path_online_network=save_path_online_network,
        save_path_classifier=save_path_classifier,
        save_path_classifier2=save_path_classifier2,
        device=device,
        writer=writer,
        awl1=awl1,
    )

    # 测试
    print("Test_result:")
    online_network = torch.load(save_path_online_network)
    classifier = torch.load(save_path_classifier)
    classifier2 = torch.load(save_path_classifier2)
    test_acc = test(
        online_network, classifier, classifier2, test_dataloader, device, awl1=awl1
    )
    return test_acc


def main():
    config = yaml.load(open("config/config.yaml", "r"), Loader=yaml.FullLoader)
    config_ft = config["finetune"]

    device = torch.device("cuda:0")
    test_acc_50 = []
    test_acc_all = []
    start = time.time()
    for i in range(config["iteration"]):
        print(f"iteration: {i}--------------------------------------------------------")
        set_seed(i)
        writer = SummaryWriter(
            f"./log_finetune/PT_{config['trainer']['class_start']}-{config['trainer']['class_end']}_FT_{config_ft['class_start']}-{config_ft['class_end']}_{config_ft['k_shot']}shot"
        )
        save_path_classifier = f"./model_weight/classifier_PT_{config['trainer']['class_start']}-{config['trainer']['class_end']}_FT_{config_ft['class_start']}-{config_ft['class_end']}_{config_ft['k_shot']}shot_{i}.pth"
        save_path_classifier2 = f"./model_weight/classifier2_PT_{config['trainer']['class_start']}-{config['trainer']['class_end']}_FT_{config_ft['class_start']}-{config_ft['class_end']}_{config_ft['k_shot']}shot_{i}.pth"
        save_path_online_network = f"./model_weight/online_network_PT_{config['trainer']['class_start']}-{config['trainer']['class_end']}_FT_{config_ft['class_start']}-{config_ft['class_end']}_{config_ft['k_shot']}shot_{i}.pth"

        # （10*shot,2,4828）
        X_train, X_test, X_val, Y_train, Y_test, Y_val = finetune_dataset_add_feature()
        # X_train, X_test, X_val, Y_train, Y_test, Y_val = FineTuneDataset_prepared()
        # X_train, X_test, X_val, Y_train, Y_test, Y_val = (
        #     finetune_dataset_add_feature_SSML10CLASS(add_feature=False)
        # )

        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
        train_dataloader = DataLoader(
            train_dataset, batch_size=config_ft["batch_size"], shuffle=True
        )

        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val))
        val_dataloader = DataLoader(
            val_dataset, batch_size=config_ft["batch_size"], shuffle=True
        )

        test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
        test_dataloader = DataLoader(
            test_dataset, batch_size=config_ft["test_batch_size"], shuffle=True
        )

        # 训练+验证：保存模型  测试
        checkpoints_folder = os.path.join(
            "runs",
            f"seed300_pretrain_class_in_{config['trainer']['class_start']}-{config['trainer']['class_end']}",
            "checkpoints",
        )

        # model_val_best.pth
        test_acc = run(
            checkpoints_folder,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            epochs=config_ft["epochs"],
            save_path_online_network=save_path_online_network,
            save_path_classifier=save_path_classifier,
            save_path_classifier2=save_path_classifier2,
            device=device,
            writer=writer,
            config=config,
        )
        test_acc_all.append(test_acc)
        writer.close()

        # 每次保存
        average_test_acc = statistics.mean(test_acc_all)
        test_acc_50.append(average_test_acc)
        df = pd.DataFrame(test_acc_50)
        df.to_excel(f"test_result/{i}_{config_ft['k_shot']}shot.xlsx")

    end = time.time()
    average_test_acc = statistics.mean(test_acc_all)
    test_acc_all.append(average_test_acc)
    print("average grade:", average_test_acc)
    print("eppch grade:", test_acc_all)
    print("all time:", end - start)
    df = pd.DataFrame(test_acc_all)
    df.to_excel(
        f"test_result/PT_{config['trainer']['class_start']}-{config['trainer']['class_end']}_FT_{config_ft['class_start']}-{config_ft['class_end']}_{config_ft['k_shot']}shot.xlsx"
    )


if __name__ == "__main__":
    main()
