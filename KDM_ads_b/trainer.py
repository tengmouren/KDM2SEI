import os
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import _create_model_training_folder, Data_augment


class BYOLTrainer:
    def __init__(
        self, online_network, target_network, predictor, optimizer, device, **params
    ):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.device = device
        self.predictor = predictor
        self.max_epochs = params["max_epochs"]
        self.writer = SummaryWriter(
            f"runs/seed300_pretrain_class_in_{params['class_start']}-{params['class_end']}"
        )
        self.m = params["m"]
        self.batch_size = params["batch_size"]
        _create_model_training_folder(
            self.writer, files_to_same=["./config/config.yaml", "main.py", "trainer.py"]
        )

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.online_network.parameters(), self.target_network.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(
            self.online_network.parameters(), self.target_network.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self, train_loader, epoch_counter, loss_min, model_checkpoints_folder):
        Augment_adv = Data_augment(
            rotate=False,
            flip=False,
            rotate_and_flip=True,
            mask=False,
            awgn=False,
            add_noise=False,
            slice=False,
            isAdvAug=True,
            isVAT=False,
        )
        Augment_vat = Data_augment(
            rotate=False,
            flip=False,
            rotate_and_flip=True,
            mask=False,
            awgn=False,
            add_noise=False,
            slice=False,
            isAdvAug=False,
            isVAT=True,
        )
        self.online_network.train()
        self.predictor.train()
        train_loss_epoch = 0

        for batch_view, _ in train_loader:
            batch_view = batch_view.to(self.device)
            batch_view_adv = Augment_adv(batch_view, online_network=self.online_network)
            batch_view_vat = Augment_vat(batch_view, online_network=self.online_network)

            self.optimizer.zero_grad()
            train_loss_batch = self.update(batch_view, batch_view_adv, batch_view_vat)

            train_loss_batch.backward()
            self.optimizer.step()
            train_loss_epoch += train_loss_batch.item()
            self._update_target_network_parameters()  # update the key encoder

        # 保存模型
        train_loss_epoch /= len(train_loader)
        if train_loss_epoch <= loss_min:
            torch.save(
                self.online_network,
                os.path.join(model_checkpoints_folder, "model_train_best.pth"),
            )
            loss_min = train_loss_epoch
        self.writer.add_scalar(
            "train_loss_epoch", train_loss_epoch, global_step=epoch_counter
        )
        print(f"The loss on train dataset: {train_loss_epoch}")
        return loss_min

    def eval(self, val_loader, epoch_counter):
        Augment_adv = Data_augment(
            rotate=False,
            flip=False,
            rotate_and_flip=True,
            mask=False,
            awgn=False,
            add_noise=False,
            slice=False,
            isAdvAug=True,
            isVAT=False,
        )
        Augment_vat = Data_augment(
            rotate=False,
            flip=False,
            rotate_and_flip=True,
            mask=False,
            awgn=False,
            add_noise=False,
            slice=False,
            isAdvAug=False,
            isVAT=True,
        )
        self.online_network.eval()
        self.predictor.eval()
        eval_loss_epoch = 0
        for batch_view, _ in val_loader:
            batch_view = batch_view.to(self.device)
            batch_view_adv = Augment_adv(batch_view, online_network=self.online_network)
            batch_view_vat = Augment_vat(batch_view, online_network=self.online_network)

            with torch.no_grad():
                eval_loss_batch = self.update(
                    batch_view, batch_view_adv, batch_view_vat
                )
                eval_loss_epoch += eval_loss_batch.item()
        eval_loss_epoch /= len(val_loader)
        self.writer.add_scalar(
            "eval_loss_epoch", eval_loss_epoch, global_step=epoch_counter
        )
        print(f"The loss on eval dataset: {eval_loss_epoch}")
        return eval_loss_epoch

    def train_and_val(self, train_dataset, val_dataset):
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, drop_last=False, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, drop_last=False, shuffle=True
        )
        model_checkpoints_folder = os.path.join(self.writer.log_dir, "checkpoints")
        self.initializes_target_network()

        loss_min = 10000000
        train_loss_min = 10000000
        for epoch_counter in range(self.max_epochs):
            print(f"Epoch={epoch_counter}")

            # 训练
            train_loss_min = self.train(
                train_loader, epoch_counter, train_loss_min, model_checkpoints_folder
            )
            # 验证
            eval_loss_epoch = self.eval(val_loader, epoch_counter)
            if eval_loss_epoch <= loss_min:
                torch.save(
                    self.online_network,
                    os.path.join(model_checkpoints_folder, "model_val_best.pth"),
                )
                loss_min = eval_loss_epoch
            print("End of epoch {}".format(epoch_counter))

        # save checkpoints
        torch.save(
            self.online_network, os.path.join(model_checkpoints_folder, "model.pth")
        )

    def kl_divergence_with_logit(self, q_logit, p_logit):
        q = F.softmax(q_logit, dim=1)
        qlogq = torch.mean(torch.sum(q * F.log_softmax(q_logit, dim=1), dim=1))
        qlogp = torch.mean(torch.sum(q * F.log_softmax(p_logit, dim=1), dim=1))
        return qlogq - qlogp

    def update(self, batch_view, batch_view_adv, batch_view_vat):
        # compute LVAT
        # logit_p = self.online_network(batch_view)[1]
        # logit_m = self.online_network(batch_view_vat)[1]
        # loss_vat = self.kl_divergence_with_logit(logit_p, logit_m)

        # compute query feature
        predictions_from_view_1 = self.predictor(self.online_network(batch_view_adv)[1])
        predictions_from_view_2 = self.predictor(self.online_network(batch_view_vat)[1])

        # compute key features
        with torch.no_grad():
            targets_to_view_2 = self.target_network(batch_view_adv)[1]
            targets_to_view_1 = self.target_network(batch_view_vat)[1]

        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
        return loss.mean()
