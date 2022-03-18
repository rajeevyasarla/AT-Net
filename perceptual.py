

# --- Imports --- #
import torch
import torch.nn.functional as F


# --- Perceptual loss network  --- #
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_model = vgg_model

    def output_features(self, x):
        return self.vgg_model(x)

    def forward(self, pred_img, gt):
        loss = []
        pred_img_features = self.output_features(pred_img)
        gt_features = self.output_features(gt)
        for pred_img_feature, gt_feature in zip(pred_img_features, gt_features):
            loss.append(F.mse_loss(pred_img_feature, gt_feature))

        return sum(loss)/len(loss)

