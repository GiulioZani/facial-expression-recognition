from torch.nn.modules.activation import ReLU
import ipdb
import torch as t
from torch import nn
import torch.nn.functional as F
from argparse import Namespace
import torch as t
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from ..resnet.model import Model
import os


def main():
    curdir = os.path.join("dl", "models", "resnet_visualization",)
    params = Namespace(
        **json.load(open(os.path.join(curdir, "default_parameters.json")))
    )
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    classifier = Model(params).to(device)
    classifier.load_state_dict(
        t.load(os.path.join(curdir, "checkpoint.ckpt"), map_location=device)
    )
    # for param in classifier.parameters():
    #    param.requires_grad = False  # freeze the pretrained model
    classifier.freeze()
    learnable_image = t.zeros(
        (1, 1, params.imsize, params.imsize), device=device, requires_grad=True
    )
    target_label = t.tensor(
        [0, 0, 0, 0, 0, 1, 0, 0,], device=device, dtype=t.float32
    ).unsqueeze(0)
    optimizer = t.optim.Adam([learnable_image], lr=1e-3)
    losses = []
    for i in tqdm(range(5000)):
        if i % 1000 == 0:
            optimizer.param_groups[0]["lr"] *= 0.1
        pred_label = classifier(t.sigmoid(learnable_image))
        # print(f"{pred_label=}")
        loss = F.cross_entropy(pred_label, target_label)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    learnable_image = t.sigmoid(learnable_image).detach().cpu()
    plt.plot(losses)
    plt.show()
    print(t.round(t.softmax(classifier(learnable_image.cuda()).cpu(), dim=1)))
    plt.imshow(learnable_image[0, 0])
    plt.show()


if __name__ == "__main__":
    main()
