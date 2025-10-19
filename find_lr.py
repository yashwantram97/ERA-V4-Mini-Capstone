from src.utils.lr_finder_utils import run_lr_finder
from config import (
    lr_finder_kwargs, mean, std, logs_dir,
    experiment_name, learning_rate, weight_decay
)
from src.utils.utils import get_transforms
from src.data_modules.imagenet_datamodule import ImageNetDataModule
from src.models.resnet_module import ResnetLightningModule
import torch
import torch.nn.functional as F

def main():
    train_transforms = get_transforms(transform_type="valid", mean=mean, std=std)
    # validation_transforms = get_transforms(transform_type="valid", mean=mean, std=std)

    imagenet_dm = ImageNetDataModule(
        batch_size=64,
        num_workers=8,
        pin_memory=True
    )

    imagenet_dm.setup(stage='fit')  # This creates imagenet_train, imagenet_val datasets

    experiment_dir = logs_dir / experiment_name

    # Instantiate the Lightning module to get the torch model
    lit_module = ResnetLightningModule(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_classes=1000,
        train_transforms=train_transforms
    )
    model = lit_module.model

    train_loader = imagenet_dm.train_dataloader()
    loss_fn = F.cross_entropy
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)

    suggested_lr = run_lr_finder(
        model,
        train_loader,
        loss_fn,
        optimizer,
        **lr_finder_kwargs,
        save_path=experiment_dir / "lr_finder.png",
        logger=None
    )

    print(f"Suggested LR: {suggested_lr}")

if __name__ == "__main__":
    main()