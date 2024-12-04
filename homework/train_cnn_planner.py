import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from homework.metrics import PlannerMetric

from .models import load_model, save_model
from .datasets.road_dataset import load_data


def train(
    exp_dir: str = "logs",
    num_epoch: int = 100,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    model_name = "cnn_planner"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data(
        "drive_data/train",
        shuffle=True,
        batch_size=batch_size,
        num_workers=2,
        transform_pipeline="aug"
    )
    val_data = load_data("drive_data/val", shuffle=False)

    # create loss function and optimizer
    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    global_step = 0
    training_metrics = PlannerMetric()
    validation_metrics = PlannerMetric()
    metrics = {
        "train_loss": [],
    }

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        training_metrics.reset()
        validation_metrics.reset()

        model.train()

        for batch in val_data:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            img = batch["image"]
            waypoints = batch["waypoints"]
            waypoints_mask = batch["waypoints_mask"]

            pred = model(img)
            training_metrics.add(pred, waypoints, waypoints_mask)

            loss = loss_fn(pred, waypoints)
            metrics["train_loss"].append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for batch in train_data:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                img = batch["image"]
                waypoints = batch["waypoints"]
                waypoints_mask = batch["waypoints_mask"]

                pred = model(img)
                validation_metrics.add(pred, waypoints, waypoints_mask)

        # log average training loss
        logger.add_scalar("train_loss", torch.as_tensor(metrics["train_loss"]).mean(), global_step)

        computed_training_metrics = training_metrics.compute()
        computed_validation_metrics = validation_metrics.compute()

        epoch_train_l1_error = torch.as_tensor(computed_training_metrics["l1_error"])
        epoch_train_longitudinal_error = torch.as_tensor(computed_training_metrics["longitudinal_error"])
        epoch_train_lateral_error = torch.as_tensor(computed_training_metrics["lateral_error"])

        epoch_val_l1_error = torch.as_tensor(computed_validation_metrics["l1_error"])
        epoch_val_longitudinal_error = torch.as_tensor(computed_validation_metrics["longitudinal_error"])
        epoch_val_lateral_error = torch.as_tensor(computed_validation_metrics["lateral_error"])

        # print on first, last, every 10th epoch
        print(
            f"Epoch {epoch + 1}/{num_epoch} "
            f"Train L1 Error: {epoch_train_l1_error:.4f} "
            f"Val L1 Error: {epoch_val_l1_error:.4f} "
            f"Train Longitudinal Error: {epoch_train_longitudinal_error:.4f} "
            f"Val Longitudinal Error: {epoch_val_longitudinal_error:.4f} "
            f"Train Lateral Error: {epoch_train_lateral_error:.4f} "
            f"Val Lateral Error: {epoch_val_lateral_error:.4f}"
        )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))