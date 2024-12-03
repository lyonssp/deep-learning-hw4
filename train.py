from homework.train_planner import train

if __name__ == '__main__':
    configs = [
        {
            "model_name": "mlp_planner",
            "lr": 1e-3,
            "batch_size": 128,
            "num_epoch": 20
        },
        # {
        #     "model_name": "transformer_planner",
        #     "lr": 1e-3,
        #     "batch_size": 128,
        #     "num_epoch": 20
        # },
        # {
        #     "model_name": "cnn_planner",
        #     "lr": 1e-3,
        #     "batch_size": 128,
        #     "num_epoch": 20
        # }
    ]
    for config in configs:
        train(
            model_name=config["model_name"],
            lr=config["lr"],
            batch_size=config["batch_size"],
            num_epoch=config["num_epoch"]
        )
