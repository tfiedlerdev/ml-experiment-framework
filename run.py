from src.args.argparsing import get_experiment_from_args

if __name__ == "__main__":
    experiment = get_experiment_from_args()
    experiment.run()