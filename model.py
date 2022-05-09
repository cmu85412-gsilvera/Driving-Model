import argparse
from model_utils import (
    seed_everything,
    results_dir,
    load_data,
)
from models import DrivingModel
from visualizer import set_results_dir

if __name__ == "__main__":
    set_results_dir(results_dir)

    seed_everything()

    """Get data"""
    argparser = argparse.ArgumentParser(description="DReyeVR recording parser")
    argparser.add_argument(
        "-f",
        "--file",
        metavar="P",
        default="jacob",
        type=str,
        help="path of the (human readable) recording file",
    )
    argparser.add_argument(
        "--load",
        action="store_true",
        help="whether or not to load (true) or train (false) from data",
    )
    argparser.add_argument(
        "--importances",
        action="store_true",
        help="whether or not to evaluate the model",
    )
    args = argparser.parse_args()
    filename: str = args.file
    load_model: bool = args.load
    vis_imp: bool = args.importances

    if filename is None:
        print("Need to pass in the recording file")
        exit(1)

    train_split, test_split, features, t_train, t_test = load_data(filename)

    model = DrivingModel(features)
    if load_model == True:
        model.load_from_cache()
    else:
        model.begin_training(
            train_split["X"],
            train_split["Y"],
            test_split["X"],
            test_split["Y"],
            t_train,
        )

    # symbolic logic for the driving inputs
    model.output(train_split, test_split, t_train, t_test)
    model.begin_evaluation(test_split["X"], test_split["Y"], t_test, vis_imp=vis_imp)
