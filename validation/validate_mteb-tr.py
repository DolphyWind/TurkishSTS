import warnings

from validation_utils import evaluate_all_mteb_tr


def main():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="'n_jobs' has no effect",
            category=FutureWarning,
            module="sklearn.linear_model._logistic",
        )
        evaluate_all_mteb_tr(25)


if __name__ == '__main__':
    main()
