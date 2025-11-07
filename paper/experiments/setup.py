if __name__ == "__main__":
    import os

    from cellmap_segmentation_challenge.utils.datasplit import make_datasplit_csv
    from common import classes, crops, datasets, force_all_classes, validation_prob

    if not os.path.exists("datasplit.csv"):
        make_datasplit_csv(
            classes=classes,
            force_all_classes=force_all_classes,
            validation_prob=validation_prob,
            datasets=datasets,
            crops=crops,
        )
