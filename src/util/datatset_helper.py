def suggest_split(
    sample_idx: int,
    num_samples: int,
    train_percentage: float,
    test_equals_val: bool = False,
) -> str:
    if sample_idx / num_samples < train_percentage:
        return "train"
    if test_equals_val:
        return "val"
    test_threshold = train_percentage + (1 - train_percentage) / 2
    if sample_idx / num_samples < test_threshold:
        return "val"
    return "test"
