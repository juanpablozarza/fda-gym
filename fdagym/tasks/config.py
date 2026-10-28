from dataclasses import dataclass


@dataclass(frozen=True)
class GenConfig:
    seed: int = 7
    test_fraction: float = 0.2
    label_open_share: float = 0.10
    # JP, Oct 15: about 5% of tasks in both splits are answerable=false (gold NOT_FOUND).
    unanswerable_share: float = 0.05
    # JP, Oct 15: earliest ORIG date inside one application is not aggregation, so those stay easy.
    single_record_earliest_is_aggregation: bool = False
