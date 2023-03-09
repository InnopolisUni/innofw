from enum import Enum
from typing import ClassVar

from pydantic import Field

from .base import SklearnSchema
from innofw.constants import TaskType


class WeightValues(str, Enum):
    UNIFORM = "uniform"
    DISTANCE = "distance"


class AlgorithmValues(str, Enum):
    AUTO = "auto"
    BALL_TREE = "ball_tree"
    KD_TREE = "kd_tree"
    BRUTE = "brute"


class MetricValues(str, Enum):
    MINKOWSKI = "minkowski"


class KNeighborsClassifierSchema(SklearnSchema):
    task: ClassVar[str] = TaskType.CLASSIFICATION
    name: ClassVar[str] = "knn_classifier"
    target: ClassVar[str] = "sklearn.neighbors.KNeighborsClassifier"

    n_neighbors: int = Field(
        default=5,
        gt=0,
        description="Number of neighbors to use by default for kneighbors queries.",
    )
    weights: WeightValues = Field(
        default=WeightValues.UNIFORM,
        description="Weight function used in prediction.",
    )
    algorithm: AlgorithmValues = Field(
        default=AlgorithmValues.AUTO,
        description="Algorithm used to compute the nearest neighbors.",
    )
    leaf_size: int = Field(
        30,
        gt=0,
        description="Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.",
    )
    p: int = Field(
        2,
        ge=1,
        description="Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.",
    )
    metric: MetricValues = Field(
        MetricValues.MINKOWSKI,
        description="Metric to use for distance computation.",
    )
    # metric_params: dict = Field(
    #     None,
    #     description="Additional keyword arguments for the metric function."
    # )
