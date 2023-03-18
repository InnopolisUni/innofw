from typing import ClassVar

from pydantic import Field

from .base import SklearnSchema
from innofw.constants import TaskType


class LinearRegressionSchema(SklearnSchema):
    task: ClassVar[TaskType] = TaskType.REGRESSION
    name: ClassVar[str] = "linear_regression"
    target: ClassVar[str] = "sklearn.linear_model.LinearRegression"

    fit_intercept: bool = Field(
        True,
        description="Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).",
    )
    normalize: bool = Field(
        False,
        description="This parameter is ignored when fit_intercept is set to False.",
    )
    copy_X: bool = Field(
        True,
        description="If True, X will be copied; else, it may be overwritten.",
    )
    positive: bool = Field(
        False,
        description="When set to True, forces the coefficients to be positive. This option is only supported for dense arrays.",
    )
