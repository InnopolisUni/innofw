class BaseIntegrationModel:  # todo: make an abstract field
    """
        Class defines an abstract integration model

        Attributes
        ----------
        framework: Optional[List[Union[str, Frameworks]]]
            the model's framework

    """
    framework = None
