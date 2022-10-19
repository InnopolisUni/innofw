# from .base import BaseModelAdapter
# from innofw.core.models import register_models_adapter
# from ..integrations import BaseWrapperAdapter
#
#
# @register_models_adapter(name="integrations_adapter")
# class IntegrationsAdapter(BaseModelAdapter):
#     @staticmethod
#     def is_suitable_model(model) -> bool:
#         return isinstance(model, BaseWrapperAdapter)
#
#     def _predict(self, data):
#         pass
#
#     def _test(self, data):
#         pass
#
#     def _train(self, data):
#         pass
