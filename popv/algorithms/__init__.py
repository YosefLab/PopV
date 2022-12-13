from ._bbknn import BBKNN as knn_on_bbknn_pred
from ._onclass import ONCLASS as onclass_pred
from ._rf import RF as rf_pred
from ._scanorama import SCANORAMA as knn_on_scanorama_pred
from ._scanvi import SCANVI_POPV as scanvi_pred
from ._scvi import SCVI_POPV as knn_on_scvi_pred
from ._svm import SVM as svm_pred

__all__ = [
    "knn_on_scvi_pred",
    "scanvi_pred",
    "knn_on_bbknn_pred",
    "svm_pred",
    "rf_pred",
    "onclass_pred",
    "knn_on_scanorama_pred",
]
