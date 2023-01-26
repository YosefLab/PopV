from ._bbknn import BBKNN as knn_on_bbknn
from ._onclass import ONCLASS as onclass
from ._rf import RF as rf
from ._scanorama import SCANORAMA as knn_on_scanorama
from ._scanvi import SCANVI_POPV as scanvi
from ._scvi import SCVI_POPV as knn_on_scvi
from ._svm import SVM as svm
from ._celltypist import CELLTYPIST as celltypist

__all__ = [
    "knn_on_scvi",
    "scanvi",
    "knn_on_bbknn",
    "svm",
    "rf",
    "onclass",
    "knn_on_scanorama",
    "celltypist",
]
