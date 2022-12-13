# PopV

PopV uses popular vote of a variety of cell-type transfer tools to classify cell-types in a query dataset based on a test dataset.
Using this variety of algorithms, we compute the agreement between those algorithms and use this agreement to predict which cell-types are with a high likelihood the same cell-types observed in the reference.

## Algorithms

Currently implemented algorithms are:

-   K-nearest neighbor classification after dataset integration with [BBKNN](https://github.com/Teichlab/bbknn)
-   K-nearest neighbor classification after dataset integration with [SCANORAMA](https://github.com/brianhie/scanorama)
-   K-nearest neighbor classification after dataset integration with [scVI](https://github.com/scverse/scvi-tools)
-   Random forest classification
-   Support vector machine classification
-   [OnClass](https://github.com/wangshenguiuc/OnClass) cell type classification
-   [scANVI](https://github.com/scverse/scvi-tools) label transfer

All algorithms are implemented as class in popv/algorithms.
To implement a new method, a class has to have several methods:

-   algorithm.compute_integration: Computes dataset integration to yield an integrated latent space.
-   algorithm.predict: Computes cell-type labels based on the specific classifier.
-   algorithm.compute_embedding: Computes UMAP embedding of previously computed integrated latent space.

## Installation

We suggest using a package manager like conda or mamba to install the package. Due to various requirements of the different algorithms requirements of PopV are strict. OnClass files for annotation based on Tabula sapiens are deposited in popv/ontology.

    conda create -n yourenv python=3.8
    conda activate yourenv
    pip install git+https://github.com/czbiohub/PopV

## Example notebook

We deposited an example notebook in Google Colab:

-   [Tutorial](https://colab.research.google.com/drive/1mVf4Ksb9WQJ77wEFFduNTHLjON4FlsGc#scrollTo=ZnoRUg58Aq4-)

This notebook will guide you through annotating a dataset based on the annotated [Tabula sapiens reference](https://tabula-sapiens-portal.ds.czbiohub.org) and demonstrates how to run annotation on your own query dataset.

Memory requirements exceed the free limit in Colab and we recommend a Pro access to run the noteook.
