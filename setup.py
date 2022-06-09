

from setuptools import setup, find_packages

setup(
    name="popv",
    version="0.0.4",
    description="The PopV algorithm",
    long_description="The PopV algorithm for unsupervised label transfer from reference datasets.",
    long_description_content_type="text/markdown",
    author="Nir Yosef",
    author_email="alexander.tarashansky@czbiohub.org",
    keywords="scrnaseq analysis label transfer annotation",
    python_requires=">=3.7",
    # py_modules=["SAM", "utilities", "SAMGUI"],
    install_requires=[
        "gdown==3.13.0",
        "obonet==0.3.0",
        "bbknn==1.5.1",
        "imgkit==1.2.2",
        "scanorama==1.7.1",
        "scanpy==1.8.1",
        "scikit-learn==0.24.0",
        "scikit-misc==0.1.4",
        "scvi-tools==0.16.3",
        "OnClass==1.2",
        "numpy==1.19.2",
        "tensorflow==2.4.4",
        "importlib-metadata==1.7.0",
        "tensorboard==2.9.1",
        "grpcio==1.32.0",
        "huggingface-hub==0.0.12",
        "tqdm==4.64.0",
        "six==1.15.0",
        "typing-extensions==3.7.4.3"
    ],
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
)
