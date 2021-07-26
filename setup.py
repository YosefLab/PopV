
from setuptools import setup, find_packages

setup(
    name="popv",
    version="0.0.1",
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
        "scvi-tools @ https://github.com/yoseflab/scvi-tools@master#egg=scvi-tools[tutorials]",
        "OnClass @ https://github.com/wangshenguiuc/OnClass@21232f293a549a7ee0da8ebe3cbb22df3e885d4c",
        "numpy~=1.19.2",
        "tensorflow<2.5.0",
        "torch==1.9.0+cu111",
        "torchvision==0.10.0+cu111",
        "typing-extensions~=3.7.2",
        "importlib-metadata<2.0,>=1.0",
        "tensorboard!=2.5.0,>=2.2.0",
        "grpcio~=1.32.0",
        "huggingface-hub==0.0.12"
    ],
    dependency_links=['https://download.pytorch.org/whl/torch_stable.html'],
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
)
