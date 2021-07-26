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
        "gdown",
        "obonet",
        "bbknn",
        "OnClass",
        "scvi-tools",
        "imgkit",
        "scanorama"
    ],
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
)
