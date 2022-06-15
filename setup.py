from setuptools import find_packages, setup


setup(
    name="tcrf",
    version="0.0.1",
    author="Amardeep Kumar",
    author_email="kumaramardipsingh@gmail.com",
    description=" A deep learning based sequence tagging library with CRF layer on the top of transformer models.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ad6398/t-crf",
    download_url="https://github.com/ad6398/t-crf",
    project_urls={
        "Bug Tracker": "https://github.com/ad6398/t-crf/issues",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    license="MIT License",
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=[
        "transformers>=4.6.0",
        "datasets",
        "scikit-learn",
        "seqeval",
    ],
)
