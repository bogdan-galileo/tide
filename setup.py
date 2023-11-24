import pathlib
from setuptools import setup
import json

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text(encoding="utf8")
with open("tidecv/__init__.py") as f:
    VERSION = json.loads(f.readlines()[0].split("=")[1].strip())

# This call to setup() does all the work
setup(
    name="rungalileo-tidecv",
    version=VERSION,
    description="A General Toolbox for Identifying ObjectDetection Errors",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/rungalileo/tide",
    author="Galileo",
    author_email="galileo@rungalileo.io",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    packages=["tidecv", "tidecv.errors"],
    include_package_data=True,
    install_requires=["appdirs", "numpy", "pycocotools", "pandas"],
    # entry_points={
    #     "console_scripts": [
    #         "tidecv=tidecv.__main__:main",
    #     ]
    # },
)
