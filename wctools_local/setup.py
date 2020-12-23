import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wctools",  # Replace with your own username
    version="0.1.20201222",
    author="LiaoWC",
    author_email="wcl.cs07@nctu.edu.tw",
    description="A small package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LiaoWC",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "opencv_python==4.4.0.46",
        "joblib==0.16.0",
        "matplotlib==3.3.3",
        "pandas==1.1.4",
        "Werkzeug==0.16.1",
        "Pillow==8.0.1",
        "scikit_learn==0.23.2",
        "seaborn==0.11.1"
    ]
)
