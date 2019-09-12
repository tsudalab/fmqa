import setuptools

setuptools.setup(
    name="fmbqm",
    version="0.0.1",
    author="Koki Kitai",
    author_email="kitai.koki@gmail.com",
    description="Trainable Binary Quadratic Model based on Factorization Machine",
    license="MIT",
    packages=["fmbqm"],
    install_requires=[
        "dimod",
        "mxnet>=1.1.0",
        "numpy>=1.15.0",
    ]
)
