import setuptools

setuptools.setup(
    name="silent_spelling",
    version="0.0.1",
    author="Sean L. Metzger and Jessie R. Liu",
    description="Code to train models, evaluate models, and create figures for the Bravo-1 Spelling Paper",
    packages=setuptools.find_packages(),
    install_requires=['numpy'],
    classifiers=[
        "Programming Language :: Python :: 3.6"
    ],
)
