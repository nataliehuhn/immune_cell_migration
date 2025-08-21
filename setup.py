from setuptools import setup, find_packages

setup(
    name="immune_cell_migration",
    version="0.1.0",
    description="A Python package for tracking immune cells.",
    url="",
    author="Natalie Huhn",
    author_email="",
    license="MIT",
    packages=find_packages(),
    zip_safe=False,
    install_requires=["numpy", "joblib", "tifffile", "matplotlib", "natsort", "scipy", "scikit-image", "clickpoints", "peewee", "tqdm", "tensorflow==2.15", "keras"]
)
