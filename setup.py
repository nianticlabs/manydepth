import setuptools

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name="manydepth",
    version="1.0.0",
    packages=['manydepth', 'manydepth.networks'],
    scripts=[],
    license='LICENSE',
    url='https://github.com/AdityaNG/monodepth2',
    description="[CVPR 2021] Self-supervised depth estimation from short sequences",
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.6',
)
