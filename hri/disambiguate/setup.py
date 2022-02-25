from setuptools import setup, find_packages

package_name = 'disambiguate'

setup(
    name=package_name,
    version='0.0.0',
    maintainer='Matteo Iovino',
    maintainer_email='matteo.iovino@se.abb.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    packages=find_packages(),
    install_requires=[
        'setuptools',
        'matplotlib',
        'mock',
        'numpy',
        'opencv_contrib_python',
        'pandas',
        'Pillow',
        'protobuf',
        'scikit_image',
        'scikit_learn',
        'scipy',
        'seaborn',
        'six',
        'scikit-image',
        'spacy',
        'tensorflow',
        'torch',
        'torchvision',
    ],
    tests_require=['pytest'],
)
