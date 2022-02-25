from setuptools import find_packages, setup

package_name = 'robot_behaviors'

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
        'numpy',
        'pickle',
        'behaviors',
        'bt_learning',
        'py_trees',
        'lfd_interface',
    ],
    tests_require=['pytest'],
)
