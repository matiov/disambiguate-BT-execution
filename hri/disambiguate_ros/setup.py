import os

from setuptools import setup

package_name = 'disambiguate_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        (os.path.join('share', package_name),
            ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Matteo Iovino',
    maintainer_email='matteo.iovino@se.abb.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'disambiguate_service = disambiguate_ros.disambiguate_service:main',
            'disambiguate_client = disambiguate_ros.disambiguate_client:main',
            'detection_hri_client = disambiguate_ros.detection_hri_client:main',
        ],
    },
)
