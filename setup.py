from setuptools import setup, find_packages

setup(
    name='cell_annotator_net',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch', 'numpy', 'scipy'
    ],
    entry_points={
        'console_scripts': [
            'can-train=train:main',
            'can-infer=infer:main',
        ],
    },
)
