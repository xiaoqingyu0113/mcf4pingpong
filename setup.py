from setuptools import setup, find_packages

setup(
    name='mcf4pingpong',
    version='0.0.0',
    description='multi-camera fusion for estimation of 3D ball trajectory',
    author='Qingyu Xiao',
    author_email='xiaoqingyu0113@gmail.com',
    url='https://github.com/xiaoqingyu0113/mcf4pingpong',
    packages=find_packages(include=['mcf4pingpong', 'mcf4pingpong.*']),
    package_data={
        'mcf4pingpong': ['darknet/*'],
    },
    install_requires=[
        'PyYAML',
        'numpy'
    ],
    extras_require={'plotting': ['matplotlib']},
)