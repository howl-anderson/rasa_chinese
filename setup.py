from setuptools import setup, find_packages

setup(
    name='rasa_chinese',
    version='2.1.0',
    packages=find_packages(),
    url='https://github.com/howl-anderson/rasa_chinese',
    license='Apache 2.0',
    author='Xiaoquan Kong',
    author_email='u1mail2me@gmail.com',
    description='A Chinese language extension package for Rasa',
    install_requires=["rasa~=2.1"]
)
