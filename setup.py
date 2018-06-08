from setuptools import setup

setup(name='imextract',
      version='0.1',
      packages=['imextract'],
      install_requires=[
          "numpy",
          "opencv-python"
      ],
      entry_points={
          'console_scripts': [
              'imextract = imextract.__main__:main'
              ]
        },
      )