from setuptools import setup

setup(name='image_extractor',
      version='0.1',
      packages=['image_extractor'],
      install_requires=[
          "numpy",
          "opencv-python"
      ],
      entry_points={
          'console_scripts': [
              'image_extractor = image_extractor.__main__:main'
              ]
        },
      )