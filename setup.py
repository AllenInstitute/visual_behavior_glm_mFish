from setuptools import setup

setup(name='visual-behavior-glm',
      packages=['visual_behavior_glm'],
      version='0.0.2',
      description='GLM for ophys mFish learning project',
      url='https://github.com/yavorska-iryna/visual_behavior_glm_mFish',
      author='Alex Piet, Iryna Yavorska, Matt Davis, Marina Garrett',
      author_email="alex.piet@alleninstitute.org, iryna.yavorska@alleninsitute.org, 'matt.davis@alleninstitue.org', marinag@alleninstitute.org",
      license='Allen Institute',
      dependency_links=['https://github.com/mattjdavis/AllenSDK.git'],
      install_requires=[
        "h5py",
        "matplotlib",
        "plotly",
        "pandas==1.5.3",
        "seaborn",
        "numpy",
        "pymongo",
        "xarray==0.15.1",
        "xarray_mongodb",
      ],
     )