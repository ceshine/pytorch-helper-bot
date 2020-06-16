from distutils.core import setup

setup(
    name='pytorch_helper_bot',
    version='0.6.0',
    packages=['pytorch_helper_bot'],
    install_requires=[
        'torch>=1.5.0',
        'dataclasses',
        'tqdm>=4.29.1',
        'scikit-learn>=0.21.2',
        'tensorboardX>=1.8'
    ],
    extras_require={
        'telegram':  ["python-telegram-bot>=12.0.0"],
        'wandb': ["wandb>=0.8.14"]
    },
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',

        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
    ],
)
