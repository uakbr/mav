from setuptools import setup

setup(
    name="mav",  
    version="0.0.1",  
    py_modules=["mav"],  # This tells setuptools it's a single file module
    install_requires=[  
        "rich",
        "torch",
        "transformers"
    ],
    entry_points={  # This defines a command-line script
        "console_scripts": [
            "mav=mav:main",  # This means `mav` command will call `main()` from `mav.py`
        ],
    },
    author="attentionmech",
    author_email="attentionmech@gmail.com",
    description="Model Activation Visualizer",
    url="https://github.com/attentionmech/mav",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.6",
)

