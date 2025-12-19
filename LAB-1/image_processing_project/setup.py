from setuptools import setup, find_packages

setup(
    name="image_processor",
    version="1.0.0",
    author="Image Processing Team",
    description="Advanced image processing package with async support",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "Pillow>=9.0.0",
        "opencv-python>=4.5.0",
        "aiohttp>=3.8.0",
        "aiofiles>=0.8.0",
        "python-dotenv>=0.19.0"
    ],
)