from setuptools import setup, find_packages


def get_requirements():
    with open("requirements.txt") as f:
        requirements = [
            line.strip()
            for line in f
            if not line.startswith("-e")
        ]

    return requirements


setup(
    name='classifiers',
    version='0.3.0',
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    zip_safe=False,
    install_requires=get_requirements()
)
