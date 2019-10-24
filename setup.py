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
    version='0.4.0',
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    zip_safe=False,
    install_requires=get_requirements(),
    classifiers=(
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: Implementation :: CPython',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    )
)
