from setuptools import setup, find_packages

setup(
    name="thumbnail-ai",
    version="1.0.0",
    description="AI-powered YouTube Thumbnail Outreach Automation System",
    author="Thumbnail AI Team",
    packages=find_packages(where="modules"),
    package_dir={"": "modules"},
    entry_points={
        "console_scripts": [
            "tai = cli:main",
        ],
    },
)
