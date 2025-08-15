import subprocess
from setuptools import setup
from setuptools.command.install import install

class CustomInstall(install):
    def run(self):
        subprocess.run(["mamba","install","pyrte_rrtmgp"])
        install.run(self)


setup(
    cmdclass={"install": CustomInstall},
)