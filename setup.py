import subprocess

from setuptools import setup
from setuptools.command.install import install


class CustomInstall(install):
    def run(self):
        print("INSTALLING PYRTE-RRTMGP\n!!!\n!!!\n!!!")
        subprocess.run(["conda", "install", "pyrte_rrtmgp"])
        install.run(self)


setup(
    cmdclass={"install": CustomInstall},
)
