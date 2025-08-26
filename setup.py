import subprocess

from setuptools import setup
from setuptools.command.build import build


class CustomBuild(build):
    def run(self):
        print("INSTALLING PYRTE-RRTMGP\n!!!\n!!!\n!!!")
        subprocess.run(["conda", "install", "pyrte_rrtmgp"])
        build.run(self)


setup(
    cmdclass={"build": CustomBuild},
)
