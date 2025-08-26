import subprocess

from setuptools import setup
# from setuptools.command.install import install
from setuptools.command.build import build


class CustomInstall(build):
    def run(self):
        print("INSTALLING PYRTE-RRTMGP\n!!!\n!!!\n!!!")
        try:
            result = subprocess.run(
                ["conda", "install", "conda-forge::pyrte_rrtmgp"],
                check=True,
                capture_output=True,
                text=True,
            )
            print("pyRTE-RRTMGP install successful\n\n\n")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Error during conda install: {e}")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
        except FileNotFoundError:
            print("Error: 'conda' command not found.")
        build.run(self)


setup(
    cmdclass={"build": CustomInstall},
)
