import subprocess

from setuptools import setup
from setuptools.command.install import install as _install
from setuptools.command.editable_wheel import editable_wheel as _editable_wheel
# from setuptools.command.build import build as _build

class editable_wheel(_editable_wheel):
    def run(self):
        print("Installing pyrte-rrtmgp")
        try:
            subprocess.run(["conda", "install", "conda-forge::pyrte_rrtmgp", "numpy==1.26.4", "netcdf4==1.7.2"], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during conda install of pyrte-rrtmgp: {e}")
            print("STDOUT: ", e.stdout)
            print("STDERR: ", e.stderr)
        _editable_wheel.run(self)

class install(_install):
    def run(self):
        print("Installing pyrte-rrtmgp")
        try:
            subprocess.run(["conda", "install", "conda-forge::pyrte_rrtmgp", "numpy==1.26.4", "netcdf4==1.7.2"], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during conda install of pyrte-rrtmgp: {e}")
            print("STDOUT: ", e.stdout)
            print("STDERR: ", e.stderr)
        _install.run(self)

# class CustomInstall(install):
#     def run(self):
#         print("INSTALLING PYRTE-RRTMGP\n")
#         try:
#             result = subprocess.run(
#                 ["conda", "install", "conda-forge::pyrte_rrtmgp"],
#                 check=True,
#                 capture_output=True,
#                 text=True,
#             )
#             print("pyRTE-RRTMGP install successful\n")
#             print("STDOUT:", result.stdout)
#             print("STDERR:", result.stderr)
#         except subprocess.CalledProcessError as e:
#             print(f"Error during conda install: {e}")
#             print("STDOUT:", e.stdout)
#             print("STDERR:", e.stderr)
#         except FileNotFoundError:
#             print("Error: 'conda' command not found.")
#         install.run(self)


setup(
    cmdclass={"editable_wheel": editable_wheel, "install": install},
    # cmdclass={"install": CustomInstall},
)
