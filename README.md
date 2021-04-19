# PyFrac

PyFrac is a simulator for the propagation of planar 3D fluid driven fractures written in Python. It is based on an implicit level set description of the fracture.

Copyright © ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2020.
All rights reserved.

PyFrac is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

You should have received a copy of the GNU Lesser General Public License along with PyFrac. If not, see <http://www.gnu.org/licenses/>.

### Disclaimer
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Contributors

- Haseeb Zia (2016-)
- Carlo Peruzzo (2019-)
- Andreas Mori (2019-)
- Brice Lecampion (2016-)
- Fatima-Ezzahra Moukhtari (2017-2019)
- Weihan Zhang (2017)
 
## How to cite

- H. Zia and B. Lecampion. PyFrac: a planar 3D hydraulic fracturing simulator. *Comp. Phys. Comm.*, 255:107368, (2020). https://doi.org/10.1016/j.cpc.2020.107368


## Current Version

1.1.1 (November 2020)


## Support

We do not have the capacities to provide support, but will do our best to address users questions. 
To do so, we have created a Slack channel for PyFrac where you can post questions. 
You can join the channel via the following link:
https://join.slack.com/t/pyfrac/shared_invite/zt-hqt8wg1w-_7YC4PBNitp7C~a_0ygm~A

Bugs should be reported as issues via GitHub.

## Getting started

Running PyFrac will require a functioning installation of Python 3, numpy, SciPy and matplotlib. You can download ANACONDA distribution with all these packages from <https://www.anaconda.com/distribution/>. To run PyFrac using Unix shell or windows command prompt, the directory containing the PyFrac source code is required to be added to the ``PYTHONPATH`` environment variable. It can be added with the following command on windows (give the local path of the ``src`` folder in place of path_of_PyFrac_src):

    set PYTHONPATH=%PYTHONPATH%;path_of_PyFrac_src

and with the following for linux or mac:

    export PYTHONPATH=${PYTHONPATH}:path_of_PyFrac_src

PyFrac uses dill package for saving files on hard disk. You can use pip to install the latest distribution of the package with the following command:

    pip install dill

If you already have python 3 installed through anaconda, update installed packages to the latest version. You can use the following to update all installed packages:

   conda update --all

### Transverse Isotropic Kernel

PyFrac uses a routine written in C++ to evaluate elasticity kernel for the transversely isotropic materials. This C++ code has to be compiled before the fracture simulation can be done for transverse isotropic materials. Use the following steps to generate the executable:

**Note:**   The setup below is required only if you want to simulate fracture propagation in transversely isotropic materials.

The code uses the Inside Loop (il) library which requires installation of OpenBLAS. See https://github.com/InsideLoop/InsideLoop. We ship the il source codes with this release for simplicity.  Follow the instruction below for your operating system in order to compile the elastic TI code for planar fracture and rectangular mesh.

#### windows

   1. Download and install OpenBLAS. You can also download binary packages available for windows (preferred).
   2. Download and install MSYS2.
   3. Install gcc and cmake for MSYS2 using the following:

    pacman -S base-devel gcc vim cmake
   4. In case you have downloaded the binary packages for OpenBLAS, you would have to provide the location of the OpenBLAS libraries. You can do that by providing the location in the CmakeLists file.
   5. Change directory to the TI_Kernel\\build folder in PyFrac. Create the executable using cmake by running the following commands one by one:

    cmake ..  
    make

   6. Add MSYS2 libraries path (typically C:\\msys64\\usr\\bin) to the windows `PATH` environment variable.

#### Linux

   1. Install OpenBlas and LAPACK with the following commands:

    sudo apt-get install libopenblas-dev  
    sudo apt-get install liblapacke-dev

   2. Install Cmake with the following.

    sudo apt-get -y install cmake

   3. Change directory to the TI_Kernel/build folder in PyFrac. Create the executable using cmake by running the following commands one by one:

    cmake ..  
    make

#### Mac

   1. Install OpenBlas with the following:

    brew install openblas

   2. Install Cmake with the following:

    brew install cmake

   3. Change directory to the TI_Kernel/build folder in PyFrac. Create the executable using cmake by running the following commands one by one:

    cmake ..  
    make

## Documentation

You can generate documentation locally using sphinx. First install shpinx using pip:

    pip install sphinx

Then change directory to the Doc folder present in the PyFrac code. Run the make command to build the documentation in html:

    make html

or in pdf as:

    make latexpdf

After the build is complete, you can access the documentation in the build folder. For html, start with the file named index. The pdf file is located in the subflolder latex.

## Running an example


Change directory to the folder containing the PyFrac code. An example can be run from the windows command prompt or linux shell by executing the example script e.g.:

    python ./examples/radial_viscosity_explicit.py

There are scripts available for a set of examples in the examples folders provided with the code, including the scripts to reproduce the results presented in the paper published in Computer Physics Communications (https://doi.org/10.1016/j.cpc.2020.107368). The corresponding example number from the paper is mentioned in the name of these scripts.

**Note:**   Some of the examples may take upto 2 hours to run.
