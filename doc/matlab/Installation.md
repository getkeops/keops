```
 ___  ____           ___                    _____            __        
|_  ||_  _|        .'   `.                 |_   _|          [  |       
  | |_/ /   .---. /  .-.  \ _ .--.   .--.    | |      ,--.   | |.--.   
  |  __'.  / /__\\| |   | |[ '/'`\ \( (`\]   | |   _ `'_\ :  | '/'`\ \ 
 _| |  \ \_| \__.,\  `-'  / | \__/ | `'.'.  _| |__/ |// | |, |  \__/ | 
|____||____|'.__.' `.___.'  | ;.__/ [\__) )|________|\'-;__/[__;.__.'  
                           [__|                                        
```

KeOpsLab contains the matlab bindings for the cpp/cuda library KeOps. It provides
standard matlab functions that can be used in any matlab codes.

**Requirements:**

- A unix-like system (typically Linux or MacOs X) with a C++ compiler (gcc>=4.8, clang)
- Cmake>=2.9
- Matlab>=R2012
- Optional : Cuda (>=9.0 is recommended)

# Matlab Install

## Packaged version (recommended)

1. Download and unzip KeOps library at a location of your choice (here denoted as ```/path/to```)

    ```
    wget https://plmlab.math.cnrs.fr/benjamin.charlier/libkeops/-/archive/master/libkeops-master.zip
    unzip libkeops-master.zip
    ```

    Note that temporary files will be written into ```/path/to/libkeops/keopslab/build```
folder, so that this directory must have write permissions.

2. Manually add the directory ```/path/to/libkeops``` to you matlab path [as explained below](#path)

3. Test your installation: [as explained below](#test)

## From source using git

1. clone keops library repo at a location of your choice (here denoted as ```/path/to```)
    
    ```
    git clone https://plmlab.math.cnrs.fr/benjamin.charlier/libkeops.git /path/to/libkeops
    ```

    Note that temporary files will be written into ```./libkeops/keopslab/build```
folder, so that this directory must have write permissions.

2. Manually add the directory ```/path/to/libkeops``` to you matlab path [as explained below](#path)

3. Test your installation: [as explained below](#test)

## Set path <a name="path"></a>

There is two ways to tell matlab where is KeOpsLab:

+ This can be done once and for all, by adding the path to to your matlab. In matlab,  

    ```matlab
    addpath(genpath('/path/to/libkeops'))
    savepath
    ```
+ Otherwise, you can add the following line to the beginning of your matlab scripts:

    ```matlab
    addpath(genpath('/path/to/libkeops'))
    ```

## Testing everything goes fine <a name="test"></a>

[Set the path](#path) and execute the following piece of code in a matlab terminal 

```matlab
x = 1:3; y = 3:5; b = 3:-1:1; p = .25;

my_conv = Kernel('Exp(-p*SqNorm2(x-y))*b','x=Vx(0,3)','y=Vy(1,3)','b=Vy(2,3)', 'p=Pm(3,1)');
my_conv(x',y',b',p)
```

It should return

```matlab
ans =
    0.1494
    0.0996
    0.0498
```

## Troubleshooting

- **Verbosity**: You can force the verbosity level of the compilation by setting the variable 

```matlab
verbosity=1
```
in the file [```/path/to/keops/keopslab/default_options.m```](https://plmlab.math.cnrs.fr/benjamin.charlier/libkeops/blob/master/keopslab/default_options.m). 

- **Old versions of Cuda**: When using KeOps with Cuda version 8 or earlier, the compilation phase for complicated formulas (typically second order gradient or higher derivatives, or even first order gradient for non-standard formulas) may be extremely slow, on the order of several minutes. Typically this happens when running "testShooting" example script. This is due to intensive use of template programming in the code, for which Cuda nvcc compiler prior to version 9 was not optimized. We strongly recommend upgrading to Cuda 9. However Cuda 9 is not anymore compatible with "old" Nvidia cards with compute capability 1 or 2 ; hence the only solution with such cards is to keep Cuda version 8.

- If an error involving cmake appears, it may be due to incorrect libstdc++ linking.
Try the following : exit matlab, then in a terminal type
```bash
export LD_PRELOAD=$(ldd $( which cmake ) | grep libstdc++ | tr ' ' '\n' | grep /)
matlab
```
This will reload matlab with hopefully the correct linking for cmake.