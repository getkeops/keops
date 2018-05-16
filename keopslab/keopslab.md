
## Quick start

Three steps:

1) Download keops library and unzip it at a location of your choice. Note that temporary files will be written into keopslab/build folder, so that this directory must hhave write permissions.

2) Within Matlab, run the out-of-the-box working examples located in `./matlab/examples/`

3) To use keops in your own Matlab codes, set the Matlab path to include "keopslab" folder and all its subfolders.

N.B. Everytime you need to update or reinstall the library, make sure you replace the full directory keopslab, so that temporary files will be erased.


## Troubleshooting

If an error involving libstdc++.so.6 occurs like

```
cmake: /usr/local/MATLAB/R2017b/sys/os/glnxa64/libstdc++.so.6: version `CXXABI_1.3.9' not found (required by cmake)
cmake: /usr/local/MATLAB/R2017b/sys/os/glnxa64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found (required by cmake)
cmake: /usr/local/MATLAB/R2017b/sys/os/glnxa64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found (required by /usr/lib/x86_64-linux-gnu/libjsoncpp.so.1)
```

try to load matlab with the following linking variable :

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6;matlab
```