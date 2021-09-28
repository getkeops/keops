import cppyy

cppyy.include("/usr/local/cuda/include/nvrtc.h")
cppyy.include("/usr/local/cuda/include/cuda.h")

cppyy.cppdef("""
template <typename TYPE>
class context {
int current_device_id;
CUcontext ctx;
CUmodule module;
char *target;
public:
context();
~context();
};
""")


cppyy.load_library("keops_nvrtc.so")

a = cppyy.gbl.context["float"]()

print("a=", a)

print("hi guy", flush=True)

a.__destruct__()

print("hi guy 2", flush=True)


