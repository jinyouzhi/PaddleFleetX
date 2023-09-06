
# load oneapi
_P="/opt/intel/oneapi/compiler/latest/env/vars.sh"
echo -e "source ${_P}"
source ${_P}
_P="/opt/intel/oneapi/mkl/latest/env/vars.sh"
echo -e "source ${_P}"
source ${_P}
unset _P
unset _comp

# load oneCCL
source $(python -c 'import site, os; print(os.path.join(site.getsitepackages()[0], "paddle_custom_device/oneCCL/env/setvars.sh"))')

export PADDLE_XCCL_BACKEND="intel_gpu"
export PADDLE_DISTRI_BACKEND="xccl"
export FLAGS_selected_intel_gpus="0,1"

export CCL_ZE_IPC_EXCHANGE=sockets
# ENV for ATS-M, double type
#export OverrideDefaultFP64Settings=1
#export IGC_EnableDPEmulation=1


