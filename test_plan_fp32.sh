export DT=fp16
# 1N1C
sh projects/ernie/auto_export_ernie_345M_mp1_intel_gpu.sh
BS=1 sh projects/ernie/run_inference_intel_gpu.sh |& tee ./fp32/inf345m_1n1c_bs1.log
BS=2 sh projects/ernie/run_inference_intel_gpu.sh |& tee ./fp32/inf345m_1n1c_bs2.log
BS=4 sh projects/ernie/run_inference_intel_gpu.sh |& tee ./fp32/inf345m_1n1c_bs4.log
BS=8 sh projects/ernie/run_inference_intel_gpu.sh |& tee ./fp32/inf345m_1n1c_bs8.log
BS=16 sh projects/ernie/run_inference_intel_gpu.sh |& tee ./fp32/inf345m_1n1c_bs16.log
BS=32 sh projects/ernie/run_inference_intel_gpu.sh |& tee ./fp32/inf345m_1n1c_bs32.log
BS=64 sh projects/ernie/run_inference_intel_gpu.sh |& tee ./fp32/inf345m_1n1c_bs64.log
BS=128 sh projects/ernie/run_inference_intel_gpu.sh |& tee ./fp32/inf345m_1n1c_bs128.log
BS=256 sh projects/ernie/run_inference_intel_gpu.sh |& tee ./fp32/inf345m_1n1c_bs256.log
BS=512 sh projects/ernie/run_inference_intel_gpu.sh |& tee ./fp32/inf345m_1n1c_bs512.log
# 1N2C DP=2
sh projects/ernie/auto_export_ernie_345M_dp2_intel_gpu.sh
BS=1 sh projects/ernie/run_inference_dp2_intel_gpu.sh |& tee ./fp32/inf345m_1n2c_dp2_bs1.log
BS=2 sh projects/ernie/run_inference_dp2_intel_gpu.sh |& tee ./fp32/inf345m_1n2c_dp2_bs2.log
BS=4 sh projects/ernie/run_inference_dp2_intel_gpu.sh |& tee ./fp32/inf345m_1n2c_dp2_bs4.log
BS=8 sh projects/ernie/run_inference_dp2_intel_gpu.sh |& tee ./fp32/inf345m_1n2c_dp2_bs8.log
BS=16 sh projects/ernie/run_inference_dp2_intel_gpu.sh |& tee ./fp32/inf345m_1n2c_dp2_bs16.log
BS=32 sh projects/ernie/run_inference_dp2_intel_gpu.sh |& tee ./fp32/inf345m_1n2c_dp2_bs32.log
BS=64 sh projects/ernie/run_inference_dp2_intel_gpu.sh |& tee ./fp32/inf345m_1n2c_dp2_bs64.log
BS=128 sh projects/ernie/run_inference_dp2_intel_gpu.sh |& tee ./fp32/inf345m_1n2c_dp2_bs128.log
BS=256 sh projects/ernie/run_inference_dp2_intel_gpu.sh |& tee ./fp32/inf345m_1n2c_dp2_bs256.log
BS=512 sh projects/ernie/run_inference_dp2_intel_gpu.sh |& tee ./fp32/inf345m_1n2c_dp2_bs512.log
# 1N2C MP=2
sh projects/ernie/auto_export_ernie_345M_mp2_intel_gpu.sh
BS=1 sh projects/ernie/run_inference_mp2_intel_gpu.sh |& tee ./fp32/inf345m_1n2c_bs1.log
BS=2 sh projects/ernie/run_inference_mp2_intel_gpu.sh |& tee ./fp32/inf345m_1n2c_bs2.log
BS=4 sh projects/ernie/run_inference_mp2_intel_gpu.sh |& tee ./fp32/inf345m_1n2c_bs4.log
BS=8 sh projects/ernie/run_inference_mp2_intel_gpu.sh |& tee ./fp32/inf345m_1n2c_bs8.log
BS=16 sh projects/ernie/run_inference_mp2_intel_gpu.sh |& tee ./fp32/inf345m_1n2c_bs16.log
BS=32 sh projects/ernie/run_inference_mp2_intel_gpu.sh |& tee ./fp32/inf345m_1n2c_bs32.log
BS=64 sh projects/ernie/run_inference_mp2_intel_gpu.sh |& tee ./fp32/inf345m_1n2c_bs64.log
BS=128 sh projects/ernie/run_inference_mp2_intel_gpu.sh |& tee ./fp32/inf345m_1n2c_bs128.log
BS=256 sh projects/ernie/run_inference_mp2_intel_gpu.sh |& tee ./fp32/inf345m_1n2c_bs256.log
BS=512 sh projects/ernie/run_inference_mp2_intel_gpu.sh |& tee ./fp32/inf345m_1n2c_bs512.log
# # 1N4C
# sh projects/ernie/auto_export_ernie_345M_mp4_intel_gpu.sh
# BS=1 sh projects/ernie/run_inference_mp4_intel_gpu.sh |& tee ./fp32/inf345m_1n4c_bs1.log
# BS=2 sh projects/ernie/run_inference_mp4_intel_gpu.sh |& tee ./fp32/inf345m_1n4c_bs2.log
# BS=4 sh projects/ernie/run_inference_mp4_intel_gpu.sh |& tee ./fp32/inf345m_1n4c_bs4.log
# BS=8 sh projects/ernie/run_inference_mp4_intel_gpu.sh |& tee ./fp32/inf345m_1n4c_bs8.log
# BS=16 sh projects/ernie/run_inference_mp4_intel_gpu.sh |& tee ./fp32/inf345m_1n4c_bs16.log
# BS=32 sh projects/ernie/run_inference_mp4_intel_gpu.sh |& tee ./fp32/inf345m_1n4c_bs32.log
# BS=64 sh projects/ernie/run_inference_mp4_intel_gpu.sh |& tee ./fp32/inf345m_1n4c_bs64.log
# BS=128 sh projects/ernie/run_inference_mp4_intel_gpu.sh |& tee ./fp32/inf345m_1n4c_bs128.log
# BS=256 sh projects/ernie/run_inference_mp4_intel_gpu.sh |& tee ./fp32/inf345m_1n4c_bs256.log
# BS=512 sh projects/ernie/run_inference_mp4_intel_gpu.sh |& tee ./fp32/inf345m_1n4c_bs512.log
# BS=1024 sh projects/ernie/run_inference_mp4_intel_gpu.sh |& tee ./fp32/inf345m_1n4c_bs1024.log
