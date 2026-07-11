gst-launch-1.0 filesrc location="/home/shangar21/Downloads/monster.mkv" ! matroskademux ! h265parse ! nvv4l2decoder ! \
  mux.sink_0 nvstreammux name=mux batch-size=1 width=1440 height=1080 ! \
  nvvideoconvert ! \
  nvdsvideotemplate customlib-name="/opt/nvidia/deepstream/deepstream-9.0/lib/libcustom_videoimpl.so" \
    customlib-props="engine-path:/home/shangar21/Documents/upscaler/model/trt_engines/apisr_fp16.trt" \
  ! nvvideoconvert ! nv3dsink
