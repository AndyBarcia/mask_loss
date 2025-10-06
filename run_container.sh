docker run -d \
	--gpus all \
	--ipc=host \
	--privileged \
	--rm -it \
	--user user \
	--name mask_loss1 \
	-e "CUDA_VISIBLE_DEVICES=1" \
	-e "CUDA_DEVICE_ORDER=PCI_BUS_ID" \
	-e "DETECTRON2_DATASETS=/data/datasets/" \
	-v $PWD:/home/user/pos_mlp_bias \
	clipdino-torch22-cu115-triton2
