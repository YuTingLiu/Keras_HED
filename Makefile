
DATA_FILE_PATH = http://vcl.ucsd.edu/hed/HED-BSDS.tar
MODEL_FILE_PATH = https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5


DATA_FILE_NAME = HED-BSDS.tar
MODEL_FILE_NAME = vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5


GET_DATA = wget -c --no-cache -P . ${DATA_FILE_PATH}
GET_MODEL = wget -c --no-cache -P . ${MODEL_FILE_PATH}
UNPACK_DATA = tar -xvf HED-BSDS.tar
DEL_TAR_FILE = rm -rf HED-BSDS.tar
CHECKPOINT_DIR = mkdir checkpoints
.PHONY: all
all: deps compile

.PHONY: deps
deps: model
	@echo "\nmaking deps"

.PHONY: model
model:
	@echo "\nmaking model"
	@if [ -e ${DATA_FILE_NAME} ] ; \
	then \
		echo "model file ${DATA_FILE_NAME} already exists, skipping download"; \
	else \
		echo "Downloading ${DATA_FILE_NAME} file"; \
		${GET_DATA}; \
		if [ -e ${DATA_FILE_NAME} ] ; \
		then \
			echo "download ${DATA_FILE_NAME} done."; \
			${UNPACK_DATA}; \
			${DEL_TAR_FILE}; \
			${CHECKPOINT_DIR}; \
		else \
			echo "***\nError - Could not download ${DATA_FILE_NAME}. Check network and proxy settings \n***\n"; \
			exit 1; \
		fi ; \
	fi
	@if [ -e ${MODEL_FILE_NAME} ] ; \
	then \
		echo "model file ${MODEL_FILE_NAME} already exists, skipping download"; \
	else \
		echo "Downloading ${MODEL_FILE_NAME} file"; \
		${GET_MODEL}; \
		if [ -e ${MODEL_FILE_NAME} ] ; \
		then \
			echo "download ${MODEL_FILE_NAME} done."; \
		else \
			echo "***\nError - Could not download ${MODEL_FILE_NAME}. Check network and proxy settings \n***\n"; \
			exit 1; \
		fi ; \
	fi

.PHONY: train
train:
	@echo "\nmaking train";
	@echo "\nTo train the network the following steps will be taken:";
	@echo "  - Download the TensorFlow mnist_deep.py example";
	@echo "  - Modify the mnist_deep.py -> mnist_deep_mod.py for NCS compatibility";
	@echo "  - run mnist_deep_mod.py to train the network, it will save trained network in:";
	@echo "      ${MODEL_DATA_FILENAME}";
	@echo "      ${MODEL_INDEX_FILENAME}";
	@echo "      ${MODEL_META_FILENAME}";
	@echo "  - modify mnist_deep_mod.py -> mnist_deep_inference.py for inference only version";
	@echo "    of the network (no training code)."
	@echo "  - run mnist_deep_inference.py to read the saved network and write an inference";
	@echo "    only version of the network that the NCSDK can compile in these files:";
	@echo "      ${MODEL_INFERENCE_DATA_FILENAME}";
	@echo "      ${MODEL_INFERENCE_INDEX_FILENAME}";
	@echo "      ${MODEL_INFERENCE_META_FILENAME}";
	@echo "  "
	@echo "After complete, you will be able to compile the network you just trained with:"
	@echo "make compile"
	@echo ""
	@read -p"Press ENTER to continue the steps above (this will take about 20 min) or Ctrl-C to cancel" variable_nps;echo;

	@if [ -e ${NETWORK_SCRIPT_FILENAME} ] ; \
	then \
		echo "network script ${NETWORK_SCRIPT_FILENAME} already exists, skipping download"; \
	else \
		echo "Downloading ${NETWORK_SCRIPT_FILENAME} file"; \
		${GET_NETWORK_SCRIPT}; \
		if [ -e ${NETWORK_SCRIPT_FILENAME} ] ; \
		then \
			echo "download ${NETWORK_SCRIPT_FILENAME} done."; \
		else \
			echo "***\nError - Could not download ${NETWORK_SCRIPT_FILENAME}. Check network and proxy settings \n***\n"; \
			exit 1; \
		fi ; \
	fi ;


	@echo "patching example for NCSDK";
	patch ${NETWORK_SCRIPT_FILENAME} -i ${PATCH_FOR_NCS_FILENAME} -o ${MNIST_NCS_FILENAME};
	patch ${MNIST_NCS_FILENAME} -i ${PATCH_FOR_INFERENCE_FILENAME} -o ${MNIST_INFERENCE_FILENAME};
	@echo "Running patched code to start training";
	python3 ./${MNIST_NCS_FILENAME}
	@echo "Running code to save for NCSDK";
	python3 ./${MNIST_INFERENCE_FILENAME}


.PHONY: run
run: deps compile
	@echo "\nmaking run"
	python3 ./run.py

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - makes the following: deps, compile";
	@echo "  make browse_profile - (TBD) runs the SDK profiler tool and brings up report in browser.";
	@echo "  make check - (TBD) runs SDK checker tool to verify an NCS graph file";
	@echo "  make clean - removes all created content"
	@echo "  make compile - runs SDK compiler tool to compile the graph file for the network";
	@echo "  make deps - downloads and prepares the trained network";
	@echo "  make profile - (TBD) runs the SDK profiler tool to profile the network creating output_report.html";
	@echo "  make run - runs the run.py python example program";
	@echo "  make train - creates NCSDK compatible version of TF network and trains it.";


clean: clean
	@echo "\nmaking clean"
	rm -f ${GRAPH_FILENAME}
	rm -f ${MODEL_INFERENCE_META_FILENAME}
	rm -f ${MODEL_INFERENCE_INDEX_FILENAME}
	rm -f ${MODEL_INFERENCE_DATA_FILENAME}
	rm -f ${NETWORK_SCRIPT_FILENAME}
	rm -f ${MODEL_META_FILENAME}
	rm -f ${MODEL_INDEX_FILENAME}
	rm -f ${MODEL_DATA_FILENAME}
	rm -f checkpoint
	rm -f ${MNIST_NCS_FILENAME}
	rm -f ${MNIST_INFERENCE_FILENAME}
	rm -f output_expected.npy

