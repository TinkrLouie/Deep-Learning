SRC	= magic_matrix_gpu.cpp
OUT	= magic_matrix_gpu
SR = magic_matrix.cpp
OU = magic_matrix
CC = nvc++
FLAGS = -fopenmp -mp=gpu
SHELL := /bin/bash
MOD := nvidia-hpc
RMS = run_all_magic_matrix.sh
NM = install_numact1.sh


default: help

.PHONY: help
help: # Show help for Makefile recipes
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done

.PHONY: mmgpu
mmgpu: $(OUT) # Build the code for GPU execution

$(OUT): $(SRC)
	module load $(MOD) && \
	$(CC) $(FLAGS) -c $^ -o $@.o && \
	$(CC) $(FLAGS) $@.o -o $@

.PHONY: xgpu
xgpu: # Run mmgpu on N=20
	./$(OUT) ./data_sets/pattern20x20.dat ./data_sets/modifier20x20.dat

.PHONY: mmcpu
mmcpu : $(OU) # Build the code for CPU execution

$(OU): $(SR)
	module load $(MOD) && \
	$(CC) $(FLAGS) -c $^ -o $@.o && \
	$(CC) $(FLAGS) $@.o -o $@

.PHONY: rms
rms: # Execute SBATCH script
	chmod +x $(RMS)
	./$(RMS)

.PHONY: nm
nm: # Execute install_numact1 script
	chmod +x $(NM)
	./$(NM)

.PHONY: run1g
run1g: # Open SLURM interactive session with 1g.10gb.1
	srun -n 1 -c 2 --gres=gpu:1g.10gb:1 --partition=ug-gpu-small --pty /bin/bash

.PHONY: runamp
runamp: # Open SLURM interactive session with full ampere GPU
	srun -n 1 -c 2 --gres=gpu:ampere --partition=ug-gpu-small --pty /bin/bash

.PHONY: clean
clean: # Cleanup
	rm -rf $(OUT) $(OUT).o $(OU) $(OU).o

clean_sh: # Clean shell scripts
	rm -rf $(RMS) $(NM)