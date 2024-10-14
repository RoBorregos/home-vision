# ----------------------------------------------------------------------
#  Robocup@Home ROS Noetic Docker Development
# ----------------------------------------------------------------------

#: Builds a Docker image with the corresponding Dockerfile file

# ----------------------------BUILD------------------------------------
# ---------vision----------
# No GPU
vision.build:
	@./docker/scripts/build.bash --area=vision

# CUDA 11.8 x86_64
vision.build.cuda:
	@./docker/scripts/build.bash --area=vision --use-cuda

# Jetson devices
vision.build.jetson:
	@./docker/scripts/build.bash --area=vision --jetson-l4t=35.4.1

# ----------------------------CREATE------------------------------------

vision.create:
	@./docker/scripts/run.bash --area=vision --volumes=$(volumes) --name=$(name)

vision.create.cuda:
	@./docker/scripts/run.bash --area=vision --use-cuda --volumes=$(volumes) --name=$(name)

# For jetpack version 35.4.1, jetson images are special in the sense that they are specific to the jetpack version
vision.create.jetson:
	@./docker/scripts/run.bash --area=vision --jetson-l4t=35.4.1 --volumes=$(volumes) --name=$(name)

# ----------------------------START------------------------------------
# Start containers
vision.up:
	@xhost +
	@docker start home-vision

# ----------------------------STOP------------------------------------
# Stop containers
vision.down:
	@docker stop home-vision 

# ----------------------------RESTART------------------------------------
# Restart containers
vision.restart:
	@docker restart home-vision 

# ----------------------------LOGS------------------------------------
# Logs of the container
vision.logs:
	@docker logs --tail 50 home-vision

# ----------------------------SHELL------------------------------------
# Fires up a bash session inside the container
vision.shell:
	@docker exec -it -e "TERM=xterm-256color" --user $(shell id -u):$(shell id -g) home-vision bash

# ----------------------------REMOVE------------------------------------
# Remove container
vision.remove:
	@docker container rm home-vision

# ----------------------------------------------------------------------
#  General Docker Utilities

#: Show a list of images.
list-images:
	@docker image ls

#: Show a list of containers.
list-containers:
	@docker container ls -as