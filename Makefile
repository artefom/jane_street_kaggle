IMAGE_NAME ?= kaggle/jange_street_kaggle
IMAGE_TAG ?= latest
CONTAINER_NAME ?= docker_jange_street_kaggle

build:
	docker build -f Dockerfile -t ${IMAGE_NAME}:${IMAGE_TAG} .
	docker push ${IMAGE_NAME}:${IMAGE_TAG}

run:
	- docker rm ${CONTAINER_NAME}
	docker run -t -d --name ${CONTAINER_NAME} ${IMAGE_NAME}:${IMAGE_TAG}
