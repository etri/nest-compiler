# Docker 사용법 및 Jetbrains Clion 연결

---

## Clion에 Docker Plugin 설치
사용법: https://www.jetbrains.com/help/clion/docker.html#using-docker-compose

CLion에서는 Pycharm-pro와 다르게 Docker가 기본으로 서비스로 연결되어 있지 않다. 아래와 같이 Marketplace에서 Docker Plugin을 설치해 준다.
![스크린샷 2021-03-03 오후 5.45.30](https://i.imgur.com/SnkXo9n.png)
- 플러긴 다운로드: https://plugins.jetbrains.com/plugin/7724-docker

설치하면 아래와 같이 settings-> Build, Execution, Deployment->Docker를 통해서 Docker Daemon을 실행 가능하다. 이 부분은 root permission으로 실행해야 하기 때문에 uBuntu라면 sudo로 실행한다. Mac의 경우 권한을 적절한 시점에서 요청한다.
![스크린샷 2021-03-03 오후 5.48.49](https://i.imgur.com/0c9zw3c.png)

서비스를 이제 실행하면 아래와 같이 Docker의 이미지와 컨테이너들의 상태를 볼 수 있다.
- 단축키: cmd+8 (Mac-OS), alt+8 (ubuntu)

![스크린샷 2021-03-05 오전 8.10.33](https://i.imgur.com/4w59bfo.png)

## Dockerfile 생성

아래 Docker파일은 필자가 사용하는 Glow compiler용이다. Pycharm에서와 다르게 Clion에서는 결국 remote toolchain을 설정해서 Docker와 연결하므로 ssh 설정이 필요하다. 아랫쪽 부분의 `sshd` 부분의 설정들이 root로 접속을 허용하며 비밀번호를 root로 설정하는 부분이다.

```bash
FROM ubuntu:20.04

ARG WORKDIR=/root/dev

# Create working folder
RUN mkdir -p $WORKDIR
WORKDIR $WORKDIR

# Update and install tools
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y clang clang-8 cmake graphviz libpng-dev \
        libprotobuf-dev llvm-8 llvm-8-dev ninja-build protobuf-compiler wget \
        opencl-headers libgoogle-glog-dev libboost-all-dev \
        libdouble-conversion-dev libevent-dev libssl-dev libgflags-dev \
        libjemalloc-dev libpthread-stubs0-dev \
        # Additional dependencies
        git python python-numpy && \
    # Delete outdated llvm to avoid conflicts
    apt-get autoremove -y llvm-6.0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Point clang to llvm-8 version
RUN update-alternatives --install /usr/bin/clang clang \
        /usr/lib/llvm-8/bin/clang 50 && \
    update-alternatives --install /usr/bin/clang++ clang++ \
        /usr/lib/llvm-8/bin/clang++ 50

# Point default C/C++ compiler to clang
RUN update-alternatives --set cc /usr/bin/clang && \
    update-alternatives --set c++ /usr/bin/clang++

# Install fmt
RUN git clone https://github.com/fmtlib/fmt && \
    mkdir fmt/build && \
    cd fmt/build && \
    cmake .. && make -j32 && \
    make install

# Clean up
RUN rm -rf fmt

# install packages for CLion
RUN apt-get update \
  && apt-get install -y ssh \
      build-essential \
      gcc \
      g++ \
      gdb \
      clang \
      cmake \
      rsync \
      tar \
      python \
      ssh \
  && apt-get clean

# sshd
RUN ( \
    echo 'LogLevel DEBUG2'; \
    echo 'PermitRootLogin yes'; \
    echo 'PasswordAuthentication yes'; \
    echo 'Subsystem sftp /usr/lib/openssh/sftp-server'; \
  ) > /etc/ssh/sshd_config_test_clion \
  && mkdir /run/sshd

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D", "-e", "-f", "/etc/ssh/sshd_config_test_clion"]

# change password
RUN echo 'root:root' | chpasswd
```

## Docker 이미지 빌드 및 실행
도커를 빌드하고 생성 할 때 IP와 volume, 공유 디렉토리, 포워딩을 포함해서 실행하면 된다.

**터미널 명령어 이용**
```bash
build && RUN
docker build -t nestc-ssh:201122-v1 .
&& docker run
-v /Users/jeminlee/glow_data:/root/glow_data
-v nest-compiler:/root/nestc
-p localhost:22:22
nestc-ssh:201122-v1
```

![스크린샷 2020-11-22 오후 6.33.43](https://i.imgur.com/SC6LYpt.png)


## Docker Pull & Run
위에 설명한 DockerFile로 이미 이미지를 생성하여 DockerHub에 업로드 해두었다.
이렇게 이미지가 이미 있는 경우에는 간단하게 `pull`한 후에 `run`을 통해서 실행만 하면 된다.

```bash
# pull the image from DockerHub
docker pull leejaymin/nestc-ssh:latest

# Run
sudo docker run --name [container name] \
-d --rm \
-v [volume name]:/root/nestc \ # volume:docker disk
-v [path to host dir]:/root/hdd \ # host disk:docker disk
-p 127.0.0.1:20:22 \ # IP:host-port:docker-port
leejaymin/nestc-ssh:latest # image name

# -it, interactive mode and tty enable
# --rm, 컨테이너 삭제 종료 후
# --name container name
# -d detached, 즉 background로 docker를 실행함
```

**Jetbrains IDE 이용**

![스크린샷 2021-03-04 오전 9.34.09](https://i.imgur.com/tYHr1jF.png)

### Docker 내부에서 sshd 데몬 실행

```bash
# 1
sudo docker exec nestc service ssh start

#open ssh  
sudo docker exec --it nestc /bash/bin
service ssh restart
```

Jetbrains IDE 이용
![스크린샷 2021-03-04 오전 9.23.37](https://i.imgur.com/fzoHylK.png)



### Docker Hub에 push

docker push 시 denied: requested access to the resource is denied 오류시
```bash
$ docker tag first-image (Docker Hub 계정)/docker-repo:first-image
$ docker push (Docker Hub 계정)/docker-repo:first-image

이미지 주소: https://hub.docker.com/repository/docker/leejaymin/nestc-ssh

$ docker pull <image name>:<tag>
docker pull leejaymin/nestc-ssh
```
### ssh 설정 방법

1) Docker port binding
2) apt update && apt install openssh-server
3) vi /etc/ssh/sshd_config
    1) LogLevel DEBUG2
    2) PermitRootLogin yes
    3) PasswordAuthentication yes
    4) Subsystem sftp /usr/lib/openssh/sftp-server
    5) port 22 -> 원하는 포트 넘버 (보통 22번은 localhost로 설정할 시 겹치니 다른 것으로 설정함)
4) change root password
- echo 'root:root' | chpasswd


## Jetbrains에 Docker remote 연결하기

이제 Docker 이미지 생성은 끝이고 결국 이제부터는 매번 Clion에서 사용 하듯이 Toolchains에 Remote Host를 하나 더 생성하고(이것이 Docker에 ssh로 접속하는 방식) Host file들을 Docker remote로 sync해서 사용하면된다. 이 후의 과정은 원격 개발환경 구축하는 방법과 동일하다.

![스크린샷 2021-03-04 오전 7.29.58](https://i.imgur.com/t66VCkX.png)
![스크린샷 2021-03-04 오전 7.30.29](https://i.imgur.com/buALKDO.png)

Remote와 파일 싱크
`Tools->Deployments->Configurations`
생성한 Rmoete Host를 선택하고 Mapping tab에서 파일 경로 두개를 맞춰줌
![스크린샷 2021-03-05 오전 8.16.32](https://i.imgur.com/oXfFHAT.png)

## Docker 명령어

### 컨네이너 접속
docker exec -it {container_id or name} /bin/bash

### Docker Volume 생성
매번 nestc 코드를 업로드하는 것을 방지하고 작업 내용을 저장하기 위함

docker volume ls

docker volume create nest-compiler

### Docker resource settings

#### CPUs
```bash
#CPU가 2개 있다고 가정하면 1.5라고 셋팅하면 host CPU에서 1.5개를 사용함.
docker run -it --cpus="1.5" ubuntu /bin/bash
```
추가 옵션: https://docs.docker.com/config/containers/resource_constraints/

### Docker Commit

```bash  
docker commit [CONTAINER] [IMAGE_NAME]
```

### exit된 docker 지우기
```bash
sudo docker ps -a | grep Exit | cut -d ' ' -f 1 | xargs sudo docker rm
```

## 자주 쓰는 Docker 명령어
```bash
docker run : 도커 시작 (creat + start)

ex) docker run --rm -d -p 8080:8080 {image명}

-d : detached mode 흔히 말하는 백그라운드 모드
-p : 호스트와 컨테이너의 디렉토리를 연결 (마운트) [호스트 : 컨테이너]
-e : 컨테이너 내에서 사용할 환경변수 설정
–name : 컨테이너 이름 설정
–rm : 프로세스 종료시 컨테이너 자동 제거
-it : -i와 -t를 동시에 사용한 것으로 터미널 입력을 위한 옵션
-w : working 디렉토리 변경
run : create + start
docker exec

-i : interactive 모드로 실행
-t : terminel 실행
docker exec -it {컨테이너명} {실행할 명령어}
ex) docker exec -it jira-container /bin/sh
docker images : docker 이미지 리스트 출력



docker rmi (docker remove image) : 도커 이미지 삭제

docker rmi {이미지ID}
docker rm : docker 컨테이너 삭제

ex) 중지된 컨테이너 ID를 가져와서 한번에 삭제
docker rm -v $(docker ps -a -q -f status=exited)
docker stop : 실행 중인 컨테이너 중지, 공백으로 구분하여 한번에 중지 가능



docker start : 도커 컨테이너 시작

이미 생성된 컨이너를 실행
재시작은 docker container restart
docker image history {이미지ID}



docker ps : 도커 프로세스 출력

-a 모든 프로세스 출력
docker version : 도커 버전 확인



docker system info : 도커 실행 환경 확인



docker system df : 도커 디스크 상태 확인



docker container stats : 도커 컨테이너 상태 확인



docker logs : 도커 로그 수집

-f : follow
--tail : docker logs --tail 10 : 마지막 10 줄만 출력
도커는 stdout, stderr 를 자동 수집함, 따라서 파일로 만들면 수집하지 않음
docker commit : 현재 상태는 저장

docker commit {container 이름} {이미지 이름}:{태그}
docker login : dockerhub 로그인



docker tag : tag 지정

docker tag {이미지 이름}:{태그}
docker push : 현재 이미지 dockerhub 에 올리기

docker push {이미지 이름}:{태그}
태그 지정했다면 생략 가능

출처: https://dmowld.tistory.com/21 [FREEFLY의 아무거나 쓰는 블로그]
```

## Troubleshotting (문제해결)

### is the docker daemon running
IDE (Clion or Pycharm)을 sudo로 실행하지 않았을 때 생기는 문제이다.




## 참고자료
[Gitbook](https://docker.books.giljae.com/7.-data-volumes/7.2-volume-volume)  
[Clion Dokcer 비밀번호 설정 및 디버깅](https://www.popit.kr/full-remote-mode-of-clion-with-docker/)  
[Jetbrains Clion 공식 예제 github](https://github.com/JetBrains/clion-remote/blob/master/Dockerfile.remote-cpp-env)  
[Jetbrain Clion Docker blog](https://blog.jetbrains.com/clion/2020/01/using-docker-with-clion/)  
[Docker 기본 정보](https://subicura.com/2017/01/19/docker-guide-for-beginners-2.html)  
[Docker Volume 활용](https://medium.com/dtevangelist/docker-%EA%B8%B0%EB%B3%B8-5-8-volume%EC%9D%84-%ED%99%9C%EC%9A%A9%ED%95%9C-data-%EA%B4%80%EB%A6%AC-9a9ac1db978c)
