################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/CCodeGenTest.cpp \
../src/Image.cpp \
../src/library.cpp \
../src/network.cpp 

OBJS += \
./src/CCodeGenTest.o \
./src/Image.o \
./src/library.o \
./src/network.o 

CPP_DEPS += \
./src/CCodeGenTest.d \
./src/Image.d \
./src/library.d \
./src/network.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -std=c++0x -O0 -g3 -Wall -fopenmp -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


