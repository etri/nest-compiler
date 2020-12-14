################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/backup/library.cpp \
../src/backup/network.cpp 

OBJS += \
./src/backup/library.o \
./src/backup/network.o 

CPP_DEPS += \
./src/backup/library.d \
./src/backup/network.d 


# Each subdirectory must supply rules for building sources it contributes
src/backup/%.o: ../src/backup/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -std=c++0x -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


