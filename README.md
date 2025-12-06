# BipedalWalker-PPO-STM32

**Train a bipedal robot with PPO and deploy it on STM32 microcontroller using TFLite Micro**
---

## 🎯 Overview

End-to-end deployment of reinforcement learning agent from simulation to embedded hardware:

1. **Train** PPO agent on BipedalWalker-v3 environment
2. **Convert** model to TFLite with INT8 quantization (~50 KB)
3. **Deploy** to STM32H743ZI2 using TensorFlow Lite Micro

## ✨ Key Features

- Complete PPO implementation in TensorFlow
- Actor-Critic architecture for bipedal locomotion
- TFLite quantization (4x size reduction, <2% accuracy loss)
- STM32 deployment with TFLite Micro
- Real-time inference (~10-15 ms per step)
- UART communication for testing


## 🏗️ Architecture

```
Training:     Gymnasium → PPO Agent → Trained Model (.h5)
Conversion:   .h5 → TFLite → Quantization → .tflite (50KB)
Deployment:   .tflite → C Header → STM32 + TFLite Micro → Inference
                                            ↓
                                   UART (Obs ↔ Actions)
```


## 📁 Project Structure

```
BipedalWalker-PPO-STM32/
├── ppo_agent.py                 # PPO algorithm
├── actor_critic_networks.py     # Neural network models
├── train_ppo.py                 # Training script
├── test_ppo.py                  # Test with visualization
├── convert_to_tflite.py         # Model conversion
├── test_tflite_model.py         # Validate TFLite model
├── tflite_to_header.py          # Generate C header
├── test_stm32_serial.py         # Serial communication test
├── requirements.txt
├── README.md
└── stm32/                       # STM32 project files
    ├── main.c                   # STM32 HAL (CubeMX managed)
    ├── main.h                   # STM32 HAL headers
    ├── main2.cpp                # TFLite Micro inference
    ├── main2.h                  # C++ interface
    └── model.h                  # TFLite model data
```
---
## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/BipedalWalker-PPO-STM32.git
cd BipedalWalker-PPO-STM32
```

**Create Virtual Environment**

Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```

Linux/Mac:
```bash
python -m venv venv
source venv/bin/activate
```

**Install Dependencies**

```bash
pip install -r requirements.txt
```

### Training

```bash
python train_ppo.py
```

Trains for 5000 episodes (~6-8 hours), saves checkpoints every 200 episodes to `./save/model_actor.weights.h5`

### Convert to TFLite

```bash
python convert_to_tflite.py
```

Outputs `actor_model_quantized.tflite` (~50 KB) in `tflite_models/` directory

### Test TFLite Model

```bash
python test_tflite_model.py
```

Validates quantized model maintains performance

### Generate C Header for STM32

```bash
python tflite_to_header.py tflite_models/actor_model_quantized.tflite model.h
```

Creates `model.h` for STM32 project


## Testing Without Hardware

```bash
python test_trained_model.py
```

Visualizes robot walking for 3 episodes with real-time stats

---

## STM32 Deployment

### Hardware Used

**Board:** STM32 Nucleo-H743ZI2
- MCU: STM32H743ZI2
- Flash: 2 MB (used ~152 KB)
- RAM: 512KB SRAM (used ~45 KB)
- Clock: 480 MHz

*Note: You can use different STM32 boards - adjust pins and peripherals accordingly.*


## Firmware Setup

### Build TensorFlow Lite Micro Library

This project uses TFLite Micro library files. The commands below help create a folder containing all required library files for the STM32 project.

```bash
# Clone TFLite Micro (from firmware directory)
git clone https://github.com/tensorflow/tflite-micro.git --depth 1
cd tflite-micro

# Build for Cortex-M4 (~10-15 minutes)
make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=cortex_m_generic \
  TARGET_ARCH=cortex-m7 \
  microlite

# Generate library tree
python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
  ../Core/tflm_tree
```
This creates a `tflm_tree/` folder which contains all library files for TFLite Micro. Place this folder in your STM32 project directory beside Core and Drivers folders.

**Note:** Sometimes the above process doesn't add all required files. 
While building the project in STM32CubeIDE, you may encounter errors 
like missing `.h` or `.cc` files. In that case, manually copy the 
missing files from https://github.com/tensorflow/tflite-micro to your 
`tflm_tree/` folder.

### Add TFLite Micro Include Paths

**Project Properties → C/C++ Build → Settings → MCU G++ Compiler → Include paths**

Add these paths:
```
../tflm_tree
../tflm_tree/tensorflow
../tflm_tree/tensorflow/lite
../tflm_tree/tensorflow/lite/micro
../tflm_tree/tensorflow/lite/micro/kernels
../tflm_tree/third_party/flatbuffers/include
../tflm_tree/third_party/gemmlowp
../tflm_tree/third_party/kissfft
../tflm_tree/third_party/ruy
```

*See firmware source files for complete TFLite Micro implementation.*


### STM32CubeIDE Project Setup

**Configuration used (adjust for your board):**
- System Clock: 480 MHz
- UART3: 115200 baud for serial communication
- Convert project to C++ (right-click `.ioc` → Convert to C++)

*Adjust clock speed, UART pins, and peripherals based on your specific board.*


### Working with C++ in STM32 (Mixed C/C++ Project)

**Why separate C and C++ files?**

TensorFlow Lite Micro is written in C++, so we need C++ support. However, STM32CubeMX regenerates `main.c` when you modify peripherals, overwriting custom code. Solution: Keep TFLite code in separate C++ files (`main2.cpp`) and use `extern "C"` to bridge C and C++ code.

#### Template: main2.h (C++ Header with C Interface)

```cpp
#ifndef MY_MODULE_H_
#define MY_MODULE_H_

#ifdef __cplusplus
extern "C" {
#endif

// Functions callable from C code
void MyModule_Init(void);
void MyModule_Process(int value);
int MyModule_GetResult(void);

#ifdef __cplusplus
}
#endif

#endif // MY_MODULE_H_
```

#### Template: main2.cpp (C++ Implementation)

```cpp
extern "C" {
#include "main.h"  // Include STM32 HAL headers as C
}

#include "main2.h"

// C++ implementation
extern "C" void MyModule_Init(void) {
    // Your C++ code here
    // Can use C++ features (classes, templates, etc.)
}

extern "C" void MyModule_Process(int value) {
    // Implementation
}

extern "C" int MyModule_GetResult(void) {
    return 42;
}
```

#### Template: main.c (Calling C++ from C)

```c
/* USER CODE BEGIN Includes */
#include "main2.h"
/* USER CODE END Includes */

int main(void)
{
  // HAL initialization...
  
  /* USER CODE BEGIN 2 */
  MyModule_Init();
  /* USER CODE END 2 */

  while (1)
  {
    /* USER CODE BEGIN 3 */
    MyModule_Process(100);
    int result = MyModule_GetResult();
    /* USER CODE END 3 */
  }
}
```

**Why extern "C"?**

C++ uses name mangling (e.g., `MyModule_Init` becomes `_Z13MyModule_Initv`), making it incompatible with C. `extern "C"` tells the C++ compiler to use C-style naming so C code can call these functions.

---

### Build and Flash

Build project in STM32CubeIDE (Ctrl+B), then flash to board via Debug/Run.


### Test via Serial

**Identify port:** Windows: Device Manager (COMx) | Linux: `/dev/ttyACM0`

```bash
python test_stm32_serial.py
```

Python sends observations → STM32 runs inference → returns actions

---

## Troubleshooting

**`No module named 'Box2D'`**
```bash
pip install Box2D
```

**Serial port not found**  
Windows: Use COM3/COM4, Linux: Use /dev/ttyACM0

**STM32 build errors**  
Verify all TFLite Micro include paths are added

---

## Blog Post

Read the detailed article on Medium: **[Coming Soon]**

---

## Contributing

Contributions welcome! Open issues or submit pull requests.

---

## Acknowledgments

- [OpenAI Gymnasium](https://gymnasium.farama.org/) - BipedalWalker environment
- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers) - MCU deployment
- [STMicroelectronics](https://www.st.com/) - STM32 development tools

---

## Contact

**Your Name**  
[LinkedIn](https://www.linkedin.com/in/aditya-bangde-372447178/) | [Medium](https://medium.com/@adityabangde)

---

**Star this repo if you find it helpful!**
