extern "C" {
#include "main.h"
#include <stdio.h>
#include <string.h>
}

#include "main2.h"
#include "model.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

extern UART_HandleTypeDef huart3;

#define OBS_DIM  24
#define ACT_DIM  4
#define TENSOR_ARENA_SIZE  (40 * 1024)

// Helper function for UART print
void uart_print(const char* format, ...) {
    char buffer[128];
    va_list args;
    va_start(args, format);
    int len = vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    HAL_UART_Transmit(&huart3, (uint8_t*)buffer, len, 100);
    HAL_UART_Transmit(&huart3, (uint8_t*)"\r\n", 2, 100);
}

namespace {
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input_tensor = nullptr;
    TfLiteTensor* output_tensor = nullptr;
    alignas(16) uint8_t tensor_arena[TENSOR_ARENA_SIZE];
    bool initialized = false;
}

extern "C" void Policy_Init(void)
{
    uart_print("=== Policy Init ===");

    // Load model
    model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        uart_print("Model version mismatch!");
        return;
    }

    // Register operations
    static tflite::MicroMutableOpResolver<2> resolver;
    resolver.AddFullyConnected();
    resolver.AddRelu();

    // Create interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;

    // Allocate tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        uart_print("Allocation failed!");
        return;
    }

    // Get tensors
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    initialized = true;
    uart_print("Init OK: [%d] -> [%d]",
               input_tensor->dims->data[1],
               output_tensor->dims->data[1]);
}

extern "C" void Policy_Step(void)
{
    if (!initialized) return;

    static float obs[OBS_DIM];
    static float act[ACT_DIM];

    // Receive observation
    if (HAL_UART_Receive(&huart3, (uint8_t*)obs, OBS_DIM*4, 5000) != HAL_OK) {
        return;
    }

    // Copy to input
    memcpy(input_tensor->data.f, obs, OBS_DIM * sizeof(float));

    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        uart_print("Invoke failed");
        return;
    }

    // Copy output
    memcpy(act, output_tensor->data.f, ACT_DIM * sizeof(float));

    // Clamp actions [-1, 1]
    for (int i = 0; i < ACT_DIM; i++) {
        if (act[i] > 1.0f) act[i] = 1.0f;
        if (act[i] < -1.0f) act[i] = -1.0f;
    }

    // Send action
    HAL_UART_Transmit(&huart3, (uint8_t*)act, ACT_DIM*4, 5000);
}
