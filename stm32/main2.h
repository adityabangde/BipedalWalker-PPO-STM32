#ifndef POLICY_H_
#define POLICY_H_

#ifdef __cplusplus
extern "C" {
#endif

// Initialize TensorFlow Lite Micro and the policy network
void Policy_Init(void);

// One policy step:
// - Receive observation over UART
// - Run inference
// - Send action over UART
void Policy_Step(void);

#ifdef __cplusplus
}
#endif

#endif // POLICY_H_
