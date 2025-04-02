#include "kernel_operator.h"
#include <type_traits>            

using namespace AscendC;     
constexpr int32_t BUFFER_NUM = 2;    
template<typename T> struct Map {using type = T;};      
template<> struct Map<int8_t> {using type = half;};             
         

template<typename TYPE_VAR, typename TYPE_INDICES, typename TYPE_UPDATES> class KernelScatterElements_Broadcast {
    using T = TYPE_VAR;    
public:
    __aicore__ inline KernelScatterElements_Broadcast() {}     
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates,  
                                uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain) {    
        
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);  
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0); 
 
        auto startPointer = core_size * GetBlockIdx(); 
        auto bufferlength = this->blockLength; 
  
        Gm_var.SetGlobalBuffer((__gm__ TYPE_VAR*)var, bufferlength);
        Gm_indices.SetGlobalBuffer((__gm__ TYPE_INDICES*)indices, bufferlength);
        Gm_updates.SetGlobalBuffer((__gm__ TYPE_UPDATES*)updates, bufferlength); 
    }    
    __aicore__ inline void Process(uint32_t TensorInfo[3 * 10], int signal, int axis) {   
    
        int Max_Dimension = 0; 
        for(int i = 0; i < 3; i++) { 
            if(TensorInfo[i * 10 + 0] > Max_Dimension) {  
                Max_Dimension = TensorInfo[i * 10 + 0]; 
            }   
        }    
        if (Max_Dimension == 1) {   
            if (signal == 0) {         
                for (int i = 0; i < TensorInfo[11]; i++) {
                    if constexpr (std::is_same_v < TYPE_VAR, uint8_t>) { 
                        int indices_element = static_cast<int>(Gm_indices(i)); 
                        int updates_element = static_cast<int>(Gm_updates(i));   
                        int var_element = static_cast<int>(Gm_var(indices_element));  

                        updates_element = updates_element + var_element;
                        Gm_var(indices_element) = static_cast<uint8_t>(updates_element);
                    }
                    else { 
                        int indices_element = static_cast<int>(Gm_indices(i)); 
                        float updates_element = static_cast<float>(Gm_updates(i));   
                        float var_element = static_cast<float>(Gm_var(indices_element));

                        updates_element = updates_element + var_element;
                        Gm_var(indices_element) = static_cast<TYPE_VAR>(updates_element);
                    }             
                }
            }
            else if (signal == 1) {    
                for (int i = 0; i < TensorInfo[11]; i++) {
                    if constexpr (std::is_same_v < TYPE_VAR, uint8_t>) {
                        int indices_element = static_cast<int>(Gm_indices(i)); 
                        int updates_element = static_cast<int>(Gm_updates(i));     
                        Gm_var(indices_element) = static_cast<uint8_t>(updates_element);
                    }
                    else {
                        int indices_element = static_cast<int>(Gm_indices(i)); 
                        float updates_element = static_cast<float>(Gm_updates(i));   
                        Gm_var(indices_element) = static_cast<TYPE_VAR>(updates_element);
                    }             
                }
            }
        }
        else if (Max_Dimension == 2) {  
            if (axis == 0) {
                if (signal == 0) {     
                    for (int i = 0; i < TensorInfo[11]; i++) {           
                        for (int j = 0; j < TensorInfo[12]; j++) {   
                            if constexpr (std::is_same_v < TYPE_VAR, uint8_t>) { 
                                int indices_element = static_cast<int>(Gm_indices(i * TensorInfo[12] + j)); 
                                int updates_element = static_cast<int>(Gm_updates(i * TensorInfo[12] + j));   
                                int var_element = static_cast<int>(Gm_var(indices_element * TensorInfo[2] + j));  

                                updates_element = updates_element + var_element;
                                Gm_var(indices_element * TensorInfo[2] + j) = static_cast<uint8_t>(updates_element);
                            }
                            else { 
                                int indices_element = static_cast<int>(Gm_indices(i * TensorInfo[12] + j)); 
                                float updates_element = static_cast<float>(Gm_updates(i * TensorInfo[12] + j));   
                                float var_element = static_cast<float>(Gm_var(indices_element * TensorInfo[2] + j));

                                updates_element = updates_element + var_element;
                                Gm_var(indices_element * TensorInfo[2] + j) = static_cast<TYPE_VAR>(updates_element);
                            }         
                        }   
                    }  
                }
                else if (signal == 1) {   
                    for (int i = 0; i < TensorInfo[11]; i++) {           
                        for (int j = 0; j < TensorInfo[12]; j++) {   
                            if constexpr (std::is_same_v < TYPE_VAR, uint8_t>) { 
                                int indices_element = static_cast<int>(Gm_indices(i * TensorInfo[12] + j)); 
                                int updates_element = static_cast<int>(Gm_updates(i * TensorInfo[12] + j));   
                                Gm_var(indices_element * TensorInfo[2] + j) = static_cast<uint8_t>(updates_element);
                            }
                            else { 
                                int indices_element = static_cast<int>(Gm_indices(i * TensorInfo[12] + j)); 
                                float updates_element = static_cast<float>(Gm_updates(i * TensorInfo[12] + j));   
                                Gm_var(indices_element * TensorInfo[2] + j) = static_cast<TYPE_VAR>(updates_element);
                            }         
                        }   
                    }  
                }
            }
            if (axis == 1){
                if (signal == 0) {     
                    for (int i = 0; i < TensorInfo[11]; i++) {           
                        for (int j = 0; j < TensorInfo[12]; j++) {   
                            if constexpr (std::is_same_v < TYPE_VAR, uint8_t>) { 
                                int indices_element = static_cast<int>(Gm_indices(i * TensorInfo[12] + j)); 
                                int updates_element = static_cast<int>(Gm_updates(i * TensorInfo[12] + j));   
                                int var_element = static_cast<int>(Gm_var(i * TensorInfo[2] + indices_element));  

                                updates_element = updates_element + var_element;
                                Gm_var(i * TensorInfo[2] + indices_element) = static_cast<uint8_t>(updates_element);
                            }
                            else { 
                                int indices_element = static_cast<int>(Gm_indices(i * TensorInfo[12] + j)); 
                                float updates_element = static_cast<float>(Gm_updates(i * TensorInfo[12] + j));   
                                float var_element = static_cast<float>(Gm_var(i * TensorInfo[2] + indices_element));

                                updates_element = updates_element + var_element;
                                Gm_var(i * TensorInfo[2] + indices_element) = static_cast<TYPE_VAR>(updates_element);
                            }         
                        }   
                    }  
                }
                else if (signal == 1) {   
                    for (int i = 0; i < TensorInfo[11]; i++) {           
                        for (int j = 0; j < TensorInfo[12]; j++) {   
                            if constexpr (std::is_same_v < TYPE_VAR, uint8_t>) { 
                                int indices_element = static_cast<int>(Gm_indices(i * TensorInfo[12] + j)); 
                                int updates_element = static_cast<int>(Gm_updates(i * TensorInfo[12] + j));   
                                Gm_var(i * TensorInfo[2] + indices_element) = static_cast<uint8_t>(updates_element);
                            }
                            else { 
                                int indices_element = static_cast<int>(Gm_indices(i * TensorInfo[12] + j)); 
                                float updates_element = static_cast<float>(Gm_updates(i * TensorInfo[12] + j));   
                                Gm_var(i * TensorInfo[2] + indices_element) = static_cast<TYPE_VAR>(updates_element);
                            }         
                        }   
                    }  
                }
            }
        }     
        else if(Max_Dimension==3){    
            if (axis == 0) {
                for (int i = 0; i < TensorInfo[11]; i++) {   
                    for (int j = 0; j < TensorInfo[12]; j++) {   
                        for (int k = 0; k < TensorInfo[13]; k++) {          
                            if constexpr (std::is_same_v < TYPE_VAR, uint8_t>) { 
                                int indices_element = static_cast<int>(Gm_indices(i * (TensorInfo[12] * TensorInfo[13]) + j * TensorInfo[13] + k)); 
                                int updates_element = static_cast<int>(Gm_updates(i * (TensorInfo[12] * TensorInfo[13]) + j * TensorInfo[13] + k));   
                                int var_element = static_cast<int>(Gm_var(indices_element * (TensorInfo[2] * TensorInfo[3]) + j * TensorInfo[3] + k));   
                                if (signal == 0) {
                                    updates_element = updates_element + var_element; 
                                    Gm_var(indices_element * (TensorInfo[2] * TensorInfo[3]) + j * TensorInfo[3] + k) = static_cast<uint8_t>(updates_element);
                                }
                                else if (signal == 1){
                                    Gm_var(indices_element * (TensorInfo[2] * TensorInfo[3]) + j * TensorInfo[3] + k) = static_cast<uint8_t>(updates_element);
                                }  
                            }
                            else { 
                                int indices_element = static_cast<int>(Gm_indices(i * (TensorInfo[12] * TensorInfo[13]) + j * TensorInfo[13] + k)); 
                                float updates_element = static_cast<float>(Gm_updates(i * (TensorInfo[12] * TensorInfo[13]) + j * TensorInfo[13] + k));   
                                float var_element = static_cast<float>(Gm_var(indices_element * (TensorInfo[2] * TensorInfo[3]) + j * TensorInfo[3] + k));   
                                if (signal == 0) {
                                    updates_element = updates_element + var_element; 
                                    Gm_var(indices_element * (TensorInfo[2] * TensorInfo[3]) + j * TensorInfo[3] + k) = static_cast<TYPE_VAR>(updates_element);
                                }
                                else if (signal == 1){
                                    Gm_var(indices_element * (TensorInfo[2] * TensorInfo[3]) + j * TensorInfo[3] + k) = static_cast<TYPE_VAR>(updates_element);
                                }  
                            }            
                        }   
                    }   
                }             
            }
            else if (axis == 1) {
                for (int i = 0; i < TensorInfo[11]; i++) {            
                    for (int j = 0; j < TensorInfo[12]; j++) {   
                        for (int k = 0; k < TensorInfo[13]; k++) {          
                            if constexpr (std::is_same_v < TYPE_VAR, uint8_t>) { 
                                int indices_element = static_cast<int>(Gm_indices(i * (TensorInfo[12] * TensorInfo[13]) + j * TensorInfo[13] + k)); 
                                int updates_element = static_cast<int>(Gm_updates(i * (TensorInfo[12] * TensorInfo[13]) + j * TensorInfo[13] + k));   
                                int var_element = static_cast<int>(Gm_var(i * (TensorInfo[2] * TensorInfo[3]) + indices_element * TensorInfo[3] + k));   
                                if (signal == 0) {
                                    updates_element = updates_element + var_element; 
                                    Gm_var(i * (TensorInfo[2] * TensorInfo[3]) + indices_element * TensorInfo[3] + k) = static_cast<uint8_t>(updates_element);
                                }
                                else if (signal == 1){
                                    Gm_var(i * (TensorInfo[2] * TensorInfo[3]) + indices_element * TensorInfo[3] + k) = static_cast<uint8_t>(updates_element);
                                }  
                            }
                            else { 
                                int indices_element = static_cast<int>(Gm_indices(i * (TensorInfo[12] * TensorInfo[13]) + j * TensorInfo[13] + k)); 
                                float updates_element = static_cast<float>(Gm_updates(i * (TensorInfo[12] * TensorInfo[13]) + j * TensorInfo[13] + k));   
                                float var_element = static_cast<float>(Gm_var(i * (TensorInfo[2] * TensorInfo[3]) + indices_element * TensorInfo[3] + k));   
                                if (signal == 0) {
                                    updates_element = updates_element + var_element; 
                                    Gm_var(i * (TensorInfo[2] * TensorInfo[3]) + indices_element * TensorInfo[3] + k) = static_cast<TYPE_VAR>(updates_element);
                                }
                                else if (signal == 1){
                                    Gm_var(i * (TensorInfo[2] * TensorInfo[3]) + indices_element * TensorInfo[3] + k) = static_cast<TYPE_VAR>(updates_element);
                                }  
                            }            
                        }   
                    }   
                }             
            }
            else if (axis == 2){
                for (int i = 0; i < TensorInfo[11]; i++) {            
                    for (int j = 0; j < TensorInfo[12]; j++) {   
                        for (int k = 0; k < TensorInfo[13]; k++) {          
                            if constexpr (std::is_same_v < TYPE_VAR, uint8_t>) { 
                                int indices_element = static_cast<int>(Gm_indices(i * (TensorInfo[12] * TensorInfo[13]) + j * TensorInfo[13] + k)); 
                                int updates_element = static_cast<int>(Gm_updates(i * (TensorInfo[12] * TensorInfo[13]) + j * TensorInfo[13] + k));   
                                int var_element = static_cast<int>(Gm_var(i * (TensorInfo[2] * TensorInfo[3]) + j * TensorInfo[3] + indices_element));   
                                if (signal == 0) {
                                    updates_element = updates_element + var_element; 
                                    Gm_var(i * (TensorInfo[2] * TensorInfo[3]) + j * TensorInfo[3] + indices_element) = static_cast<uint8_t>(updates_element);
                                }
                                else if (signal == 1){
                                    Gm_var(i * (TensorInfo[2] * TensorInfo[3]) + j * TensorInfo[3] + indices_element) = static_cast<uint8_t>(updates_element);
                                }  
                            }
                            else { 
                                int indices_element = static_cast<int>(Gm_indices(i * (TensorInfo[12] * TensorInfo[13]) + j * TensorInfo[13] + k)); 
                                float updates_element = static_cast<float>(Gm_updates(i * (TensorInfo[12] * TensorInfo[13]) + j * TensorInfo[13] + k));   
                                float var_element = static_cast<float>(Gm_var(i * (TensorInfo[2] * TensorInfo[3]) + j * TensorInfo[3] + indices_element));   
                                if (signal == 0) {
                                    updates_element = updates_element + var_element; 
                                    Gm_var(i * (TensorInfo[2] * TensorInfo[3]) + j * TensorInfo[3] + indices_element) = static_cast<TYPE_VAR>(updates_element);
                                }
                                else if (signal == 1){
                                    Gm_var(i * (TensorInfo[2] * TensorInfo[3]) + j * TensorInfo[3] + indices_element) = static_cast<TYPE_VAR>(updates_element);
                                }  
                            }            
                        }   
                    }   
                }           
            }  
        }   
        else if (Max_Dimension == 4) {    
            if (axis == 0) {
                for (int i = 0; i < TensorInfo[11]; i++) { 
                    for (int j = 0; j < TensorInfo[12]; j++) {           
                        for (int k = 0; k < TensorInfo[13]; k++) {   
                            for (int n = 0; n < TensorInfo[14]; n++) {          
                                if constexpr (std::is_same_v < TYPE_VAR, uint8_t>) { 
                                    int indices_element = static_cast<int>(Gm_indices(i * (TensorInfo[12] * TensorInfo[13] * TensorInfo[14]) 
                                                                                + j * (TensorInfo[13] * TensorInfo[14])
                                                                                + k * TensorInfo[14] 
                                                                                + n)); 
                                    int updates_element = static_cast<int>(Gm_updates(i * (TensorInfo[12] * TensorInfo[13] * TensorInfo[14]) 
                                                                                + j * (TensorInfo[13] * TensorInfo[14])
                                                                                + k * TensorInfo[14] 
                                                                                + n));   
                                    int var_element = static_cast<int>(Gm_var(indices_element * (TensorInfo[2] * TensorInfo[3] * TensorInfo[4]) 
                                                                                + j * (TensorInfo[3] * TensorInfo[4])
                                                                                + k * TensorInfo[4] 
                                                                                + n));   
                                    if (signal == 0) {
                                        updates_element = updates_element + var_element; 
                                        Gm_var(indices_element * (TensorInfo[2] * TensorInfo[3] * TensorInfo[4]) 
                                                                                + j * (TensorInfo[3] * TensorInfo[4])
                                                                                + k * TensorInfo[4] 
                                                                                + n) = static_cast<uint8_t>(updates_element);
                                    }
                                    else if (signal == 1){
                                        Gm_var(indices_element * (TensorInfo[2] * TensorInfo[3] * TensorInfo[4]) 
                                                                                + j * (TensorInfo[3] * TensorInfo[4])
                                                                                + k * TensorInfo[4] 
                                                                                + n) = static_cast<uint8_t>(updates_element);
                                    }  
                                }
                                else { 
                                    int indices_element = static_cast<int>(Gm_indices(i * (TensorInfo[12] * TensorInfo[13] * TensorInfo[14]) 
                                                                                + j * (TensorInfo[13] * TensorInfo[14])
                                                                                + k * TensorInfo[14] 
                                                                                + n)); 
                                    float updates_element = static_cast<float>(Gm_updates(i * (TensorInfo[12] * TensorInfo[13] * TensorInfo[14]) 
                                                                                + j * (TensorInfo[13] * TensorInfo[14])
                                                                                + k * TensorInfo[14] 
                                                                                + n));   
                                    float var_element = static_cast<float>(Gm_var(indices_element * (TensorInfo[2] * TensorInfo[3] * TensorInfo[4]) 
                                                                                + j * (TensorInfo[3] * TensorInfo[4])
                                                                                + k * TensorInfo[4] 
                                                                                + n));   
                                    if (signal == 0) {
                                        updates_element = updates_element + var_element; 
                                        Gm_var(indices_element * (TensorInfo[2] * TensorInfo[3] * TensorInfo[4]) 
                                                                                + j * (TensorInfo[3] * TensorInfo[4])
                                                                                + k * TensorInfo[4] 
                                                                                + n) = static_cast<TYPE_VAR>(updates_element);
                                    }
                                    else if (signal == 1){
                                        Gm_var(indices_element * (TensorInfo[2] * TensorInfo[3] * TensorInfo[4]) 
                                                                                + j * (TensorInfo[3] * TensorInfo[4])
                                                                                + k * TensorInfo[4] 
                                                                                + n) = static_cast<TYPE_VAR>(updates_element);
                                    }  
                                }          
                            }   
                        }    
                    }   
                }     
            }
            else if (axis == 1) {
                for (int i = 0; i < TensorInfo[11]; i++) { 
                    for (int j = 0; j < TensorInfo[12]; j++) {           
                        for (int k = 0; k < TensorInfo[13]; k++) {   
                            for (int n = 0; n < TensorInfo[14]; n++) {          
                                if constexpr (std::is_same_v < TYPE_VAR, uint8_t>) { 
                                    int indices_element = static_cast<int>(Gm_indices(i * (TensorInfo[12] * TensorInfo[13] * TensorInfo[14]) 
                                                                                + j * (TensorInfo[13] * TensorInfo[14])
                                                                                + k * TensorInfo[14] 
                                                                                + n)); 
                                    int updates_element = static_cast<int>(Gm_updates(i * (TensorInfo[12] * TensorInfo[13] * TensorInfo[14]) 
                                                                                + j * (TensorInfo[13] * TensorInfo[14])
                                                                                + k * TensorInfo[14] 
                                                                                + n));   
                                    int var_element = static_cast<int>(Gm_var(i * (TensorInfo[2] * TensorInfo[3] * TensorInfo[4]) 
                                                                                + indices_element * (TensorInfo[3] * TensorInfo[4])
                                                                                + k * TensorInfo[4] 
                                                                                + n));   
                                    if (signal == 0) {
                                        updates_element = updates_element + var_element; 
                                        Gm_var(i * (TensorInfo[2] * TensorInfo[3] * TensorInfo[4]) 
                                                                                + indices_element * (TensorInfo[3] * TensorInfo[4])
                                                                                + k * TensorInfo[4] 
                                                                                + n) = static_cast<uint8_t>(updates_element);
                                    }
                                    else if (signal == 1){
                                        Gm_var(i * (TensorInfo[2] * TensorInfo[3] * TensorInfo[4]) 
                                                                                + indices_element * (TensorInfo[3] * TensorInfo[4])
                                                                                + k * TensorInfo[4] 
                                                                                + n) = static_cast<uint8_t>(updates_element);
                                    }  
                                }
                                else { 
                                    int indices_element = static_cast<int>(Gm_indices(i * (TensorInfo[12] * TensorInfo[13] * TensorInfo[14]) 
                                                                                + j * (TensorInfo[13] * TensorInfo[14])
                                                                                + k * TensorInfo[14] 
                                                                                + n)); 
                                    float updates_element = static_cast<float>(Gm_updates(i * (TensorInfo[12] * TensorInfo[13] * TensorInfo[14]) 
                                                                                + j * (TensorInfo[13] * TensorInfo[14])
                                                                                + k * TensorInfo[14] 
                                                                                + n));   
                                    float var_element = static_cast<float>(Gm_var(i * (TensorInfo[2] * TensorInfo[3] * TensorInfo[4]) 
                                                                                + indices_element * (TensorInfo[3] * TensorInfo[4])
                                                                                + k * TensorInfo[4] 
                                                                                + n));   
                                    if (signal == 0) {
                                        updates_element = updates_element + var_element; 
                                        Gm_var(i * (TensorInfo[2] * TensorInfo[3] * TensorInfo[4]) 
                                                                                + indices_element * (TensorInfo[3] * TensorInfo[4])
                                                                                + k * TensorInfo[4] 
                                                                                + n) = static_cast<TYPE_VAR>(updates_element);
                                    }
                                    else if (signal == 1){
                                        Gm_var(i * (TensorInfo[2] * TensorInfo[3] * TensorInfo[4]) 
                                                                                + indices_element * (TensorInfo[3] * TensorInfo[4])
                                                                                + k * TensorInfo[4] 
                                                                                + n) = static_cast<TYPE_VAR>(updates_element);
                                    }  
                                }          
                            }   
                        }    
                    }   
                }     
            }
            else if (axis == 2) {
                for (int i = 0; i < TensorInfo[11]; i++) { 
                    for (int j = 0; j < TensorInfo[12]; j++) {           
                        for (int k = 0; k < TensorInfo[13]; k++) {   
                            for (int n = 0; n < TensorInfo[14]; n++) {          
                                if constexpr (std::is_same_v < TYPE_VAR, uint8_t>) { 
                                    int indices_element = static_cast<int>(Gm_indices(i * (TensorInfo[12] * TensorInfo[13] * TensorInfo[14]) 
                                                                                + j * (TensorInfo[13] * TensorInfo[14])
                                                                                + k * TensorInfo[14] 
                                                                                + n)); 
                                    int updates_element = static_cast<int>(Gm_updates(i * (TensorInfo[12] * TensorInfo[13] * TensorInfo[14]) 
                                                                                + j * (TensorInfo[13] * TensorInfo[14])
                                                                                + k * TensorInfo[14] 
                                                                                + n));   
                                    int var_element = static_cast<int>(Gm_var(i * (TensorInfo[2] * TensorInfo[3] * TensorInfo[4]) 
                                                                                + j * (TensorInfo[3] * TensorInfo[4])
                                                                                + indices_element * TensorInfo[4] 
                                                                                + n));   
                                    if (signal == 0) {
                                        updates_element = updates_element + var_element; 
                                        Gm_var(i * (TensorInfo[2] * TensorInfo[3] * TensorInfo[4]) 
                                                                                + j * (TensorInfo[3] * TensorInfo[4])
                                                                                + indices_element * TensorInfo[4] 
                                                                                + n) = static_cast<uint8_t>(updates_element);
                                    }
                                    else if (signal == 1){
                                        Gm_var(i * (TensorInfo[2] * TensorInfo[3] * TensorInfo[4]) 
                                                                                + j * (TensorInfo[3] * TensorInfo[4])
                                                                                + indices_element * TensorInfo[4] 
                                                                                + n) = static_cast<uint8_t>(updates_element);
                                    }  
                                }
                                else { 
                                    int indices_element = static_cast<int>(Gm_indices(i * (TensorInfo[12] * TensorInfo[13] * TensorInfo[14]) 
                                                                                + j * (TensorInfo[13] * TensorInfo[14])
                                                                                + k * TensorInfo[14] 
                                                                                + n)); 
                                    float updates_element = static_cast<float>(Gm_updates(i * (TensorInfo[12] * TensorInfo[13] * TensorInfo[14]) 
                                                                                + j * (TensorInfo[13] * TensorInfo[14])
                                                                                + k * TensorInfo[14] 
                                                                                + n));   
                                    float var_element = static_cast<float>(Gm_var(i * (TensorInfo[2] * TensorInfo[3] * TensorInfo[4]) 
                                                                                + j * (TensorInfo[3] * TensorInfo[4])
                                                                                + indices_element * TensorInfo[4] 
                                                                                + n));   
                                    if (signal == 0) {
                                        updates_element = updates_element + var_element; 
                                        Gm_var(i * (TensorInfo[2] * TensorInfo[3] * TensorInfo[4]) 
                                                                                + j * (TensorInfo[3] * TensorInfo[4])
                                                                                + indices_element * TensorInfo[4] 
                                                                                + n) = static_cast<TYPE_VAR>(updates_element);
                                    }
                                    else if (signal == 1){
                                        Gm_var(i * (TensorInfo[2] * TensorInfo[3] * TensorInfo[4]) 
                                                                                + j * (TensorInfo[3] * TensorInfo[4])
                                                                                + indices_element * TensorInfo[4] 
                                                                                + n) = static_cast<TYPE_VAR>(updates_element);
                                    }  
                                }          
                            }   
                        }    
                    }   
                }     
            }
            else if (axis == 3) {
                for (int i = 0; i < TensorInfo[11]; i++) { 
                    for (int j = 0; j < TensorInfo[12]; j++) {           
                        for (int k = 0; k < TensorInfo[13]; k++) {   
                            for (int n = 0; n < TensorInfo[14]; n++) {          
                                if constexpr (std::is_same_v < TYPE_VAR, uint8_t>) { 
                                    int indices_element = static_cast<int>(Gm_indices(i * (TensorInfo[12] * TensorInfo[13] * TensorInfo[14]) 
                                                                                + j * (TensorInfo[13] * TensorInfo[14])
                                                                                + k * TensorInfo[14] 
                                                                                + n)); 
                                    int updates_element = static_cast<int>(Gm_updates(i * (TensorInfo[12] * TensorInfo[13] * TensorInfo[14]) 
                                                                                + j * (TensorInfo[13] * TensorInfo[14])
                                                                                + k * TensorInfo[14] 
                                                                                + n));   
                                    int var_element = static_cast<int>(Gm_var(i * (TensorInfo[2] * TensorInfo[3] * TensorInfo[4]) 
                                                                                + j * (TensorInfo[3] * TensorInfo[4])
                                                                                + k * TensorInfo[4] 
                                                                                + indices_element));   
                                    if (signal == 0) {
                                        updates_element = updates_element + var_element; 
                                        Gm_var(i * (TensorInfo[2] * TensorInfo[3] * TensorInfo[4]) 
                                                                                + j * (TensorInfo[3] * TensorInfo[4])
                                                                                + k * TensorInfo[4] 
                                                                                + indices_element) = static_cast<uint8_t>(updates_element);
                                    }
                                    else if (signal == 1){
                                        Gm_var(i * (TensorInfo[2] * TensorInfo[3] * TensorInfo[4]) 
                                                                                + j * (TensorInfo[3] * TensorInfo[4])
                                                                                + k * TensorInfo[4] 
                                                                                + indices_element) = static_cast<uint8_t>(updates_element);
                                    }  
                                }
                                else { 
                                    int indices_element = static_cast<int>(Gm_indices(i * (TensorInfo[12] * TensorInfo[13] * TensorInfo[14]) 
                                                                                + j * (TensorInfo[13] * TensorInfo[14])
                                                                                + k * TensorInfo[14] 
                                                                                + n)); 
                                    float updates_element = static_cast<float>(Gm_updates(i * (TensorInfo[12] * TensorInfo[13] * TensorInfo[14]) 
                                                                                + j * (TensorInfo[13] * TensorInfo[14])
                                                                                + k * TensorInfo[14] 
                                                                                + n));   
                                    float var_element = static_cast<float>(Gm_var(i * (TensorInfo[2] * TensorInfo[3] * TensorInfo[4]) 
                                                                                + j * (TensorInfo[3] * TensorInfo[4])
                                                                                + k * TensorInfo[4] 
                                                                                + indices_element));   
                                    if (signal == 0) {
                                        updates_element = updates_element + var_element; 
                                        Gm_var(i * (TensorInfo[2] * TensorInfo[3] * TensorInfo[4]) 
                                                                                + j * (TensorInfo[3] * TensorInfo[4])
                                                                                + k * TensorInfo[4] 
                                                                                + indices_element) = static_cast<TYPE_VAR>(updates_element);
                                    }
                                    else if (signal == 1){
                                        Gm_var(i * (TensorInfo[2] * TensorInfo[3] * TensorInfo[4]) 
                                                                                + j * (TensorInfo[3] * TensorInfo[4])
                                                                                + k * TensorInfo[4] 
                                                                                + indices_element) = static_cast<TYPE_VAR>(updates_element);
                                    }  
                                }          
                            }   
                        }    
                    }   
                }     
            }  
        }
    }   
private:
    TPipe pipe;    
    GlobalTensor<TYPE_VAR> Gm_var;     
    GlobalTensor<TYPE_INDICES> Gm_indices;  
    GlobalTensor<TYPE_UPDATES> Gm_updates;  
    uint32_t blockLength;   
};
    
extern "C" __global__ __aicore__ void scatter_elements(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);    

    KernelScatterElements_Broadcast<DTYPE_VAR, DTYPE_INDICES, DTYPE_UPDATES> op;    
        op.Init(var, indices, updates, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);   
        op.Process(tiling_data.TensorInfo, tiling_data.signal, tiling_data.axis);    
}


 
