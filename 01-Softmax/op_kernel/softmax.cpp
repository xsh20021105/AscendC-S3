#include "kernel_operator.h"
#include <type_traits>            

using namespace AscendC;     
constexpr int32_t BUFFER_NUM = 2;    
template<typename T> struct Map {using type = T;};      
template<> struct Map<int8_t> {using type = half;};             
         

template<typename TYPE_X, typename TYPE_Y> class KernelSoftmax_Broadcast {
    using T = TYPE_Y;    
public:
    __aicore__ inline KernelSoftmax_Broadcast() {}     
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,  
                                uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain) {    
        
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);  
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0); 
 
        auto startPointer = core_size * GetBlockIdx(); 
        auto bufferlength = this->blockLength; 
   
        Gm_x.SetGlobalBuffer((__gm__ TYPE_X*)x, bufferlength);
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y, bufferlength);   
    
        pipe.InitBuffer(tmp1Buffer, 1 * sizeof(DTYPE_Y));   
    }    
    __aicore__ inline void Process(uint32_t TensorInfo[1 * 10], int dim) {   
        if (dim < 0 ) dim = dim + TensorInfo[0];
        LocalTensor<float> tmp1 = tmp1Buffer.Get<float>();  
        float sum_exp = 0.0f;   
         
        int Max_Dimension = 0; 
        for(int i = 0; i < 1; i++){ 
            if(TensorInfo[i * 10 + 0] > Max_Dimension){  
                Max_Dimension = TensorInfo[i * 10 + 0]; 
            }   
        }      

        if (Max_Dimension == 1) {   
            float sum_exp = 0.0f; 
            for (int i = 0; i < TensorInfo[1]; i++) {     
                float x_element = static_cast<float>(Gm_x(i));   
                Duplicate<float>(tmp1, static_cast<TYPE_Y>(x_element), 1);    
                Exp(tmp1, tmp1, 1);
                sum_exp += tmp1(0);   
            } 
            for (int i = 0; i < TensorInfo[1]; i++) {     
                float x_element = static_cast<float>(Gm_x(i));   
                Duplicate<float>(tmp1, static_cast<TYPE_Y>(x_element), 1);    
                Exp(tmp1, tmp1, 1);
                tmp1(0) = tmp1(0) / sum_exp;      
                Gm_y(i) = static_cast<TYPE_Y>(tmp1(0));  
            }       
        }
        else if (Max_Dimension == 2) {
            if (dim == 0 ){
                float sum_exp;
                for (int j = 0; j < TensorInfo[2]; j++){     
                    sum_exp = 0.0f;
                    for (int i = 0; i < TensorInfo[1]; i++) {             
                        float x_element = static_cast<float>(Gm_x (i * TensorInfo[2] + j));    
                        Duplicate<float>(tmp1, static_cast<TYPE_Y>(x_element), 1);       
                        Exp(tmp1, tmp1, 1); 
                        sum_exp += tmp1(0);
                    } 
                    for (int i = 0; i < TensorInfo[1]; i++) {             
                        float x_element = static_cast<float>(Gm_x (i * TensorInfo[2] + j));    
                        Duplicate<float>(tmp1, static_cast<TYPE_Y>(x_element), 1);       
                        Exp(tmp1, tmp1, 1);     
                        tmp1(0) = tmp1(0) / sum_exp;
                        Gm_y(i * TensorInfo[2] + j) = static_cast<TYPE_Y>(tmp1(0));            
                    } 
                }
            }else if (dim == 1){
                float sum_exp;
                for (int i = 0; i < TensorInfo[1]; i++){     
                    sum_exp = 0.0f;
                    for (int j = 0; j < TensorInfo[2]; j++) {              
                        float x_element = static_cast<float>(Gm_x (i * TensorInfo[2] + j));   
                        Duplicate<float>(tmp1, static_cast<TYPE_Y>(x_element), 1);       
                        Exp(tmp1, tmp1, 1); 
                        sum_exp += tmp1(0); 
                    } 
                    for (int j = 0; j < TensorInfo[2]; j++) {             
                        float x_element = static_cast<float>(Gm_x (i * TensorInfo[2] + j));    
                        Duplicate<float>(tmp1, static_cast<TYPE_Y>(x_element), 1);       
                        Exp(tmp1, tmp1, 1);   
                        tmp1(0) = tmp1(0) / sum_exp;
                        Gm_y(i * TensorInfo[2] + j) = static_cast<TYPE_Y>(tmp1(0));               
                    } 
                }
            }
        }  
        else if(Max_Dimension == 3){
            if (dim == 0) { 
                float sum_exp;   
                for (int i = 0; i < TensorInfo[3]; i++) {            
                    for (int j = 0; j < TensorInfo[2]; j++) {   
                        sum_exp = 0.0f;
                        for (int k = 0; k < TensorInfo[1]; k++) {           
                            float x_element = static_cast<float>(Gm_x (k * (TensorInfo[2] * TensorInfo[3]) + j * TensorInfo[3] + i));  
                            Duplicate<float>(tmp1, static_cast<TYPE_Y>(x_element), 1);    
                            Exp(tmp1, tmp1, 1);
                            sum_exp += tmp1(0); 
                        } 
                        for (int k = 0; k < TensorInfo[1]; k++) {  
                            float x_element = static_cast<float>(Gm_x (k * (TensorInfo[2] * TensorInfo[3]) + j * TensorInfo[3] + i));     
                            Duplicate<float>(tmp1, static_cast<TYPE_Y>(x_element), 1);    
                            Exp(tmp1, tmp1, 1);
                            tmp1(0) = tmp1(0) / sum_exp;      
                            Gm_y(k * (TensorInfo[2] * TensorInfo[3]) + j * TensorInfo[3] + i) = static_cast<TYPE_Y>(tmp1(0));  
                        }     
                    }
                } 
            }else if (dim == 1){
                for (int i = 0; i < TensorInfo[3]; i++) {            
                    for (int k = 0; k < TensorInfo[1]; k++) {    
                        sum_exp = 0.0f;
                        for (int j = 0; j < TensorInfo[2]; j++) {           
                            float x_element = static_cast<float>(Gm_x (k * (TensorInfo[2] * TensorInfo[3]) + j * TensorInfo[3] + i));  
                            Duplicate<float>(tmp1, static_cast<TYPE_Y>(x_element), 1);    
                            Exp(tmp1, tmp1, 1);   
                            sum_exp += tmp1(0); 
                        } 
                        for (int j = 0; j < TensorInfo[2]; j++) {  
                            float x_element = static_cast<float>(Gm_x (k * (TensorInfo[2] * TensorInfo[3]) + j * TensorInfo[3] + i));     
                            Duplicate<float>(tmp1, static_cast<TYPE_Y>(x_element), 1);    
                            Exp(tmp1, tmp1, 1);
                            tmp1(0) = tmp1(0) / sum_exp;      
                            Gm_y(k * (TensorInfo[2] * TensorInfo[3]) + j * TensorInfo[3] + i) = static_cast<TYPE_Y>(tmp1(0));   
                        }     
                    }
                } 
            }else if (dim == 2){
                for (int j = 0; j < TensorInfo[2]; j++) {            
                    for (int k = 0; k < TensorInfo[1]; k++) {   
                        sum_exp = 0.0f;
                        for (int i = 0; i < TensorInfo[3]; i++) {          
                            float x_element = static_cast<float>(Gm_x (k * (TensorInfo[2] * TensorInfo[3]) + j * TensorInfo[3] + i));  
                            Duplicate<float>(tmp1, static_cast<TYPE_Y>(x_element), 1);    
                            Exp(tmp1, tmp1, 1);
                            sum_exp += tmp1(0); 
                        } 
                        for (int i = 0; i < TensorInfo[3]; i++) { 
                            float x_element = static_cast<float>(Gm_x (k * (TensorInfo[2] * TensorInfo[3]) + j * TensorInfo[3] + i));     
                            Duplicate<float>(tmp1, static_cast<TYPE_Y>(x_element), 1);    
                            Exp(tmp1, tmp1, 1);
                            tmp1(0) = tmp1(0) / sum_exp;      
                            Gm_y(k * (TensorInfo[2] * TensorInfo[3]) + j * TensorInfo[3] + i) = static_cast<TYPE_Y>(tmp1(0));  
                        }     
                    }
                } 
            }
        }
    }   
private:
    TPipe pipe;    
    GlobalTensor<TYPE_X> Gm_x;      
    GlobalTensor<TYPE_Y> Gm_y;  
    TBuf<QuePosition::VECCALC> tmp1Buffer; 

    uint32_t blockLength;   
};
    
extern "C" __global__ __aicore__ void softmax(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);    

    KernelSoftmax_Broadcast<DTYPE_X, DTYPE_Y> op;    
    op.Init(x, y, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);   
    op.Process(tiling_data.TensorInfo,tiling_data.dim);        
}
 

 
