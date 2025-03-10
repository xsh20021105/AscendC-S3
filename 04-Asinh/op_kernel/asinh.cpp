#include "kernel_operator.h"
#include <type_traits>           
using namespace AscendC;      
constexpr int32_t BUFFER_NUM = 2; 
template<typename T> struct Map {using type = T;};      
template<> struct Map<int8_t> {using type = half;};          
           
template<typename TYPE_X, typename TYPE_Y> class KernelAsinh_Official { 
    using T = TYPE_Y;          
public:  
                      
    __aicore__ inline KernelAsinh_Official() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain) {            
                                          
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);  
        this->tileLength = block_size;                                                           
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);
        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);
                 
        auto xPointer = core_size * GetBlockIdx();  
        auto bufferlength = this->blockLength;
     
        Gm_x.SetGlobalBuffer((__gm__ TYPE_X*)x + xPointer, bufferlength);
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y + xPointer, bufferlength); 

        pipe.InitBuffer(Q_x, BUFFER_NUM, this->tileLength * sizeof(TYPE_X));
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y)); 

        pipe.InitBuffer(tmpXBuffer, this->tileLength * sizeof(float)); 
        pipe.InitBuffer(tmpYBuffer, this->tileLength * sizeof(float)); 
    }    
    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount-1; i++) {   
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }

        uint32_t length = this->blockLength - this->tileLength * (loopCount - 1);  
        CopyIn(loopCount - 1, length);  
        Compute(loopCount - 1, length); 
        CopyOut(loopCount - 1, length); 
    }
        
private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_X> x = Q_x.AllocTensor<TYPE_X>(); 
        DataCopy(x, Gm_x[progress * this->tileLength], length); 
        Q_x.EnQue(x);
    }
   __aicore__ inline void Compute(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_X> x = Q_x.DeQue<TYPE_X>();
        LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();   

        LocalTensor<float> tmpX = tmpXBuffer.Get<float>();
        LocalTensor<float> tmpY = tmpYBuffer.Get<float>();  

        if constexpr (std::is_same_v<T, half>) { 
            Cast(tmpX, x, RoundMode::CAST_NONE, length);      
            Cast(tmpY, y, RoundMode::CAST_NONE, length);       
            Mul(tmpY, tmpX, tmpX, length);                         
            Adds(tmpY, tmpY, static_cast<float>(1), length); 
            Sqrt(tmpY, tmpY, length);                                     
            Sub(tmpY, tmpY, tmpX, length);                          
            Ln(tmpY, tmpY, length);  
            Muls(tmpY, tmpY,static_cast<float>(-1), length);                                    
            Cast(y, tmpY, RoundMode::CAST_CEIL, length);  
        }else if constexpr (std::is_same_v<T, float>) {  
            Mul(y, x, x, length);                          
            Adds(y, y, static_cast<TYPE_Y>(1), length);  
            Sqrt(y, y, length);                                    
            Sub(y, y, x, length);                           
            Ln(y, y, length);    
            Muls(y, y,static_cast<TYPE_Y>(-1), length);                                 
        }
        Q_x.FreeTensor(x);
        Q_y.EnQue<TYPE_Y>(y); 
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_Y> y = Q_y.DeQue<TYPE_Y>();
        DataCopy(Gm_y[progress * this->tileLength], y, length);
        Q_y.FreeTensor(y);
    } 

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x; 
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
    TBuf<QuePosition::VECCALC> tmpXBuffer,tmpYBuffer;   
    GlobalTensor<TYPE_X> Gm_x;
    GlobalTensor<TYPE_Y> Gm_y;

    uint32_t blockLength;  
    uint32_t tileNum;
    uint32_t tileLength;    

}; 
    
extern "C" __global__ __aicore__ void asinh(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);  

    KernelAsinh_Official<DTYPE_X, DTYPE_Y> op;  
    op.Init(x, y, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);   
    op.Process();      
}
 

