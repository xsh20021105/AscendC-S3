#include "register/tilingdata_base.h"
 
namespace optiling {   
BEGIN_TILING_DATA_DEF(TilingData)

  TILING_DATA_FIELD_DEF(int, dim); 
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(uint32_t, ALIGN_NUM);
  TILING_DATA_FIELD_DEF(uint32_t, core_size);
  TILING_DATA_FIELD_DEF(uint32_t, block_size); 
  TILING_DATA_FIELD_DEF(uint32_t, core_remain);   
  TILING_DATA_FIELD_DEF(uint32_t, total_length);    
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 1, inputLength);  
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 1 * 10, TensorInfo);
  
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Softmax, TilingData)
}


