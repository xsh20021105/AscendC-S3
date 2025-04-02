#include "scatter_elements_tiling.h" 
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>       
 
namespace optiling {                            
const uint32_t BLOCK_SIZE = 32;     
static ge::graphStatus TilingFunc(gert::TilingContext* context) {   
    TilingData tiling;  
    int32_t NUM = 6;     
    uint32_t length = 0; 
    uint32_t total_length = 0;
    uint32_t sizeofdatatype; 
    
    uint32_t TensorInfo[3 * 10] = {};  
    uint32_t inputLength[3] = {};     

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ub_size; 
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);  
    auto aivNum = ascendcPlatform.GetCoreNum(); 
   
    for (int i = 0; i < 3; ++i) 
        length = std::max<uint32_t>(length, context->GetInputShape(i)->GetStorageShape().GetDimNum());
    for (int i = 0; i < 3; ++i) { 
        const gert::StorageShape* shape = context->GetInputShape(i);
        inputLength[i] = context->GetInputTensor(i)->GetShapeSize();       
        TensorInfo[i * 10 + 0] = shape->GetStorageShape().GetDimNum();             
        for (int j = 1; j <= shape->GetStorageShape().GetDimNum(); j++) {  
            TensorInfo[i * 10 + j] = shape->GetStorageShape().GetDim(j - 1);                   
        } 
    }        
    tiling.set_TensorInfo(TensorInfo);  
       
    for (int i = 0; i < 3; ++i) {  
        total_length = std::max<uint32_t>(total_length, context->GetInputTensor(i)->GetShapeSize());
    }      
     
    tiling.set_inputLength(inputLength);  
    tiling.set_total_length(total_length); 
    
    auto dt = context->GetInputTensor(0)->GetDataType();
   
    if (dt == ge::DT_INT8) {
        sizeofdatatype = 1;
        NUM = 12;
    }
    else if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16) { 
        sizeofdatatype = 2;
    }         
    else {
        sizeofdatatype = 4;
    }              
  
    uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;   
    tiling.set_ALIGN_NUM(ALIGN_NUM);

    uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / 2) / NUM;         
    tiling_size = tiling_size <= 8 ? tiling_size : tiling_size / 8 * 8; 
    
    uint32_t block_size = tiling_size * ALIGN_NUM;
    tiling.set_block_size(block_size); 

    aivNum = (aivNum < total_length / block_size) ? aivNum : (total_length / block_size);
    aivNum = aivNum >= 1 ? aivNum : 1;  
    
    uint32_t core_size = (total_length / aivNum) / (ALIGN_NUM * 8) * (ALIGN_NUM * 8);
    tiling.set_core_size(core_size); 

    uint32_t core_remain = total_length - aivNum * core_size;
    tiling.set_core_remain(core_remain);  

    int axis = *context->GetAttrs()->GetInt(0);   
    tiling.set_axis(axis); 

    const char *str = context->GetAttrs()->GetAttrPointer<char>(1);
    int signal = 0;
    if (strcmp(str, "add") == 0){
        signal = 0;
    }
    else if(strcmp(str, "multiply") == 0){
        signal = 1;
    }else{
        signal = 1;
    }
    printf("signala = %d\n", signal);
    tiling.set_signal(signal);

    context->SetBlockDim(aivNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}                   
} 
 
namespace ge { 
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}
 
namespace ops {
class ScatterElements : public OpDef {
public:
    explicit ScatterElements(const char* name) : OpDef(name)
    {
        this->Input("var")   
            .ParamType(REQUIRED) 
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("indices")     
            .ParamType(REQUIRED) 
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("updates")     
            .ParamType(REQUIRED) 
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("axis").AttrType(OPTIONAL).Int(0);
        this->Attr("reduce").AttrType(OPTIONAL).String("None");
        this->SetInferShape(ge::InferShape);   
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");
    }
};
 
OP_ADD(ScatterElements); 
}
  
