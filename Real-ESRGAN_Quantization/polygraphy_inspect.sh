clear

rm ./*pb ./result-*.txt

python remove_initializer_from_input.py --input ./model.onnx --output ./model_output.onnx

# 导出 .onnx 详细信息
#polygraphy inspect model model_output.onnx \
#    --mode=full \
#    > result-inspectOnnxModel.txt

# 用上面的 onnx生成一个.plan 及其 tactics
#polygraphy run model.onnx \
#    --trt \
#    --workspace 1000000000 \
#    --save-engine="./model.plan" \
#    --save-tactics="./model.tactic" \
#    --save-inputs="./model-input.txt" \
#    --save-outputs="./model-output.txt" \
#    --silent \
#    --trt-min-shapes 'x:[1,3,3,3]' \
#    --trt-opt-shapes 'x:[4,3,3,3]' \
#    --trt-max-shapes 'x:[64,3,3,3]' \
#    --input-shapes   'x:[4,3,3,3]'

# 导出上面 .plan 的详细信息（要求 TensorRT >= 8.2）
#polygraphy inspect model model.plan \
#    --mode=full \
#    > result-inspectPlanModel.txt

polygraphy inspect tactics model.tactic \
    > result-inspectPlanTactic.txt

polygraphy inspect capability model.onnx > result-NonZero.txt