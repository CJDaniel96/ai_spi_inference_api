import tensorrt as trt
import os

# Define file paths
onnx_file_path = r'D:\Dre\JQ_SPI_02_AI_API\models\patchcore\model.onnx\weights\onnx\model.onnx'
engine_file_path = r'D:\Dre\JQ_SPI_02_AI_API\models\patchcore\spi.engine'

# Define builder, network, and parser
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1GB

print(f'Loading ONNX file from path {onnx_file_path}...')
if not os.path.exists(onnx_file_path):
    print('ONNX file not found, please check the path')
    exit(0)

print('Beginning ONNX file parsing')
with open(onnx_file_path, 'rb') as model:
    if not parser.parse(model.read()):
        print ('ERROR: Failed to parse the ONNX file.')
        for error in range(parser.num_errors):
            print (parser.get_error(error))
        exit(0)

print('Completed parsing of ONNX file')

# --- Start of new robust check ---
if network.num_inputs == 0:
    print("ERROR: The ONNX model has no inputs after parsing. Check the ONNX export process, especially the opset version.")
    exit(0)

print(f"Found {network.num_inputs} inputs and {network.num_outputs} outputs.")
# --- End of new robust check ---


# Create an optimization profile
profile = builder.create_optimization_profile()
input_name = network.get_input(0).name
print(f"Input tensor name: {input_name}")

# Define the input dimensions: (min, optimal, max)
min_shape = (1, 3, 256, 256)
opt_shape = (1, 3, 256, 256)
max_shape = (1, 3, 256, 256)

profile.set_shape(input_name, min_shape, opt_shape, max_shape)
config.add_optimization_profile(profile)

print(f'Building an engine from file {onnx_file_path}; this may take a while...')

serialized_engine = builder.build_serialized_network(network, config)

if serialized_engine:
    with open(engine_file_path, 'wb') as f:
        f.write(serialized_engine)
    print(f'Serialized engine successfully saved at {engine_file_path}')
else:
    print('ERROR: Engine creation failed.')