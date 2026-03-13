#ifndef NETWORKS_H
#define NETWORKS_H

// 3-layer Small Neural Networks for testing different layer types

const char* DENSE_NETWORK_JSON = "{"
    "\"depth\": 3, \"rows\": 1, \"cols\": 1, \"layers_per_cell\": 1,"
    "\"layers\": ["
    "  {\"z\": 0, \"y\": 0, \"x\": 0, \"l\": 0, \"type\": \"Dense\", \"input_height\": 10, \"output_height\": 10, \"activation\": \"ReLU\", \"dtype\": \"F32\"},"
    "  {\"z\": 1, \"y\": 0, \"x\": 0, \"l\": 0, \"type\": \"Dense\", \"input_height\": 10, \"output_height\": 10, \"activation\": \"ReLU\", \"dtype\": \"F32\"},"
    "  {\"z\": 2, \"y\": 0, \"x\": 0, \"l\": 0, \"type\": \"Dense\", \"input_height\": 10, \"output_height\": 1, \"activation\": \"Softmax\", \"dtype\": \"F32\"}"
    "]"
"}";

const char* SWIGLU_NETWORK_JSON = "{"
    "\"depth\": 3, \"rows\": 1, \"cols\": 1, \"layers_per_cell\": 1,"
    "\"layers\": ["
    "  {\"z\": 0, \"y\": 0, \"x\": 0, \"l\": 0, \"type\": \"SwiGLU\", \"input_height\": 10, \"output_height\": 10, \"dtype\": \"F32\"},"
    "  {\"z\": 1, \"y\": 0, \"x\": 0, \"l\": 0, \"type\": \"SwiGLU\", \"input_height\": 10, \"output_height\": 10, \"dtype\": \"F32\"},"
    "  {\"z\": 2, \"y\": 0, \"x\": 0, \"l\": 0, \"type\": \"Dense\", \"input_height\": 10, \"output_height\": 1, \"activation\": \"Softmax\", \"dtype\": \"F32\"}"
    "]"
"}";

const char* MHA_NETWORK_JSON = "{"
    "\"depth\": 3, \"rows\": 1, \"cols\": 1, \"layers_per_cell\": 1,"
    "\"layers\": ["
    "  {\"z\": 0, \"y\": 0, \"x\": 0, \"l\": 0, \"type\": \"MHA\", \"input_height\": 10, \"output_height\": 10, \"num_heads\": 2, \"d_model\": 10, \"dtype\": \"F32\"},"
    "  {\"z\": 1, \"y\": 0, \"x\": 0, \"l\": 0, \"type\": \"Dense\", \"input_height\": 10, \"output_height\": 10, \"activation\": \"ReLU\", \"dtype\": \"F32\"},"
    "  {\"z\": 2, \"y\": 0, \"x\": 0, \"l\": 0, \"type\": \"Dense\", \"input_height\": 10, \"output_height\": 1, \"activation\": \"Softmax\", \"dtype\": \"F32\"}"
    "]"
"}";

const char* CNN2_NETWORK_JSON = "{"
    "\"depth\": 1, \"rows\": 1, \"cols\": 1, \"layers_per_cell\": 1,"
    "\"layers\": ["
    "  {\"z\": 0, \"y\": 0, \"x\": 0, \"l\": 0, \"type\": \"CNN2\", \"input_channels\": 1, \"filters\": 2, \"kernel_size\": 3, \"stride\": 1, \"padding\": 1, \"input_height\": 4, \"input_width\": 4, \"dtype\": \"F32\"}"
    "]"
"}";

const char* CNN3_NETWORK_JSON = "{"
    "\"depth\": 1, \"rows\": 1, \"cols\": 1, \"layers_per_cell\": 1,"
    "\"layers\": ["
    "  {\"z\": 0, \"y\": 0, \"x\": 0, \"l\": 0, \"type\": \"CNN3\", \"input_channels\": 1, \"filters\": 2, \"kernel_size\": 3, \"stride\": 1, \"padding\": 1, \"input_depth\": 4, \"input_height\": 4, \"input_width\": 4, \"dtype\": \"F32\"}"
    "]"
"}";

const char* RNN_NETWORK_JSON = "{"
    "\"depth\": 1, \"rows\": 1, \"cols\": 1, \"layers_per_cell\": 1,"
    "\"layers\": ["
    "  {\"z\": 0, \"y\": 0, \"x\": 0, \"l\": 0, \"type\": \"RNN\", \"input_height\": 10, \"output_height\": 10, \"dtype\": \"F32\"}"
    "]"
"}";

const char* LSTM_NETWORK_JSON = "{"
    "\"depth\": 1, \"rows\": 1, \"cols\": 1, \"layers_per_cell\": 1,"
    "\"layers\": ["
    "  {\"z\": 0, \"y\": 0, \"x\": 0, \"l\": 0, \"type\": \"LSTM\", \"input_height\": 10, \"output_height\": 10, \"dtype\": \"F32\"}"
    "]"
"}";

const char* EMBEDDING_NETWORK_JSON = "{"
    "\"depth\": 1, \"rows\": 1, \"cols\": 1, \"layers_per_cell\": 1,"
    "\"layers\": ["
    "  {\"z\": 0, \"y\": 0, \"x\": 0, \"l\": 0, \"type\": \"Embedding\", \"vocab_size\": 100, \"embedding_dim\": 64, \"dtype\": \"F32\"}"
    "]"
"}";

#endif // NETWORKS_H
