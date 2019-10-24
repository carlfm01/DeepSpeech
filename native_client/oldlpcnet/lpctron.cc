#include <iostream>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/memmapped_file_system.h"
// #include "tensorflow/data_flow_ops.h"
// #include "tensorflow/core/kernels/fifo_queue.h"
// #include "tensorflow/core/platform/types.h"

#include "tensorflow/cc/framework/scope.h"
//#include "tensorflow/cc/ops/standard_ops.h"
//#include "tensorflow/cc/training/coordinator.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/notification.h"
// #include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
// #include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/queue_runner.pb.h"
#include "tensorflow/core/public/session.h"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <map>
#include <chrono>
#include <iomanip>
extern "C"
{
#include "lpcnet.h"
}
extern "C"
{
#include "lpctron.h"
}

static clock_t start, end, mid;
//static tensorflow::Tensor input_lengths_t(tensorflow::DT_INT32, tensorflow::TensorShape({1}));
//static auto flat_input_lengths = input_lengths_t.flat<int>();
static std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
static tensorflow::Session *session;
//aábcdeéfghiíjklmnñoópqrstuúüvwxyz!,.:;?¡¿-|\" AÁBCDEÉFGHIÍJKLMNOÓPQRSTUÚVWYZ«»°—
static std::map<char, int> char2seq = {
    {'_', 0},
    {'~', 1},
    {'a', 2},
    {'á', 3},
    {'b', 4},
    {'c', 5},
    {'d', 6},
    {'e', 7},
    {'é', 8},
    {'f', 9},
    {'g', 10},
    {'h', 11},
    {'i', 12},
    {'í', 13},
    {'j', 14},
    {'k', 15},
    {'l', 16},
    {'m', 17},
    {'n', 18},
    {'ñ', 19},
    {'o', 20},
    {'ó', 21},
    {'p', 22},
    {'q', 23},
    {'r', 24},
    {'s', 25},
    {'t', 26},
    {'u', 27},
    {'ú', 28},
    {'ü', 29},
    {'v', 30},
    {'w', 31},
    {'x', 32},
    {'y', 33},
    {'z', 34},
    {'!', 35},
    {',', 36},
    {'.', 37},
    {':', 38},
    {';', 39},
    {'?', 40},
    {'¡', 41},
    {'¿', 42},
    {'-', 43},
    {'|', 44},
    {'"', 45},
    {' ', 46},
    {'A', 47},
    {'Á', 48},
    {'B', 49},
    {'C', 50},
    {'D', 51},
    {'E', 52},
    {'É', 53},
    {'F', 54},
    {'G', 55},
    {'H', 56},
    {'I', 57},
    {'Í', 58},
    {'J', 59},
    {'K', 60},
    {'L', 61},
    {'M', 62},
    {'N', 63},
    {'O', 64},
    {'Ó', 65},
    {'P', 66},
    {'Q', 67},
    {'R', 68},
    {'S', 69},
    {'T', 70},
    {'U', 71},
    {'Ú', 72},
    {'V', 73},
    {'W', 74},
    {'Y', 75},
    {'Z', 76},
    {'«', 77},
    {'»', 78},
    {'°', 79},
    {'—', 80}};

static std::ofstream pcm_file;
void callback(short *pcm, int num_elements)
{
    pcm_file.write((char *)pcm, num_elements * sizeof(short));
}

int init_lpctron(const char* modelpath)
{
    tensorflow::GraphDef graph_def;
    std::cout << "Loading model at"<< modelpath << std::endl;

    tensorflow::Status status;
    tensorflow::MemmappedEnv *mmap_env_;
    tensorflow::SessionOptions options;

    mmap_env_ = new tensorflow::MemmappedEnv(tensorflow::Env::Default());
     
    bool is_mmap = std::string(modelpath).find(".pbmm") != std::string::npos;
    if (!is_mmap)
    {
        std::cerr << "Warning: reading entire model file into memory. Transform model file into an mmapped graph to reduce heap usage." << std::endl;
    }
    else
    {

        /* Procedure to convert protobuf model to memory mapped model 
        bazel build tensorflow/contrib/util:convert_graphdef_memmapped_format
        bazel-bin/tensorflow/contrib/util/convert_graphdef_memmapped_format --in_graph=inference_model.pb --out_graph=inference_model.pbmm
        */

        status = mmap_env_->InitializeFromFile(modelpath);
        if (!status.ok())
        {
            std::cerr << status << std::endl;
            return -1;
        }

        options.config.mutable_graph_options()
            ->mutable_optimizer_options()
            ->set_opt_level(tensorflow::OptimizerOptions::L0);
        options.env = mmap_env_;
    }

    if (!status.ok())
    {
        std::cout << status.ToString() << std::endl;
        return -1;
    }
    std::cout << "model loaded" << std::endl;

    std::cout << "Creating session" << std::endl;

    tensorflow::ConfigProto &config = options.config;
    config.set_inter_op_parallelism_threads(1);
    config.set_intra_op_parallelism_threads(1);
    config.set_use_per_session_threads(true);

    status = tensorflow::NewSession(options, &session);

    if (is_mmap)
    {
        status = ReadBinaryProto(mmap_env_,
                                 tensorflow::MemmappedFileSystem::kMemmappedPackageDefaultGraphDef,
                                 &graph_def);

        std::cout << "Mmaping the model file" << std::endl;
    }
    else
    {
        status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), modelpath, &graph_def);
        std::cout << "Loading PB Memory " << std::endl;
    }

    if (!status.ok())
    {
        std::cout << status.ToString() << "\n";
        return -1;
    }
    std::cout << "Session created" << std::endl;

    std::cout << "Connecting session to graph..." << std::endl;
    status = session->Create(graph_def);
    if (!status.ok())
    {
        std::cout << "Create error: " << status.ToString() << "\n";
        return -1;
    }
    std::cout << "Graph connected" << std::endl;
    //printf('Graph created');
    init_lpcnet();
    return 0;
}
void init(const char* modelpath)
{
    init_lpctron(modelpath);
}
int tts_lpctron(std::string text)
{

    std::vector<int> seq_inputs;
    start = std::clock();
    std::cout << "Creating char inputs" << std::endl;
    static tensorflow::Tensor input_lengths_t(tensorflow::DT_INT32, tensorflow::TensorShape({1}));
    static auto flat_input_lengths = input_lengths_t.flat<int>();
    std::cout << "Entering for to create :" << text << std::endl;
    for (const char &c : text)
    {
        if (char2seq.find(c) != char2seq.end())
        {
            seq_inputs.push_back(char2seq.at(c));
            std::cout << c << ":" << char2seq.at(c) << " ";
        }
    }
    std::cout << std::endl;

    int inputs_size = seq_inputs.size();
    if (inputs_size > 0)
    {

        tensorflow::Tensor inputs_t(tensorflow::DT_INT32, tensorflow::TensorShape({1, inputs_size}));

        auto flat_inputs = inputs_t.tensor<int, 2>();
        flat_input_lengths(0) = inputs_size;
        for (int i = 0; i < inputs_size; i++)
        {
            flat_inputs(0, i) = seq_inputs[i];
        }

        inputs.clear();
        inputs.emplace_back(std::string("inputs"), inputs_t);
        inputs.emplace_back(std::string("input_lengths"), input_lengths_t);

        std::cout << "Input created" << std::endl;

        std::vector<tensorflow::Tensor> outputs;

        std::cout << "Running inference..." << std::endl;

        tensorflow::Status status = session->Run(inputs, {std::string("Tacotron_model/inference/add")}, {}, &outputs);
        if (!status.ok())
        {
            std::cout << status.ToString() << "\n";
            return 1;
        }
        std::cout << "got " << (int)outputs.size() << " outputs" << std::endl;
        if (outputs.size() == 1)
        {
            const tensorflow::Tensor &mels = outputs[0];
            auto mels_map = outputs[0].tensor<float, 3>();
            int dim0 = mels.shape().dim_size(0);
            int dim1 = mels.shape().dim_size(1);
            int dim2 = mels.shape().dim_size(2);
            std::cout << "mels dimensions = " << dim0 << ", " << dim1 << ", " << dim2 << std::endl;
            // float *mels_data = new float[dim0 * dim1 * dim2];
            std::vector<float> mels_data;

            for (int i = 0; i < dim0; i++)
            {
                for (int j = 0; j < dim1; j++)
                {
                    for (int k = 0; k < dim2; k++)
                    {
                        // *(mels_data + i * dim1 * dim2 + j * dim2 + k) = mels_map(i, j, k);
                        mels_data.push_back(mels_map(i, j, k));
                    }
                }
            }
            std::ofstream data_file; // pay attention here! ofstream
            data_file.open("inference_model_cpp.f32", std::ios::out | std::ios::binary);
            data_file.write(reinterpret_cast<char *>(&mels_data[0]), mels_data.size() * sizeof(float));
            data_file.close();
            /*
            std::ofstream data_file;      // pay attention here! ofstream
            data_file.open("inference_model_cpp.f32", std::ios::out | std::ios::binary);
            data_file.write(reinterpret_cast<char*>(&mels_data[0]), mels_data.size()*sizeof(float)); 
            data_file.close();
        
            
            float *features = new float[mels_data.size()];
            for (int i = 0; i < mels_data.size(); i++) {
                features[i] = mels_data[i];
            }
            */
            mid = clock();
            double time_mid = double(mid - start) / double(CLOCKS_PER_SEC);
            std::cout << "Time taken by program Tactron2 : " << std::fixed
                      << time_mid << std::setprecision(5);
            std::cout << " sec " << std::endl;

            run_lpcnet(reinterpret_cast<float *>(&mels_data[0]), mels_data.size());
            //delete[] features;
            return 0;
            // std::cout << "Writing " << mels_data.size() << " values to file..." << std::endl;
            // // std::ofstream outfile ("inference_model_cpp.f32",std::ofstream::binary);
            // // outfile.write((char *)mels_data,(dim0 * dim1 * dim2)*sizeof(float));
            // // outfile.close();
            // std::ofstream data_file;      // pay attention here! ofstream
            // data_file.open("inference_model_cpp.f32", std::ios::out | std::ios::binary);
            // data_file.write(reinterpret_cast<char*>(&mels_data[0]), mels_data.size()*sizeof(float));
            // data_file.close();

            // std::cout << "File written" << std::endl;

            // delete[] mels_data;
        }
    }

    return -1;
}

void tts(const char *input)
{
    std::string text(input);
    std::cout << "Input text: " << text << std::endl;
    //if (init_lpctron() == 0)
    // {

    std::cout << "About to enter tts_lpctron" << std::endl;
    if (tts_lpctron(text) != 0)
    {
        std::cout << "error with tts_lpctron" << std::endl;
    }
    //}
    // else
    // {
    //    std::cout << "error with init_lpctron" << std::endl;
    // }
    std::cout << "done" << std::endl;
    end = clock();
    double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
    std::cout << "Time taken by program is : " << std::fixed
              << time_taken << std::setprecision(5);
    std::cout << " sec " << std::endl;
}
