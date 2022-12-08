#include <iostream>
#include <string>

#include "globals.h"
#include "mpc/AESObject.h"
#include "mpc/Precompute.h"
#include "util/connect.h"
#include "nn/NeuralNetConfig.h"
#include "nn/NeuralNetwork.h"
#include "test/unitTests.h"
#include "util/Profiler.h"
#include "util/model.h"
#include "nn/MaxpoolLayer.h"
#include "nn/ReLULayer.h"
#include "mpc/RSS.h"
#include "mpc/TPC.h"
#include "mpc/FPC.h"
#include "mpc/OPC.h"
#include "util/util.cuh"
#include "../ext/cxxopts.hpp"
#include <json.hpp>
#include <loguru.hpp>

int partyNum;
std::vector<AESObject*> aes_objects;
//AESObject* aes_indep;
//AESObject* aes_next;
//AESObject* aes_prev;
Precompute PrecomputeObject;

extern std::string *addrs;
extern BmrNet **communicationSenders;
extern BmrNet **communicationReceivers;

extern Profiler matmul_profiler;
Profiler func_profiler;
Profiler memory_profiler;
Profiler comm_profiler;
Profiler debug_profiler;

nlohmann::json piranha_config;

size_t db_bytes = 0;
size_t db_layer_max_bytes = 0;
size_t db_max_bytes = 0;

size_t train_dataset_size = 60000;
size_t test_dataset_size = 10000;
int log_learning_rate = 5;
size_t INPUT_SIZE;
size_t NUM_CLASSES;

void printUsage(const char *bin);
template<typename T, template<typename, typename...> typename Share>
void train(NeuralNetwork<T, Share> *, NeuralNetConfig *config, std::string, std::ifstream &, std::ifstream &, std::ifstream &, std::ifstream &, std::vector<int> &);
template<typename T, template<typename, typename...> typename Share>
void test(NeuralNetwork<T, Share> *, std::ifstream &, std::ifstream &);
void getBatch(std::ifstream &, std::istream_iterator<double> &, std::vector<double> &);
void deleteObjects();

int main(int argc, char** argv) {

    loguru::init(argc, argv);
    // Parse options -- retrieve party id and config JSON
    cxxopts::Options options("piranha", "GPU-accelerated platform for MPC computation");
    options.add_options()
        ("p,party", "Party number", cxxopts::value<int>())
        ("c,config", "Configuration file", cxxopts::value<std::string>())
        ;
    options.allow_unrecognised_options();

    auto parsed_options = options.parse(argc, argv);

    // Print help
    if (parsed_options.count("help")) {
        std::cout << options.help() << std::endl;
        std::cout << "Report bugs to jlw@berkeley.edu" << std::endl;
        return 0;
    }

    partyNum = parsed_options["party"].as<int>();

    std::ifstream input_config(parsed_options["config"].as<std::string>());
    input_config >> piranha_config;

    // Start memory profiler and initialize communication between parties
    memory_profiler.start();
    int cuda_device_count = -1;
    CUDA_CHECK(cudaGetDeviceCount(&cuda_device_count));
    LOG_S(INFO) << "Number of available cuda devices: " << cuda_device_count;
    //XXX initializeCommunication(options.ip_file, partyNum);
    std::vector<std::string> party_ips;
    for (int i = 0; i < piranha_config["num_parties"]; i++) {
	    party_ips.push_back(piranha_config["party_ips"][i]);
    }
    if (piranha_config["pipeline_parallel"]) {
        NUMCONNECTIONS = piranah_config["pipeline_parallel_groups"];
    } else {
        // TODO: document why the default value is 3.
        NUMCONNECTIONS = 3;
    }
    initializeCommunication(party_ips, partyNum, piranha_config["num_parties"]);

    synchronize(10000, piranha_config["num_parties"]); // wait for everyone to show up :)
    LOG_S(INFO) << "All " << piranha_config["num_parties"] << " have showed up" << std::endl;
    for (size_t i = 0; i < piranha_config["num_parties"]; i++) {
        // --------------> AES_TODO
        //Get AES strings from file and create vector of AESObjects
        //options.aes_file;
        //aes_objects[i] = new AESObject(options.)
    }
    //aes_indep = new AESObject(options.aes_indep_file);
    //aes_next = new AESObject(options.aes_next_file);
    //aes_prev = new AESObject(options.aes_prev_file);

    // Unit tests
    if (piranha_config["run_unit_tests"]) {
        LOG_S(0) << "Running unittests";
        int returnCode = runTests(argc, argv);
        if (returnCode != 0 || piranha_config["unit_test_only"]) {
            LOG_S(ERROR) << "Unittests failed";
            exit(returnCode);
        }
    }

    LOG_S(INFO) << "Do not run unittests";
    NeuralNetConfig nn_config;
    LOG_S(INFO) << "Loading network config from " << piranha_config["network"];
    loadModel(&nn_config, piranha_config["network"]);

#ifdef ONEPC
    LOG_S(INFO) << "Running One Party Computation";
    NeuralNetwork<uint64_t, OPC> net(&nn_config, piranha_config["nn_seed"]);
#else
#ifdef TWOPC
    LOG_S(INFO) << "Running Two Party Computation";

    NeuralNetwork<uint64_t, TPC> net(&nn_config, piranha_config["nn_seed"]);
#else
#ifdef FOURPC
    LOG_S(INFO) << "Running Four Party Computation";
    NeuralNetwork<uint64_t, FPC> net(&nn_config, piranha_config["nn_seed"]);
#else
    LOG_S(INFO) << "Running Three Party Computation";
    NeuralNetwork<uint64_t, RSS> net(&nn_config, piranha_config["nn_seed"]);
#endif
#endif
#endif

    net.printNetwork();
    net.printPipelineGroup();
    // Preload network weights if so-configured
    if (piranha_config["preload"]) {
        LOG_S(INFO) << "Loading network snapshot from" << piranha_config["preload_path"];
        net.loadSnapshot(piranha_config["preload_path"]);
    }

    // Load learning rate schedule
    int lr_len = piranha_config["lr_schedule"].size();
    std::vector<int> lr_schedule;
    for (int i = 0; i < lr_len; i++) {
	    lr_schedule.push_back(piranha_config["lr_schedule"][i]);
    }
    LOG_S(INFO) << "Learning rate schedule per layers: " << piranha_config["lr_schedule"]; 

    // Load training and test datasets
    std::ifstream train_data_file(std::string("files/") + nn_config.dataset + "/train_data");
    if (!train_data_file.is_open()) {
        std::cout << "Error opening training data file at " <<
            std::string("files/") + nn_config.dataset + "/train_data" <<
            std::endl;
    }

    std::ifstream train_label_file(std::string("files/") + nn_config.dataset + "/train_labels");
    if (!train_label_file.is_open()) {
        std::cout << "Error opening training label file at " <<
            std::string("files/") + nn_config.dataset + "/train_labels" <<
            std::endl;
    }

    std::ifstream test_data_file(std::string("files/") + nn_config.dataset + "/test_data");
    if (!test_data_file.is_open()) {
        std::cout << "Error opening test data file at " <<
            std::string("files/") + nn_config.dataset + "/test_data" <<
            std::endl;
    }

    std::ifstream test_label_file(std::string("files/") + nn_config.dataset + "/test_labels");
    if (!test_label_file.is_open()) {
        std::cout << "Error opening test label file at " <<
            std::string("files/") + nn_config.dataset + "/test_label" <<
            std::endl;
    }
    
    if (piranha_config["test_only"]) {
        LOG_S(INFO) << "Testing only.";
        test(&net, test_data_file, test_label_file);
    } else { // do training!
        LOG_S(INFO) << "Training and testing.";
        train(&net, &nn_config, piranha_config["run_name"], train_data_file, train_label_file, test_data_file, test_label_file, lr_schedule);
    }

    //delete aes_indep;
    //delete aes_next;
    //delete aes_prev;

    // ----------> AES_TODO Delete AES objects
    //for (int i = 0; i < aes_objects.size(); ++i) {
    //    delete aes_objects[i]; // Calls ~AESObject and deallocates *aes_objects[i]
    //}
    //aes_objects.clear();

    deleteObjects();

    // wait a bit for the prints to flush
    std::cout << std::flush;
    for(int i = 0; i < 10000000; i++);
   
    return 0;
}

template<typename T, template<typename, typename...> typename Share>
void updateAccuracy(NeuralNetwork<T, Share> *net, std::vector<double> &labels, int &correct) {

    Share<T> *activations = net->layers[net->layers.size() - 1]->getActivation();
    //printShareFinite(*activations, "last layer activations", 10);

    DeviceData<T> reconstructedOutput(activations->size());
    reconstruct(*activations, reconstructedOutput);

    std::vector<double> hostOutput(reconstructedOutput.size());
    copyToHost(reconstructedOutput, hostOutput, true);

    int nClasses = hostOutput.size() / MINI_BATCH_SIZE;

    for(int i = 0; i < MINI_BATCH_SIZE; i++) {
        auto result = std::max_element(hostOutput.begin() + (i * nClasses), hostOutput.begin() + ((i+1) * nClasses));
        int max_index = std::distance(hostOutput.begin(), result);

        if (labels[max_index] == 1.0) {
            correct++;
        }
    }
}

template<typename T, template<typename, typename...> typename Share>
void test(NeuralNetwork<T, Share> *net, std::ifstream &test_data, std::ifstream &test_labels) {

    std::istream_iterator<double> data_it(test_data);
    std::istream_iterator<double> label_it(test_labels);
    LOG_S(INFO) << "Start testing";
    //size_t numIterations = 1;
    size_t numIterations = test_dataset_size / MINI_BATCH_SIZE;
    int correct = 0;

    for (int i = 0; i < numIterations; i++) {

        std::vector<double> batch_data(MINI_BATCH_SIZE * INPUT_SIZE);
        std::vector<double> batch_labels(MINI_BATCH_SIZE * NUM_CLASSES);
        LOG_S(1) << "Start loading batch " << i;
        getBatch(test_data, data_it, batch_data);
        getBatch(test_labels, label_it, batch_labels);
        LOG_S(1) << "Loaded batch " << i;
        net->forward(batch_data);

        updateAccuracy(net, batch_labels, correct);
    }

    double acc = ((double)correct) / (numIterations * MINI_BATCH_SIZE);

    if (piranha_config["eval_accuracy"]) {
        LOG_F(INFO, "Test Accuracy: %f", acc);
    }
}

template<typename T, template<typename, typename...> typename Share>
void train(NeuralNetwork<T, Share> *net, NeuralNetConfig *config, std::string run_name,
        std::ifstream &train_data, std::ifstream &train_labels,
        std::ifstream &test_data, std::ifstream &test_labels,
        std::vector<int> &learning_rate) {

    size_t numIterations = (train_dataset_size / MINI_BATCH_SIZE); //+ 1; // assumes dataset doesn't divide equally into batches
    if (piranha_config["custom_iterations"]) {
        numIterations = (size_t) piranha_config["custom_iteration_count"];
    }

    size_t numEpochs = learning_rate.size(); // learning_rate.size() / ((numIterations / iterations_per_lr) + 1);
    if (piranha_config["custom_epochs"]) {
        numEpochs = (size_t) piranha_config["custom_epoch_count"];
    } 

    LOG_F(INFO, "TRAINING, EPOCHS = %d ITERATIONS = %d\n", numEpochs, numIterations);

    std::istream_iterator<double> data_it(train_data);
    std::istream_iterator<double> label_it(train_labels);

    LOG_S(INFO) << " == Training (" << numEpochs << " epochs) ==";

    int lr_ctr = 0;

    double total_comm_tx_mb = 0.0;
    double total_comm_rx_mb = 0.0;
    double total_time_s = 0.0;

    for (int e = 0; e < numEpochs; e++) {

        LOG_S(INFO) << " -- Epoch " << e << " (" << numIterations << " iterations, log_lr = " << learning_rate[e] << ") --";
        log_learning_rate = learning_rate[e];

        int correct = 0;
        for (int i = 0; i < numIterations; i++) {

            comm_profiler.clear();
            LOG_F(0, "Iteration %d", i);
            std::vector<double> batch_data(MINI_BATCH_SIZE * INPUT_SIZE);
            std::vector<double> batch_labels(MINI_BATCH_SIZE * NUM_CLASSES);

            getBatch(train_data, data_it, batch_data);
            getBatch(train_labels, label_it, batch_labels);

            Profiler toplevel_profiler;

            toplevel_profiler.start();
            net->forward(batch_data);
            toplevel_profiler.accumulate("fw-pass");

            updateAccuracy(net, batch_labels, correct);

            if (piranha_config["eval_inference_stats"]) {
                double fw_ms = toplevel_profiler.get_elapsed("fw-pass");
                LOG_F(0, "inference iteration (ms),%f\n", fw_ms);
                LOG_F(0, "inference TX comm (bytes),%d\n", comm_profiler.get_comm_tx_bytes());
                LOG_F(0, "inference RX comm (bytes),%d\n", comm_profiler.get_comm_rx_bytes());
            }

            if (piranha_config["eval_fw_peak_memory"]) {
                std::stringstream st;
                for(double i : memory_profiler.get_max_mem_mb()) st << i << " ";
                LOG_F(0, "fw pass peak memory (MB), %s\n", st.str().c_str());
            }

            total_comm_tx_mb += ((double)comm_profiler.get_comm_tx_bytes()) / 1024.0 / 1024.0;
            total_comm_rx_mb += ((double)comm_profiler.get_comm_rx_bytes()) / 1024.0 / 1024.0;

            if (piranha_config["inference_only"]) {
                continue;
            }

            toplevel_profiler.start();
            net->backward(batch_labels);
            toplevel_profiler.accumulate("bw-pass");

            if (piranha_config["eval_train_stats"]) {
                double fw_bw_ms = toplevel_profiler.get_elapsed_all();
                LOG_F(0, "training iteration (ms),%f\n", fw_bw_ms);
                LOG_F(0, "training TX comm (MB),%f\n", comm_profiler.get_comm_tx_bytes() / 1024.0 / 1024.0);
                LOG_F(0, "training RX comm (MB),%f\n", comm_profiler.get_comm_rx_bytes() / 1024.0 / 1024.0);

                double comm_ms = comm_profiler.get_elapsed("comm-time");
                LOG_F(0, "training comm (ms),%f\n", comm_ms);
                LOG_F(0, "training computation (ms),%f\n", fw_bw_ms - comm_ms);
            }

            if (piranha_config["eval_bw_peak_memory"]) {
                std::stringstream st;
                for(double i : memory_profiler.get_max_mem_mb()) st << i << " ";
                LOG_F(0, "total fw-bw pass peak memory (MB), %s\n", st.str().c_str());
            }

            total_time_s += toplevel_profiler.get_elapsed_all() / 1000.0;

            
            if (piranha_config["iteration_snapshots"]) {
                std::string snapshot_path = "output/"+run_name+"-epoch-"+std::to_string(e)+"-iteration-"+std::to_string(i);
                net->saveSnapshot(snapshot_path);

                if (piranha_config["test_iteration_snapshots"]) {
                    NeuralNetwork<uint64_t, OPC> test_net(config, 0);
                    test_net.loadSnapshot(snapshot_path);

                    test(&test_net, test_data, test_labels);
                    fflush(stdout);
                }
            }
        }

        if (piranha_config["eval_epoch_stats"]) {
            LOG_F(INFO, "epoch,%d\n", e);
            LOG_F(INFO, "total time (s),%f\n", total_time_s);
            LOG_F(INFO, "total tx comm (MB),%f\n", total_comm_tx_mb);
            LOG_F(INFO, "total rx comm (MB),%f\n", total_comm_rx_mb);
        }

        double acc = ((double)correct) / (numIterations * MINI_BATCH_SIZE);
        if (piranha_config["eval_accuracy"]) {
            LOG_F(INFO, "train accuracy %f at epoch %i", acc, e);
        }

        if (!piranha_config["no_test"]) {
            test(net, test_data, test_labels);
        }

        if (piranha_config["last_test"] && e == numEpochs - 1) {
            test(net, test_data, test_labels);
        }

        if (piranha_config["epoch_snapshots"]) {
            net->saveSnapshot("output/"+run_name+"-epoch-"+std::to_string(e));
        }

        fflush(stdout);
    }
}

void getBatch(std::ifstream &f, std::istream_iterator<double> &it, std::vector<double> &vals) {
    
    std::istream_iterator<double> eos;
    for(int i = 0; i < vals.size(); i++) {
        if (it == eos) {
            f.clear();
            f.seekg(0);
            std::istream_iterator<double> new_it(f);    
            it = new_it;
        }

        vals[i] = *it++;
    }
}

void deleteObjects() {
	//close connection
	for (int i = 0; i < piranha_config["num_parties"]; i++) {
		if (i != partyNum) {
			delete communicationReceivers[i];
			delete communicationSenders[i];
		}
	}
	delete[] communicationReceivers;
	delete[] communicationSenders;
	delete[] addrs;
}

