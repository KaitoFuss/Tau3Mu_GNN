#include <ROOT/RDataFrame.hxx>
#include <TString.h>
#include <map>
#include <tuple>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

static std::unordered_map<std::string, float> scoreMap;

// std::string root_file = "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/signal_W/signal_skimmed_no_resonances.root";
// std::string root_file = "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_W/data_skimmed_no_resonances.root";

std::string root_file = "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/signal_HF/ds_large_signal_skimmed_no_resonances.root";
// std::string root_file = "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_HF/data_skimmed_no_resonances_part1.root"; // files 1-4!!!

std::string output_folder = "/work/kfuss/SVJ_GNN/output/early_stop_fix_HF_BDT_vs1/";

std::string csv_filename = "scores_signal.csv";
// std::string csv_filename = "scores_background.csv";


void LoadScoresCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open CSV file: " << filename << std::endl;
        return;
    }
    

    std::string line;
    std::getline(file, line); // skip header, if present
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string item;
        unsigned int run, lumi;
        unsigned long long event;
        double score;

        std::getline(ss, item, ','); run = static_cast<unsigned int>(std::stoul(item));
        std::getline(ss, item, ','); lumi = static_cast<unsigned int>(std::stoul(item));
        std::getline(ss, item, ','); event = std::stoull(item);
        std::getline(ss, item, ','); try {
        score = std::stod(item);
    } catch (std::exception &e) {
        std::cerr << "Error parsing score in line: " << line << std::endl;
        std::cerr << "Score field was: '" << item << "'" << std::endl;
        throw;
    }


        std::string key = std::to_string(run) + ":" + std::to_string(lumi) + ":" + std::to_string(event);
        scoreMap[key] = score;
    }
}

float read_score(unsigned int run, unsigned int lumi, unsigned long long event) {
    std::string key = std::to_string(run) + ":" + std::to_string(lumi) + ":" + std::to_string(event);
    auto it = scoreMap.find(key);
    if (it != scoreMap.end()) return it->second;
    return -1.0f;
}

void create_score_root() {
    ROOT::EnableImplicitMT();

    std::cout << "Going to load csv file" << std::endl;

    LoadScoresCSV(output_folder + csv_filename);

    std::cout << "Going to load root file" << std::endl;

    ROOT::RDataFrame rdf("Events", root_file);

    std::cout << "Going to write GNN score as new branch" << std::endl;

    // auto rdf2 = rdf.Define("GNN_Score", [](unsigned int run, unsigned int lumi, unsigned long long event) {
    //     return read_score(run, lumi, event);
    // }, {"run", "luminosityBlock", "event"});
    auto rdf2 = rdf.Define("GNN_Score", [](unsigned int run, unsigned int lumi, unsigned long long event) {
        return read_score(run, lumi, event);
    }, {"run", "luminosityBlock", "event"});

     

    std::string base_name = TString(root_file).ReplaceAll(".root", "_with_scores.root").Data();
    size_t pos = base_name.find_last_of('/');
    if (pos != std::string::npos) {
        base_name = base_name.substr(pos + 1);
    }

    std::string sep = (output_folder.back() == '/') ? "" : "/";
    std::string output_file = output_folder + sep + base_name;

    std::cout << "Saving to file: " << output_file << std::endl;


    std::cout << "Going to save new root file" << std::endl;

    rdf2.Snapshot("Events", output_file);

    std::cout << "GNN scores added to: " << output_file << std::endl;

    return;
}
