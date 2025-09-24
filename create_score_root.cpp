#include <ROOT/RDataFrame.hxx>
#include <TString.h>
#include <map>
#include <tuple>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

static std::unordered_map<std::string, float> scoreMap;
std::string root_file = "/work/kfuss/SVJ_GNN/data/background/data_skimmed_no_resonances.root";
std::string output_folder = "/work/kfuss/SVJ_GNN/output/fully_new_vs8/";

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
        float score;

        std::getline(ss, item, ','); run = static_cast<unsigned int>(std::stoul(item));
        std::getline(ss, item, ','); lumi = static_cast<unsigned int>(std::stoul(item));
        std::getline(ss, item, ','); event = std::stoull(item);
        std::getline(ss, item, ','); score = std::stof(item);


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

int main_() {
    ROOT::EnableImplicitMT();

    std::cout << "Going to load csv file" << std::endl;

    LoadScoresCSV(output_folder + "scores_bkg.csv");

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

    return 0;
}
