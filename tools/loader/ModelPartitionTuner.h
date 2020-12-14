/**
 * ETRI
 */
#ifndef GLOW_TOOLS_PARTITION_TUNER_H
#define GLOW_TOOLS_PARTITION_TUNER_H

namespace glow {

/// This is struct for saving the profiled information.
struct PartitionProfileInfo {
  std::map<size_t, std::map<std::string, float>> peProfileMap;
};

class CostNode {
public:
  //Node* node_;
  size_t peID = -1;
  size_t partitionID = -1;
  std::string name;
  std::string backendName;
  size_t totalCost = INFINITY;

};

} // namespace glow

#endif // GLOW_TOOLS_PARTITION_TUNER_H
