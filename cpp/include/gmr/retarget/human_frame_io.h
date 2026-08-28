#pragma once

#include <filesystem>
#include <unordered_map>
#include <vector>

#include "gmr/retarget/retargeter.h"

namespace gmr {

struct HumanFrameSequence {
  std::vector<HumanFrame> frames;
  std::vector<std::unordered_map<std::string, bool>> footContacts;
  int fps = 30;
  std::string srcHuman;
  double actualHumanHeight = 0.0;
};

HumanFrameSequence loadHumanFrameSequence(const std::filesystem::path& filePath);

}  // namespace gmr
