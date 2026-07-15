#pragma once

#include <filesystem>
#include <vector>

#include "gmr/retarget/retargeter.h"

namespace gmr {

struct HumanFrameSequence {
  std::vector<HumanFrame> frames;
  int fps = 30;
  std::string srcHuman;
  double actualHumanHeight = 0.0;
};

HumanFrameSequence loadHumanFrameSequence(const std::filesystem::path& filePath);

}  // namespace gmr
