#pragma once

#include <filesystem>
#include <optional>
#include <string>

#include "gmr/retarget/contact_ground.h"

namespace gmr {

    ContactGroundConfig robotContactGroundPreset(const std::filesystem::path& gmrRoot, const std::string& robot);

    ContactGroundConfig buildContactGroundConfig(const std::filesystem::path& gmrRoot, const std::string& robot,
                                                 const std::filesystem::path& ikConfigPath,
                                                 const std::string& humanRootName,
                                                 const std::optional<bool>& cliOverride = std::nullopt);

}  // namespace gmr
