#pragma once

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>

namespace certified_output_guard
{
inline constexpr const char* kMarkerName = ".certified_immutable";
inline constexpr const char* kRefusalToken = "WRITE_REFUSED_BY_IMMUTABLE_GUARD";

inline std::optional<std::filesystem::path> marker_for(const std::filesystem::path& target)
{
  std::error_code error;
  auto current = std::filesystem::absolute(target, error);
  if (error) current = target;
  if (!std::filesystem::is_directory(current, error)) current = current.parent_path();
  while (!current.empty())
  {
    const auto marker = current / kMarkerName;
    error.clear();
    if (std::filesystem::is_regular_file(marker, error)) return marker;
    const auto parent = current.parent_path();
    if (parent == current) break;
    current = parent;
  }
  return std::nullopt;
}

[[noreturn]] inline void refuse_write(const std::filesystem::path& target,
                                      const std::filesystem::path& marker)
{
  std::cerr << kRefusalToken << ": target=" << target << "; marker=" << marker << std::endl;
  std::_Exit(86);
}

inline void assert_output_path_writable(const std::filesystem::path& target)
{
  if (const auto marker = marker_for(target)) refuse_write(target, marker.value());
}
}  // namespace certified_output_guard
