#include <filesystem>
#include <fstream>
#include <iostream>

#include "certified_output_guard.hpp"

int main(int argc, char** argv)
{
  if (argc != 3)
  {
    std::cerr << "usage: certified_output_guard_probe ROOT TARGET\n";
    return 64;
  }
  const std::filesystem::path root(argv[1]);
  const std::filesystem::path target(argv[2]);
  certified_output_guard::assert_output_path_writable(target);
  std::filesystem::create_directories(root);
  std::ofstream stream(target, std::ios::out | std::ios::trunc);
  if (!stream) return 74;
  stream << "guard_probe_write\n";
  return 0;
}
