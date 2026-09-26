#include <zstd.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
  const bool decompress = argc > 1 && std::string(argv[1]) == "-d";
  constexpr std::size_t buffer_size = 1U << 20;
  std::vector<char> input(buffer_size);
  std::vector<char> output(buffer_size);
  if (decompress)
  {
    ZSTD_DStream* stream = ZSTD_createDStream();
    if (!stream) return 2;
    if (ZSTD_isError(ZSTD_initDStream(stream))) return 3;
    while (true)
    {
      const std::size_t read = std::fread(input.data(), 1, input.size(), stdin);
      if (read == 0) break;
      ZSTD_inBuffer in{input.data(), read, 0};
      while (in.pos < in.size)
      {
        ZSTD_outBuffer out{output.data(), output.size(), 0};
        const std::size_t result = ZSTD_decompressStream(stream, &out, &in);
        if (ZSTD_isError(result)) return 4;
        if (std::fwrite(output.data(), 1, out.pos, stdout) != out.pos) return 5;
      }
    }
    ZSTD_freeDStream(stream);
    return 0;
  }
  ZSTD_CStream* stream = ZSTD_createCStream();
  if (!stream) return 6;
  if (ZSTD_isError(ZSTD_initCStream(stream, 3))) return 7;
  while (true)
  {
    const std::size_t read = std::fread(input.data(), 1, input.size(), stdin);
    if (read == 0) break;
    ZSTD_inBuffer in{input.data(), read, 0};
    while (in.pos < in.size)
    {
      ZSTD_outBuffer out{output.data(), output.size(), 0};
      const std::size_t result = ZSTD_compressStream2(stream, &out, &in, ZSTD_e_continue);
      if (ZSTD_isError(result)) return 8;
      if (std::fwrite(output.data(), 1, out.pos, stdout) != out.pos) return 9;
    }
  }
  while (true)
  {
    ZSTD_outBuffer out{output.data(), output.size(), 0};
    ZSTD_inBuffer in{nullptr, 0, 0};
    const std::size_t result = ZSTD_compressStream2(stream, &out, &in, ZSTD_e_end);
    if (ZSTD_isError(result)) return 10;
    if (std::fwrite(output.data(), 1, out.pos, stdout) != out.pos) return 11;
    if (result == 0) break;
  }
  ZSTD_freeCStream(stream);
  return 0;
}
