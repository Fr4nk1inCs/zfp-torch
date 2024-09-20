#include "c10/core/TensorOptions.h"
#include <iostream>
#include <torch/extension.h>

#include "compress.hpp"

int main() {
  auto tensor = torch::randn(
      {8}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));

  std::cout << tensor << std::endl;
  auto compressed = zfp_torch::compress(tensor, 8);
  std::cout << compressed << std::endl;
  auto decompressed = zfp_torch::decompress(compressed);
  std::cout << decompressed << std::endl;
}