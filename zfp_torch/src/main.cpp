#include "c10/core/TensorOptions.h"
#include "zfp_compresser.hpp"
#include <iostream>
#include <torch/extension.h>

int main() {
  auto tensor = torch::randn(
      {8}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));

  std::cout << tensor << std::endl;
  auto compresser = ZFPCompresser(8);
  auto compressed = compresser.compress(tensor);
  std::cout << compressed << std::endl;
  auto decompressed = compresser.decompress(compressed);
  std::cout << decompressed << std::endl;
}