//==============================================================================
//
// Copyright 2018 The InsideLoop Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//==============================================================================

#ifndef IL_PPM_H
#define IL_PPM_H

#include <cstdio>
#include <string>

#include <il/Array2D.h>
#include <il/StaticArray.h>
#include <il/Status.h>

namespace il {

struct Pixel {
  unsigned char red;
  unsigned char green;
  unsigned char blue;
};

il::Array2D<il::Pixel> readPpm(const std::string& filename, il::io_t,
                               il::Status& status) {
  il::Array2D<il::Pixel> image{};

#ifdef IL_UNIX
  FILE* fp = std::fopen(filename.c_str(), "rb");
  if (!fp) {
    status.SetError(il::Error::FilesystemFileNotFound);
    return image;
  }
#else
  FILE* fp;
  const errno_t error_nb = fopen_s(&fp, filename.c_str(), "rb");
  if (error_nb != 0) {
    status.SetError(il::Error::FilesystemFileNotFound);
    return image;
  }
#endif

  char buffer[16];
  if (!std::fgets(buffer, sizeof(buffer), fp)) {
    status.SetError(il::Error::BinaryFileWrongFormat);
    return image;
  }
  if (buffer[0] != 'P' || buffer[1] != '6') {
    status.SetError(il::Error::BinaryFileWrongFormat);
    return image;
  }

  // Check for comments
  int c{std::getc(fp)};
  while (c == '#') {
    while (std::getc(fp) != '\n') {
    };
    c = std::getc(fp);
  }
  std::ungetc(c, fp);

  // read image size information
  int width;
  int height;
#ifdef IL_UNIX
  if (std::fscanf(fp, "%d %d", &width, &height) != 2) {
    status.SetError(il::Error::BinaryFileWrongFormat);
    return image;
  }
#else
  const int error_no = fscanf_s(fp, "%d %d", &width, &height);
  if (error_no != 2) {
    status.SetError(il::Error::BinaryFileWrongFormat);
    return image;
  }
#endif
  // read rgb component
  int rgb_comp_color;
#ifdef IL_UNIX
  if (std::fscanf(fp, "%d", &rgb_comp_color) != 1) {
    status.SetError(il::Error::BinaryFileWrongFormat);
    return image;
  }
#else
  {
    const int error_no = fscanf_s(fp, "%d", &rgb_comp_color);
    if (error_no != 1) {
      status.SetError(il::Error::BinaryFileWrongFormat);
      return image;
    }
  }
#endif
  // check rgb component depth
  if (rgb_comp_color != 255) {
    status.SetError(il::Error::BinaryFileWrongFormat);
    return image;
  }
  while (std::fgetc(fp) != '\n') {
  };

  // read pixel data from file
  image.Resize(width, height);
  if (std::fread(image.data(), 3 * width, height, fp) != height) {
    status.SetError(il::Error::BinaryFileWrongFormat);
    image.Resize(0, 0);
    return image;
  }

  std::fclose(fp);
  status.SetOk();

  return image;
}
}  // namespace il

#endif  // IL_PPM_H
