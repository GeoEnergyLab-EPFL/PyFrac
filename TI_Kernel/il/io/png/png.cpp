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

#include <il/io/png/png.h>

#include "png.h"

namespace il {

il::Array3C<unsigned char> loadPng(const std::string &filename, il::io_t,
                                   il::Status &status) {
  il::Array3C<unsigned char> image{};

  unsigned char header[8];
  FILE *fp{fopen(filename.c_str(), "rb")};
  if (fp == nullptr) {
    status.SetError(il::Error::FilesystemFileNotFound);
    IL_SET_SOURCE(status);
    fclose(fp);
    return image;
  }

  fread(header, 1, 8, fp);
  if (png_sig_cmp(header, 0, 8)) {
    status.SetError(il::Error::BinaryFileWrongFormat);
    IL_SET_SOURCE(status);
    fclose(fp);
    return image;
  }

  png_structp png_ptr{
      png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr)};
  if (png_ptr == nullptr) {
    status.SetError(il::Error::Undefined);
    IL_SET_SOURCE(status);
    fclose(fp);
    return image;
  }
  png_infop info_ptr{png_create_info_struct(png_ptr)};
  if (info_ptr == nullptr) {
    status.SetError(il::Error::Undefined);
    IL_SET_SOURCE(status);
    fclose(fp);
    return image;
  }
  if (setjmp(png_jmpbuf(png_ptr))) {
    status.SetError(il::Error::Undefined);
    IL_SET_SOURCE(status);
    fclose(fp);
    return image;
  }

  png_init_io(png_ptr, fp);
  png_set_sig_bytes(png_ptr, 8);

  png_read_info(png_ptr, info_ptr);

  int width{static_cast<int>(png_get_image_width(png_ptr, info_ptr))};
  int height{static_cast<int>(png_get_image_height(png_ptr, info_ptr))};
  //  png_byte color_type{png_get_color_type(png_ptr, info_ptr)};
  png_byte bit_depth{png_get_bit_depth(png_ptr, info_ptr)};
  IL_EXPECT_FAST(bit_depth == 8);

  //  int number_of_passes{png_set_interlace_handling(png_ptr)};
  png_read_update_info(png_ptr, info_ptr);

  if (setjmp(png_jmpbuf(png_ptr))) {
    status.SetError(il::Error::Undefined);
    IL_SET_SOURCE(status);
    fclose(fp);
    return image;
  }

  png_bytep *row_pointers{(png_bytep *)malloc(sizeof(png_bytep) * height)};
  unsigned long nb_colors{png_get_rowbytes(png_ptr, info_ptr) / width};
  for (int ky = 0; ky < height; ++ky) {
    row_pointers[ky] = (png_byte *)malloc(png_get_rowbytes(png_ptr, info_ptr));
  }

  png_read_image(png_ptr, row_pointers);

  image.Resize(static_cast<il::int_t>(height), static_cast<il::int_t>(width),
               static_cast<il::int_t>(nb_colors));
  for (int ky = 0; ky < height; ++ky) {
    png_bytep row{row_pointers[ky]};
    for (int kx = 0; kx < width; ++kx) {
      for (int kc = 0; kc < image.size(2); ++kc) {
        image(ky, kx, kc) = row[nb_colors * kx + kc];
      }
    }
  }

  png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
  for (int ky = 0; ky < image.size(0); ky++) {
    free(row_pointers[ky]);
  }
  free(row_pointers);

  fclose(fp);

  status.SetOk();

  return image;
}

void savePng(const il::Array3C<unsigned char> &image,
             const il::String &filename, il::io_t, il::Status &status) {
  FILE *fp{fopen(filename.asCString(), "wb")};
  if (fp == nullptr) {
    status.SetError(il::Error::FilesystemFileNotFound);
    IL_SET_SOURCE(status);
    return;
  }

  png_structp png_ptr{png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr,
                                              nullptr, nullptr)};
  if (png_ptr == nullptr) {
    status.SetError(il::Error::Undefined);
    IL_SET_SOURCE(status);
    return;
  }

  png_infop info_ptr{png_create_info_struct(png_ptr)};
  if (info_ptr == nullptr) {
    status.SetError(il::Error::Undefined);
    IL_SET_SOURCE(status);
    return;
  }
  if (setjmp(png_jmpbuf(png_ptr))) {
    status.SetError(il::Error::Undefined);
    IL_SET_SOURCE(status);
    return;
  }

  png_init_io(png_ptr, fp);
  if (setjmp(png_jmpbuf(png_ptr))) {
    status.SetError(il::Error::Undefined);
    IL_SET_SOURCE(status);
    return;
  }

  const png_uint_32 width{static_cast<png_uint_32>(image.size(1))};
  const png_uint_32 height{static_cast<png_uint_32>(image.size(0))};
  const int bit_depth = 8;
  int color_type = 0;
  switch (image.size(2)) {
    case 3:
      color_type = 2;
      break;
    default:
      IL_EXPECT_FAST(false);
  }
  png_set_IHDR(png_ptr, info_ptr, width, height, bit_depth, color_type,
               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE,
               PNG_FILTER_TYPE_BASE);

  png_write_info(png_ptr, info_ptr);

  /* write bytes */
  if (setjmp(png_jmpbuf(png_ptr))) {
    status.SetError(il::Error::Undefined);
    IL_SET_SOURCE(status);
    return;
  }

  png_bytep *row_pointers{(png_bytep *)malloc(sizeof(png_bytep) * height)};
  for (il::int_t ky = 0; ky < image.size(0); ++ky) {
    row_pointers[ky] = (png_byte *)malloc(png_get_rowbytes(png_ptr, info_ptr));
  }

  for (il::int_t ky = 0; ky < image.size(0); ++ky) {
    png_bytep row{row_pointers[ky]};
    for (il::int_t kx = 0; kx < image.size(1); ++kx) {
      for (il::int_t kc = 0; kc < image.size(2); ++kc) {
        row[image.size(2) * kx + kc] = image(ky, kx, kc);
      }
    }
  }

  png_write_image(png_ptr, row_pointers);

  /* end write */
  if (setjmp(png_jmpbuf(png_ptr))) {
    status.SetError(il::Error::Undefined);
    IL_SET_SOURCE(status);
    return;
  }

  png_write_end(png_ptr, nullptr);

  for (il::int_t ky = 0; ky < image.size(0); ky++) {
    free(row_pointers[ky]);
  }
  free(row_pointers);

  fclose(fp);
  status.SetOk();
}
}  // namespace il
