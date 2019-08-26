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

#include <il/io/numpy/numpy.h>

#include <il/String.h>
#include <il/container/string/algorithmString.h>

namespace il {

NumpyInfo getNumpyInfo(il::io_t, std::FILE* fp, il::Status& status) {
  NumpyInfo numpy_info;

  char first_buffer[11];
  first_buffer[10] = '\0';
  il::StringView buffer{il::StringType::Byte, first_buffer, 10};

  // Read the first 10 bytes of the files. It should contain:
  // - The magic string "\x93NUMPY"
  // - The major version number
  // - The minor version number
  // - The number of bytes for the header length
  //
  std::size_t count = 10;
  const std::size_t read = fread(first_buffer, sizeof(char), count, fp);
  if (read != count) {
    status.SetError(il::Error::BinaryFileWrongFormat);
    IL_SET_SOURCE(status);
    return numpy_info;
  }
  if (!(buffer.subview(0, 6) == "\x93NUMPY")) {
    status.SetError(il::Error::BinaryFileWrongFormat);
    IL_SET_SOURCE(status);
    return numpy_info;
  }
  char major_version = buffer[6];
  char minor_version = buffer[7];
  unsigned short header_length =
      *reinterpret_cast<unsigned short*>(first_buffer + 8);
  if (major_version != 1 || minor_version != 0) {
    status.SetError(il::Error::BinaryFileWrongFormat);
    IL_SET_SOURCE(status);
    return numpy_info;
  }

  // Read the header
  //
  il::Array<char> second_buffer{header_length + 1};
  StringView header =
      StringView{il::StringType::Byte, second_buffer.Data(), header_length + 1};
  char* success = fgets(second_buffer.Data(), header_length + 1, fp);
  if (success == nullptr || !(header[header.size() - 2] == '\n')) {
    status.SetError(il::Error::BinaryFileWrongFormat);
    IL_SET_SOURCE(status);
    return numpy_info;
  }

  // Read the endianness, the type of the Numpy array, and the byte size
  //
  const il::int_t i4 = il::search("descr", header);
  if (i4 == -1 || i4 + 12 >= header.size()) {
    status.SetError(il::Error::BinaryFileWrongFormat);
    IL_SET_SOURCE(status);
    return numpy_info;
  }
  const bool little_endian = header[i4 + 9] == '<' || header[i4 + 9] == '|';
  if (!little_endian) {
    status.SetError(il::Error::BinaryFileWrongFormat);
    IL_SET_SOURCE(status);
    return numpy_info;
  }

  StringView type_string = header.subview(i4 + 10, header.size());
  const il::int_t i5 = il::search("'", type_string);
  numpy_info.type = il::String{il::StringType::Byte, type_string.data(), i5};

  // Read the ordering for multidimensional arrays
  //
  const il::int_t i0 = il::search("fortran_order", header);
  if (i0 == -1 || i0 + 20 > header.size()) {
    status.SetError(il::Error::BinaryFileWrongFormat);
    IL_SET_SOURCE(status);
    return numpy_info;
  }

  StringView fortran_order_string = header.subview(i0 + 16, i0 + 20);
  numpy_info.fortran_order = (fortran_order_string == "True") ? true : false;

  // Read the dimensions
  //
  const il::int_t i1 = il::search("(", header);
  const il::int_t i2 = il::search(")", header);
  if (i1 == -1 || i2 == -1 || i2 - i1 <= 1) {
    status.SetError(il::Error::BinaryFileWrongFormat);
    IL_SET_SOURCE(status);
    return numpy_info;
  }
  StringView shape_string = header.subview(i1 + 1, i2);
  if (shape_string.back(0) == ',') {
    numpy_info.shape.Resize(1);
  } else {
    const il::int_t n_dim = il::count(',', shape_string) + 1;
    numpy_info.shape.Resize(n_dim);
  }
  il::int_t i = 0;
  while (true) {
    numpy_info.shape[i] = std::atoll(shape_string.asCString());
    if (i == numpy_info.shape.size() - 1) {
      break;
    }
    const il::int_t i3 = il::search(",", shape_string);
    shape_string = shape_string.subview(i3 + 1, shape_string.size());
    ++i;
  }

  status.SetOk();
  return numpy_info;
}

void saveNumpyInfo(const NumpyInfo& numpy_info, il::io_t, std::FILE* fp,
                   il::Status& status) {
  il::String header{};
  header.Append("{'descr': '");
  // Little endian
  header.Append("<");
  // type of the array
  header.Append(numpy_info.type);
  // ordering
  header.Append("', 'fortran_order': ");
  if (numpy_info.fortran_order) {
    header.Append("True");
  } else {
    header.Append("False");
  }
  // shape of the array
  // The buffer can hold enough digits for any 64-bit integer
  char buffer[21];
  header.Append(", 'shape': (");
  std::sprintf(buffer, "%td", numpy_info.shape[0]);
  header.Append(il::StringType::Byte, buffer, il::size(buffer));
  for (il::int_t i = 1; i < numpy_info.shape.size(); ++i) {
    header.Append(", ");
    std::sprintf(buffer, "%td", numpy_info.shape[i]);
    header.Append(il::StringType::Byte, buffer, il::size(buffer));
  }
  if (numpy_info.shape.size() == 1) {
    header.Append(",");
  }
  header.Append("), }");
  il::int_t remainder = 64 - (10 + 1 + header.size()) % 64;
  header.Append(remainder, ' ');
  header.Append('\n');

  il::String magic{};
  magic.Append("\x93NUMPY");
  // Numpy format major version
  magic.Append(static_cast<char>(0x01));
  // Numpy format minor version
  magic.Append(static_cast<char>(0x00));
  // Size of the header
  unsigned short short_int = static_cast<unsigned short>(header.size());
  magic.Append(il::StringType::Byte, reinterpret_cast<char*>(&short_int), 2);
  magic.Append(header);

  std::size_t written = std::fwrite(magic.asCString(), sizeof(char),
                                    static_cast<std::size_t>(magic.size()), fp);
  if (static_cast<il::int_t>(written) != magic.size()) {
    status.SetError(il::Error::FilesystemCanNotWriteToFile);
    IL_SET_SOURCE(status);
    return;
  }

  status.SetOk();
}
}  // namespace il
